"""Recurrent-depth self-speculative decoding (no extra parameters).

Draft: the same model run with fewer loop iterations (``draft_depth``)
proposes γ tokens.  Verify: one full-depth forward over the γ+1 positions
scores them in a single batched pass (weights/experts read once).  Greedy
outputs are token-for-token identical to full-depth greedy decoding; sampled
outputs use standard rejection sampling, which preserves the full-depth
distribution exactly.

Cache contract: draft KV is written to a throw-away cache; the persistent
cache is only ever extended by verified forwards, then truncated to the
accepted prefix (MLA slices ``c_kv``; sliding-window layers keep
``window + γ_max`` entries so a rollback never loses in-window keys).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


def _truncate_entry(entry, length: int):
    if entry is None:
        return None
    if len(entry) <= 2:  # MLA latent cache (c_kv,) or (c_kv, k_rope); tensors or QuantizedKV
        return tuple(t[:, :length] for t in entry)
    k, v, end = entry
    drop = int(end) - length
    if drop < 0:
        raise ValueError("cannot extend a cache by truncation")
    if drop == 0:
        return entry
    if drop > k.shape[1]:
        raise ValueError(f"rollback of {drop} exceeds retained sliding-window cache {k.shape[1]}")
    return (k[:, :k.shape[1] - drop], v[:, :v.shape[1] - drop], length)


def truncate_cache(cache, length: int):
    return [_truncate_entry(e, length) for e in cache]


def set_cache_slack(model, slack: int) -> None:
    from chronos.model.hybrid_attention import SlidingWindowAttention
    for module in model.modules():
        if isinstance(module, SlidingWindowAttention):
            module.cache_slack = max(int(getattr(module, "cache_slack", 0)), int(slack))


def speculative_supported(cfg) -> bool:
    """True when a shallow draft can share the KV stream with full-depth verify.

    Needs a recurrent core deeper than one iteration and a KV layout whose
    prefix does not depend on depth (``shared_first`` / ``first_only``).
    ``per_iteration`` writes one KV slot per virtual layer, so callers fall
    back to plain decoding (``draft_depth = full depth``), as the MLX loop does.
    """
    from chronos.config_schema import resolve_recurrent_attention_mode
    depth = int(getattr(cfg, "max_inference_depth", None) or getattr(cfg, "recurrent_depth", 1) or 1)
    recurrent = bool(getattr(cfg, "recurrent_enabled", True)) and depth > 1
    return recurrent and resolve_recurrent_attention_mode(cfg) != "per_iteration"


@dataclass
class SpecStats:
    proposed: int = 0
    accepted: int = 0
    verify_calls: int = 0
    draft_calls: int = 0
    generated: int = 0
    elapsed_s: float = 0.0
    gamma_history: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def as_dict(self) -> dict:
        return {"proposed": self.proposed, "accepted": self.accepted,
                "acceptance_rate": round(self.acceptance_rate, 4),
                "verify_calls": self.verify_calls, "draft_calls": self.draft_calls,
                "generated": self.generated,
                "tokens_per_verify": round(self.generated / max(self.verify_calls, 1), 3),
                "tokens_per_s": round(self.generated / self.elapsed_s, 2) if self.elapsed_s else None}


def _probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.float() / max(temperature, 1e-6)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(int(top_k), logits.shape[-1]))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        logits = logits.masked_fill(mask.scatter(-1, idx, mask), float("-inf"))
    return torch.softmax(logits, dim=-1)


def rejection_step(p: torch.Tensor, q: torch.Tensor, token: int, generator=None) -> tuple[bool, int]:
    """Speculative-sampling acceptance (Leviathan et al. / Chen et al. 2023).

    ``token`` was drawn from the draft distribution ``q``.  Accept with
    probability min(1, p/q); otherwise draw from norm(max(p − q, 0)).  The
    returned token is distributed exactly as ``p``.
    """
    ratio = (p[token] / q[token].clamp_min(1e-12)).clamp(max=1.0)
    if bool(torch.rand((), generator=generator, device=p.device) <= ratio):
        return True, int(token)
    residual = (p - q).clamp_min(0)
    total = residual.sum()
    return False, int(torch.multinomial(residual / total if total > 0 else p, 1, generator=generator))


@torch.no_grad()      # not inference_mode: callers may page experts in and reuse them later
def speculative_generate(model, input_ids: torch.Tensor, *, max_new_tokens: int = 128,
                         draft_depth: int = 1, gamma: int = 4, adaptive_gamma: bool = True,
                         gamma_max: int = 8, do_sample: bool = False, temperature: float = 1.0,
                         top_k: int = 0, top_p: float = 1.0, eos_token_id=None,
                         generator: torch.Generator | None = None, on_token=None,
                         forward_fn=None, should_stop=None):
    """Return (output_ids [1, T], SpecStats).  Batch size 1 (serving path).

    Loop invariant: ``pending`` is final (already emitted) but not yet in the
    persistent cache.  Each round drafts from ``pending`` at shallow depth,
    then one full-depth forward over ``[pending, d1..dγ]`` both writes
    ``pending`` into the cache and verifies the drafts.  A rejection's
    correction (or the all-accepted bonus token) becomes the next pending
    token, so no extra forward is ever needed.
    """
    from chronos.model.model_chronos import cache_sequence_length
    if input_ids.shape[0] != 1:
        raise ValueError("speculative_generate supports batch size 1")
    cfg = model.config
    full_depth = int(getattr(cfg, "max_inference_depth", None) or cfg.recurrent_depth)
    draft_depth = max(1, min(int(draft_depth), full_depth))
    eos = set([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
    gamma_max = max(1, int(gamma_max))
    gamma = max(1, min(int(gamma), gamma_max))
    set_cache_slack(model, gamma_max + 1)
    device = input_ids.device
    stats, t0 = SpecStats(), time.perf_counter()
    tokens = input_ids[0].tolist()
    # ``forward_fn(ids, cache, depth)`` lets a runtime wrap every forward with
    # expert residency work (SSD lazy loading, lookahead prefetch, masks).
    if forward_fn is None:
        def forward_fn(ids, cache, depth):
            return model(ids, past_key_values=cache, use_cache=True, recurrent_depth=depth)

    def dist(row):
        return _probs(row, temperature, top_k, top_p)

    def pick(row):
        if not do_sample:
            return int(torch.argmax(row))
        return int(torch.multinomial(dist(row), 1, generator=generator))

    def emit(tok):
        tokens.append(tok)
        stats.generated += 1
        if on_token:
            on_token(tok)

    out = forward_fn(input_ids, None, None)
    cache = out.past_key_values
    pending = pick(out.logits[0, -1])
    emit(pending)
    while stats.generated < max_new_tokens and pending not in eos:
        if should_stop is not None and should_stop():
            break
        committed = cache_sequence_length(cache)
        room = max_new_tokens - stats.generated          # tokens we may still emit
        draft, draft_q, dcache, last = [pending], [], cache, pending
        if draft_depth < full_depth:
            for _ in range(min(gamma, room - 1) if room > 1 else 0):
                d = forward_fn(torch.tensor([[last]], device=device), dcache, draft_depth)
                stats.draft_calls += 1
                dcache = d.past_key_values
                row = d.logits[0, -1]
                last = pick(row)
                if do_sample:
                    draft_q.append(dist(row))
                draft.append(last)
                if last in eos:
                    break
        v = forward_fn(torch.tensor([draft], device=device), cache, None)
        stats.verify_calls += 1
        logits = v.logits[0]                               # logits[i] predicts the token after draft[i]
        n_ok, nxt, stop = 0, None, False
        for i in range(1, len(draft)):
            row, tok = logits[i - 1], draft[i]
            if not do_sample:
                target = int(torch.argmax(row))
                ok = target == tok
                correction = target
            else:
                ok, correction = rejection_step(dist(row), draft_q[i - 1], tok, generator)
            if not ok:
                nxt = correction
                break
            n_ok += 1
            emit(tok)
            if tok in eos or stats.generated >= max_new_tokens:
                stop = True
                break
        stats.proposed += len(draft) - 1
        stats.accepted += n_ok
        cache = truncate_cache(v.past_key_values, committed + 1 + n_ok)
        if stop:
            break
        if nxt is None:                                  # every draft accepted: bonus token
            nxt = pick(logits[len(draft) - 1])
        pending = nxt
        emit(pending)
        if adaptive_gamma and len(draft) > 1:
            rate = n_ok / (len(draft) - 1)
            if rate >= 0.8:
                gamma = min(gamma_max, gamma + 1)
            elif rate < 0.4:
                gamma = max(1, gamma - 1)
            stats.gamma_history.append(gamma)
    stats.elapsed_s = time.perf_counter() - t0
    return torch.tensor([tokens], device=device), stats
