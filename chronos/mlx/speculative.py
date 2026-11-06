"""Recurrent-depth self-speculative decoding for the MLX model (Torch parity).

Same algorithm and loop invariant as ``chronos.runtime.speculative``:

* draft γ tokens with the same weights at ``draft_depth`` < full depth;
* verify them with one full-depth forward over ``[pending, d1..dγ]``
  (weights and experts are read once for γ+1 positions);
* greedy output is token-for-token identical to full-depth greedy decoding,
  sampled output uses rejection sampling and keeps the full-depth distribution;
* the persistent cache is only extended by verify forwards and then truncated
  to the accepted prefix.  MLX arrays are immutable, so the draft extends a
  separate list of entries and the committed cache is never modified.

Only depth-independent cache layouts (``shared_first`` / ``first_only``) can
share one KV stream between draft and verify; ``per_iteration`` falls back to
plain decoding.
"""
from __future__ import annotations

import time

import mlx.core as mx

from chronos.runtime.speculative import SpecStats


def cache_length(cache) -> int:
    """Absolute token count of an MLX Chronos cache (MLA ``(c_kv,...)`` or SWA ``(k, v, end)``)."""
    for entry in cache or ():
        if entry is not None:
            return int(entry[2]) if len(entry) > 2 else int(entry[0].shape[1])
    return 0


def _truncate_entry(entry, length: int):
    if entry is None:
        return None
    if len(entry) <= 2:                                   # MLA: (c_kv,) or (c_kv, k_rope)
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


def _cache_arrays(cache) -> list:
    """Every array inside a cache (plain, int8 ``QuantizedKVMLX`` or SWA end offsets skipped)."""
    out = []
    for entry in cache or ():
        for t in entry or ():
            if isinstance(t, mx.array):
                out.append(t)
            elif hasattr(t, "arrays"):
                out.extend(t.arrays())
    return out


def set_cache_slack(model, slack: int) -> None:
    """Let every sliding-window layer keep ``slack`` extra entries for rollback."""
    from chronos.mlx.attention import SlidingWindowAttentionMLX
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, SlidingWindowAttentionMLX):
            object.__setattr__(attn, "cache_slack", max(int(getattr(attn, "cache_slack", 0)), int(slack)))


def _probs(logits: mx.array, temperature: float, top_k: int, top_p: float) -> mx.array:
    logits = logits.astype(mx.float32) / max(float(temperature), 1e-6)
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        kth = mx.sort(logits)[-int(top_k)]
        logits = mx.where(logits < kth, -mx.inf, logits)
    if top_p < 1.0:
        order = mx.argsort(-logits)
        sorted_logits = logits[order]
        cum = mx.cumsum(mx.softmax(sorted_logits), axis=-1)
        drop_sorted = (cum - mx.softmax(sorted_logits)) >= top_p     # keep the first token always
        drop = drop_sorted[mx.argsort(order)]                       # back to vocabulary order
        logits = mx.where(drop, -mx.inf, logits)
    return mx.softmax(logits)


def _categorical(p: mx.array, key) -> int:
    return int(mx.random.categorical(mx.log(mx.maximum(p, 1e-30)), key=key).item())


def rejection_step(p: mx.array, q: mx.array, token: int, key) -> tuple[bool, int]:
    """Accept ``token`` (drawn from ``q``) with prob. min(1, p/q), else draw from norm(max(p − q, 0))."""
    k1, k2 = mx.random.split(key)
    ratio = min(1.0, float(p[token].item()) / max(float(q[token].item()), 1e-12))
    if float(mx.random.uniform(key=k1).item()) <= ratio:
        return True, int(token)
    residual = mx.maximum(p - q, 0.0)
    total = float(residual.sum().item())
    return False, _categorical(residual / total if total > 0 else p, k2)


def speculative_tokens(model, input_ids, stats: SpecStats, *, max_new_tokens: int = 128, draft_depth: int = 1,
                       gamma: int = 4, adaptive_gamma: bool = True, gamma_max: int = 8, do_sample: bool = False,
                       temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, eos_token_id=None,
                       seed: int | None = None, forward_fn=None, should_stop=None):
    """Yield generated token ids one by one (batch size 1), filling ``stats``.

    Same loop invariant as the Torch version: ``pending`` is emitted but not
    yet cached; one full-depth forward over ``[pending, d1..dγ]`` caches it
    and verifies the drafts, and the cache is truncated to the accepted
    prefix.  ``forward_fn(ids, cache, depth) -> (logits, lookahead, cache)``
    lets a runtime add expert masks / prefetch around every forward."""
    from chronos.config_schema import resolve_recurrent_attention_mode
    ids = mx.array(input_ids, dtype=mx.int32)
    if ids.ndim == 1:
        ids = ids[None]
    if ids.shape[0] != 1:
        raise ValueError("speculative decoding supports batch size 1")
    cfg = model.config
    full_depth = int(getattr(cfg, "max_inference_depth", None) or cfg.recurrent_depth)
    draft_depth = max(1, min(int(draft_depth), full_depth))
    recurrent = bool(getattr(cfg, "recurrent_enabled", False)) and int(cfg.recurrent_depth) > 1
    if not recurrent or resolve_recurrent_attention_mode(cfg) == "per_iteration":
        draft_depth = full_depth                            # no shared KV stream → plain decoding
    eos = set([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
    gamma_max = max(1, int(gamma_max))
    gamma = max(1, min(int(gamma), gamma_max))
    set_cache_slack(model, gamma_max + 1)
    key = mx.random.key(int(seed) if seed is not None else int(time.time_ns() % (2**31)))
    t0 = time.perf_counter()
    if forward_fn is None:
        def forward_fn(x, cache, depth):
            return model(x, cache=cache, recurrent_depth=depth)

    def next_key():
        nonlocal key
        key, sub = mx.random.split(key)
        return sub

    def dist(row):
        return _probs(row, temperature, top_k, top_p)

    def pick(row):
        if not do_sample:
            return int(mx.argmax(row).item())
        return _categorical(dist(row), next_key())

    try:
        logits, _, cache = forward_fn(ids, None, None)
        mx.eval(logits, *_cache_arrays(cache))
        pending = pick(logits[0, -1])
        stats.generated += 1
        yield pending
        while stats.generated < max_new_tokens and pending not in eos:
            if should_stop is not None and should_stop():
                break
            committed = cache_length(cache)
            room = max_new_tokens - stats.generated
            draft, draft_q, dcache, last = [pending], [], cache, pending
            if draft_depth < full_depth:
                for _ in range(min(gamma, room - 1) if room > 1 else 0):
                    dl, _, dcache = forward_fn(mx.array([[last]], dtype=mx.int32), dcache, draft_depth)
                    stats.draft_calls += 1
                    row = dl[0, -1]
                    last = pick(row)                        # .item() evaluates the draft step
                    if do_sample:
                        draft_q.append(dist(row))
                    draft.append(last)
                    if last in eos:
                        break
            vlogits, _, vcache = forward_fn(mx.array([draft], dtype=mx.int32), cache, None)
            stats.verify_calls += 1
            rows = vlogits[0]                               # rows[i] predicts the token after draft[i]
            if not do_sample:
                targets = [int(t) for t in mx.argmax(rows, axis=-1).tolist()]   # one host sync per round
            n_ok, nxt, stop = 0, None, False
            for i in range(1, len(draft)):
                tok = draft[i]
                if not do_sample:
                    ok, correction = targets[i - 1] == tok, targets[i - 1]
                else:
                    ok, correction = rejection_step(dist(rows[i - 1]), draft_q[i - 1], tok, next_key())
                if not ok:
                    nxt = correction
                    break
                n_ok += 1
                stats.generated += 1
                yield tok
                if tok in eos or stats.generated >= max_new_tokens:
                    stop = True
                    break
            stats.proposed += len(draft) - 1
            stats.accepted += n_ok
            cache = truncate_cache(vcache, committed + 1 + n_ok)
            mx.eval(*_cache_arrays(cache))
            if stop:
                break
            if nxt is None:                                 # every draft accepted: bonus token
                nxt = targets[len(draft) - 1] if not do_sample else pick(rows[len(draft) - 1])
            pending = nxt
            stats.generated += 1
            yield pending
            if adaptive_gamma and len(draft) > 1:
                rate = n_ok / (len(draft) - 1)
                if rate >= 0.8:
                    gamma = min(gamma_max, gamma + 1)
                elif rate < 0.4:
                    gamma = max(1, gamma - 1)
                stats.gamma_history.append(gamma)
    finally:
        stats.elapsed_s = time.perf_counter() - t0


def speculative_generate(model, input_ids, *, on_token=None, **kwargs):
    """Return ``(prompt + generated ids as list[int], SpecStats)``; kwargs as ``speculative_tokens``."""
    stats = SpecStats()
    ids = mx.array(input_ids, dtype=mx.int32)
    tokens = [int(t) for t in (ids if ids.ndim == 1 else ids[0]).tolist()]
    for tok in speculative_tokens(model, ids, stats, **kwargs):
        tokens.append(tok)
        if on_token:
            on_token(tok)
    return tokens, stats
