"""Adaptive training planner.

``plan_training(PlanRequest)`` turns a handful of user targets (total
parameters, MoE on/off, expert count, active parameters *or* active experts,
memory-saving level, optional dataset paths, hardware memory) into a complete
Chronos v3 configuration plus independent hyperparameters per training stage.

How each number is derived (the report repeats this per value):

* **Architecture** — closed-form parameter counts (``_count``) searched over
  width / depth / expert width, then the best candidate is re-counted on the
  ``meta`` device with ``factory.count_parameters`` (the allocation truth).
* **Pretrain LR and batch** — DeepSeek LLM hyperparameter scaling laws
  (arXiv 2401.02954, §3.1): ``η_opt = 0.3118·C^-0.125`` and
  ``B_opt = 0.2920·C^0.3271`` tokens, with ``C = M·D``, where ``M`` is the
  non-embedding FLOPs/token ``72·L·h² + 12·L·h·s`` (L counts executed
  virtual layers) and ``D`` the planned token count.  The law was fitted on
  dense models over C≈1e17–2e19; below / outside that range it is an
  extrapolation, so results are clamped and labelled ``estimated``.
* **Later stages** — conventional ratios to the pretrain LR (SFT ≈ 1/3,
  preference stages ≈ 1/10–1/20, distill ≈ 1/3); these are heuristics.
* **Memory** — weights / optimizer state from the parameter count; the
  activation term uses ``ACT_BYTES_PER_TOKEN_LAYER_H`` calibrated on the live
  50M MLX run (measured peak 8.64 GB at 8192 micro-tokens, H=384,
  18 virtual layers, bf16).  Treat it as an order-of-magnitude estimate.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, field

STAGES = ("pretrain", "sft", "dpo", "orpo", "grpo", "distill")
# Keys a stage may override; the Train tab merges ``stage_overrides[mode]``.
STAGE_KEYS = ("learning_rate", "batch_size", "accumulation_steps", "epochs", "max_steps", "max_seq_len",
              "weight_decay", "grad_clip", "save_interval", "lambda_router_anchor", "beta", "alpha",
              "lambda_or", "temperature", "num_generations", "max_gen_len")
MEMORY_LEVELS = ("none", "balanced", "aggressive", "extreme")
# Activation bytes per (token · executed layer · hidden unit), MLX bf16,
# derived from the measured 50M run peak (see module docstring).
ACT_BYTES_PER_TOKEN_LAYER_H = 120.0
DEEPSEEK_FIT_RANGE = (1e17, 2e19)


@dataclass
class PlanRequest:
    target_total: float                      # parameters, e.g. 50e6
    use_moe: bool = True
    num_experts: int = 32
    active_params: float | None = None       # either this ...
    active_experts: int | None = None        # ... or this (top-k); active_params wins if both
    memory_saving: str = "none"              # MEMORY_LEVELS
    data_paths: tuple = ()                   # pretrain + sft + preference files/dirs
    vocab_size: int | None = None            # None: tokenizer report / dataset, else 16384
    seq_len: int | None = None               # None: from the dataset profile
    ram_gb: float | None = None              # None: detected
    token_budget: float | None = None        # pretrain tokens; None: dataset tokens (one pass)
    recurrent_depth: int = 4
    dtype: str = "bfloat16"


@dataclass
class DatasetProfile:
    files: list = field(default_factory=list)
    kinds: dict = field(default_factory=dict)          # stage -> [files]
    pretrain_tokens: float = 0.0
    pretrain_tokens_source: str = "none"               # measured | estimated | none
    sft_records: float = 0.0
    pref_records: float = 0.0
    rl_records: float = 0.0
    chars_per_token: float | None = None
    seq_p95_tokens: int | None = None                  # sampled records (estimated)
    seq_cap_tokens: int | None = None                  # exact cap from a shard manifest
    vocab_size: int | None = None
    tokenizer_dir: str = ""
    notes: list = field(default_factory=list)


# ── dataset profile ────────────────────────────────────────────────────
# Fallback chars/token when no tokenizer report is available (estimated,
# typical byte-level BPE at 16–32k vocab); replaced by the report values.
_DEFAULT_CPT = {"zh": 1.4, "ja": 1.2, "ko": 1.5, "latin": 3.8, "cyrillic": 3.2, "other": 2.5}
_SAMPLE_RECORDS = 2000


def _record_kind(rec) -> str:
    if isinstance(rec, dict):
        if "chosen" in rec and "rejected" in rec:
            return "pref"
        turns = rec.get("conversations") if isinstance(rec.get("conversations"), list) else rec.get("messages")
        if isinstance(turns, list):
            # GRPO prompt files end in an empty assistant turn to be generated.
            last = turns[-1] if turns and isinstance(turns[-1], dict) else {}
            if last.get("role") == "assistant" and not str(last.get("content") or "").strip():
                return "rl"
            return "sft"
        if "prompt" in rec and ("answer" in rec or "reward" in rec or "solution" in rec):
            return "rl"
    return "pretrain"


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _tokenizer_report(paths) -> tuple[dict, str]:
    """First ``tokenizer_report.json`` next to / referenced by the inputs."""
    for p in paths:
        for cand in (p, os.path.dirname(p)):
            rep = os.path.join(cand, "tokenizer_report.json")
            if os.path.isfile(rep):
                return _read_json(rep), cand
            man = _read_json(os.path.join(cand, "shards_manifest.json"))
            tdir = man.get("tokenizer_dir")
            if not tdir:
                continue
            # Relative paths in a manifest are relative to the repo (cwd) or to the manifest.
            for base in ("", cand):
                full = os.path.abspath(os.path.join(base, tdir))
                if os.path.isfile(os.path.join(full, "tokenizer_report.json")):
                    return _read_json(os.path.join(full, "tokenizer_report.json")), full
    return {}, ""


def _count_records(path: str, sampled: int, sample_bytes: int) -> tuple[float, str]:
    """Exact count for parquet / small text files, size-extrapolated otherwise."""
    size = os.path.getsize(path)
    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            return float(pq.ParquetFile(path).metadata.num_rows), "measured"
        except Exception:
            pass
    if path.endswith((".jsonl", ".txt", ".csv", ".tsv")) and size <= 512 << 20:
        with open(path, "rb") as fh:
            lines = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))
        return float(max(lines - (1 if path.endswith((".csv", ".tsv")) else 0), sampled)), "measured"
    if sampled and sample_bytes:
        return float(sampled) * size / sample_bytes, "estimated"
    return float(sampled), "estimated"


def profile_datasets(paths, *, max_files: int = 64) -> DatasetProfile:
    """Profile pretrain / SFT / preference files for the planner.

    Prefers exact counts from a Chronos ``shards_manifest.json`` (token
    shards) and the tokenizer report; otherwise samples records, maps each
    text's script bucket to chars/token, and extrapolates by file size.
    """
    from chronos.tokenizer.extract import _texts_from_record, expand, iter_records, script_bucket
    prof = DatasetProfile()
    paths = [os.path.expanduser(str(p)) for p in (paths or ()) if str(p).strip()]
    report, prof.tokenizer_dir = _tokenizer_report(paths)
    cpt_by_bucket = dict(_DEFAULT_CPT)
    for bucket, stats in (report.get("heldout_eval") or {}).items():
        if isinstance(stats, dict) and stats.get("chars_per_token") and int(stats.get("texts", 0)) >= 100:
            cpt_by_bucket[bucket] = float(stats["chars_per_token"])
    if report.get("vocab_size"):
        prof.vocab_size = int(report["vocab_size"])
    for p in paths:
        for cand in (p, os.path.dirname(p)):
            man = _read_json(os.path.join(cand, "shards_manifest.json"))
            if man.get("pretrain_train_tokens"):
                prof.pretrain_tokens = float(man["pretrain_train_tokens"])
                prof.pretrain_tokens_source = "measured"
                prof.sft_records = float(man.get("sft_train_records") or 0)
                if man.get("sft_max_len"):
                    prof.seq_cap_tokens = int(man["sft_max_len"])
                prof.notes.append(f"token counts from {os.path.join(cand, 'shards_manifest.json')}")
                break
        if prof.pretrain_tokens_source == "measured":
            break
    if prof.pretrain_tokens_source == "measured":
        return prof
    files = expand(paths)[:max_files]
    prof.files = files
    tok_lens, chars, tokens = [], 0, 0.0
    for path in files:
        kind_counts, sampled, sample_bytes = {}, 0, 0
        file_tokens = 0.0
        try:
            for rec in iter_records(path):
                if sampled >= _SAMPLE_RECORDS:
                    break
                sampled += 1
                kind = _record_kind(rec)
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                rec_tokens = 0.0
                for text in _texts_from_record(rec, None):
                    n = len(text)
                    sample_bytes += len(text.encode("utf-8", "replace")) + 8
                    chars += n
                    rec_tokens += n / cpt_by_bucket.get(script_bucket(text), 2.5)
                file_tokens += rec_tokens
                tok_lens.append(rec_tokens)
        except Exception as exc:                    # unreadable file: report, keep going
            prof.notes.append(f"skipped {os.path.basename(path)}: {type(exc).__name__}")
            continue
        if not sampled:
            continue
        kind = max(kind_counts, key=kind_counts.get)
        prof.kinds.setdefault(kind, []).append(path)
        records, how = _count_records(path, sampled, sample_bytes)
        tokens += file_tokens / sampled * records if kind == "pretrain" else 0.0
        if kind == "sft":
            prof.sft_records += records
        elif kind == "pref":
            prof.pref_records += records
        elif kind == "rl":
            prof.rl_records += records
        if how == "estimated":
            prof.notes.append(f"{os.path.basename(path)}: record count extrapolated from file size")
    if tokens:
        prof.pretrain_tokens, prof.pretrain_tokens_source = tokens, "estimated"
    if tok_lens:
        tok_lens.sort()
        prof.seq_p95_tokens = int(math.ceil(tok_lens[int(0.95 * (len(tok_lens) - 1))]))
        total_tok = sum(tok_lens)
        prof.chars_per_token = round(chars / total_tok, 3) if total_tok else None
        prof.notes.append(f"median record ≈ {statistics.median(tok_lens):.0f} tokens (estimated)")
    return prof


# ── architecture search ────────────────────────────────────────────────
HEAD_DIM = 64
ROPE_DIM = 32
LOOKAHEAD_STEPS = 2


def _kv_heads(heads: int) -> int:
    kv = max(1, round(heads / 3))                 # GQA group of ~3 (recipe: 6 heads / 2 KV)
    while heads % kv:
        kv -= 1
    return kv


def _latent(h: int) -> int:
    return int(min(512, max(32, round(h / 3 / 32) * 32)))


def _attn_params(h: int, heads: int, kv: int, latent: int, mla: bool) -> int:
    d = HEAD_DIM
    if mla:                                       # mirrors ui/estimator.py::_attn_params
        return 2 * h * heads * d + h * latent + 2 * latent * kv * d + d + latent
    return 2 * h * heads * d + 2 * h * kv * d + 2 * d


def _base_params(h: int, layers: int, groups: int, depth: int, vocab: int, experts: int, use_moe: bool) -> int:
    """Everything except FFN / expert weights (embedding tied, MLA on even layers)."""
    heads = h // HEAD_DIM
    kv, latent = _kv_heads(heads), _latent(h)
    attn = sum(_attn_params(h, heads, kv, latent, i % 2 == 0) for i in range(layers))
    ph = max(1, h // 4)
    lookahead = h * ph + ph * max(experts, 1) * (LOOKAHEAD_STEPS + 1) * groups   # layer-aware predictor
    controller = (depth + groups) * h + 2 * h + 1 + groups
    routers = layers * h * experts if use_moe else 0
    return vocab * h + attn + (2 * layers + 1) * h + routers + lookahead + controller


@dataclass
class ArchCandidate:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    kv_latent_dim: int
    ffn_width: int                 # moe_intermediate_size (MoE) or intermediate_size (dense)
    num_experts: int
    num_experts_per_tok: int
    total: int
    active: int
    union_upper: int
    score: float


def search_architecture(target_total: float, *, use_moe: bool, num_experts: int, vocab_size: int,
                        depth: int, active_params: float | None = None, active_experts: int | None = None,
                        shared_experts: int = 1) -> ArchCandidate:
    """Closed-form search: FFN width is solved from the total-parameter target.

    For each width h (multiples of 64, head_dim 64), physical depth L
    (prelude 1 + groups + coda 1) and top-k, the expert width that hits
    ``target_total`` is solved analytically and rounded to a multiple of 32.
    Score = |log total/T| (+ |log active/A| when A is given) + a mild shape
    prior (h / L in [32, 128], expert width in [h/4, 2h], k ≥ 2) and weak
    tie-breakers toward h/L ≈ 64 and expert width ≈ h/2 (fine-grained).
    """
    if target_total <= 0:
        raise ValueError("target_total must be positive")
    E = max(1, int(num_experts)) if use_moe else 1
    if use_moe and active_experts and not active_params:
        k_choices = [max(1, min(E, int(active_experts)))]
    elif use_moe:
        k_choices = list(range(1, min(E, 8) + 1))
    else:
        k_choices = [1]
    best = None
    for h in range(128, 4097, 64):
        if vocab_size * h > 0.85 * target_total:
            break                                         # embedding alone exhausts the budget
        heads = h // HEAD_DIM
        kv_heads = _kv_heads(heads)
        # A prime head count collapses GQA to one KV head (MQA); prefer widths
        # whose head count has a divisor near heads/3.
        gqa_penalty = 0.03 if heads >= 4 and heads // kv_heads > 4 else 0.0
        for layers in range(3, 49):
            groups = layers - 2
            base = _base_params(h, layers, groups, depth, vocab_size, E, use_moe)
            if base >= target_total:
                break
            for k in k_choices:
                slope = layers * 3 * h * ((E + shared_experts) if use_moe else 1)
                raw = (target_total - base) / slope
                step = 32 if use_moe else 64
                for width in {max(step, int(raw // step) * step), max(step, int(math.ceil(raw / step)) * step)}:
                    total = base + slope * width
                    per = 3 * h * width
                    active = base + layers * per * (k + shared_experts) if use_moe else total
                    union = active + groups * (min(E, k * depth) - k) * per if use_moe else total
                    score = abs(math.log(total / target_total))
                    if use_moe and active_params:
                        score += abs(math.log(active / float(active_params)))
                    ratio = h / layers
                    score += 0.05 * max(0.0, math.log(32 / ratio), math.log(ratio / 128))
                    lo, hi = ((h / 4, 2 * h) if use_moe else (2 * h, 4 * h))
                    score += 0.10 * max(0.0, math.log(lo / width), math.log(width / hi))
                    # Tie-breakers among exact fits (heuristics, not fitted): h/L near 64,
                    # fine-grained experts near h/2 (E ≥ 16) or h (E < 16), dense FFN near 8/3·h.
                    centre = (h / 2 if E >= 16 else h) if use_moe else 8 * h / 3
                    score += 0.02 * abs(math.log(ratio / 64)) + 0.02 * abs(math.log(width / centre))
                    if use_moe and k == 1:
                        score += 0.02                     # top-1 loses gate gradients under norm_topk
                    score += gqa_penalty
                    if best is None or score < best.score:
                        best = ArchCandidate(h, layers, heads, kv_heads, _latent(h), width, E if use_moe else 0,
                                             k, int(total), int(active), int(union), score)
    if best is None:
        raise ValueError(f"no architecture fits {target_total:.3g} parameters with vocab {vocab_size}")
    return best


# ── memory model ───────────────────────────────────────────────────────
# memory_saving → (inference options, training options).  Every option is an
# existing Chronos flag; nothing here changes numerics during training except
# activation checkpointing (recompute) and optimizer paging (SSD round trips).
MEMORY_POLICIES = {
    "none":       dict(kv_cache_dtype="auto", mla_cache_rope=True, router_scope="per_iteration",
                       resident_fraction=1.0, ram_fraction=1.0, activation_checkpointing=False,
                       optimizer_paging=False),
    "balanced":   dict(kv_cache_dtype="int8", mla_cache_rope=False, router_scope="per_iteration",
                       resident_fraction=0.5, ram_fraction=0.75, activation_checkpointing=False,
                       optimizer_paging=False),
    "aggressive": dict(kv_cache_dtype="int8", mla_cache_rope=False, router_scope="shared",
                       resident_fraction=0.25, ram_fraction=0.5, activation_checkpointing=True,
                       optimizer_paging=False),
    "extreme":    dict(kv_cache_dtype="int8", mla_cache_rope=False, router_scope="shared",
                       resident_fraction=0.125, ram_fraction=0.25, activation_checkpointing=True,
                       optimizer_paging=True),
}
# Share of physical RAM a plan may target for the training peak (rest: OS,
# data loader, UI, other jobs).  Lowered by the memory-saving level.
_RAM_SHARE = {"none": 0.6, "balanced": 0.45, "aggressive": 0.3, "extreme": 0.2}


def detect_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:
        return 16.0


def training_peak_bytes(total_params: int, micro_tokens: int, hidden: int, executed_layers: int,
                        vocab: int, *, checkpointing: bool = False, optimizer_paging: bool = False,
                        routed_params: int = 0) -> int:
    """Estimated MLX/Torch training peak (bytes).

    * state: FP32 master + grad + Adam m, v = 16 B/param (optimizer paging
      moves the routed experts' Adam state, 8 B/param, to SSD);
    * activations: ``ACT_BYTES_PER_TOKEN_LAYER_H`` · tokens · layers · h
      (checkpointing keeps ~1 layer boundary per layer, ≈ ×0.25 here);
    * logits: FP32 logits + grad + softmax ≈ 12 B per token·vocab.
    """
    state = 16 * total_params - (8 * routed_params if optimizer_paging else 0)
    act = ACT_BYTES_PER_TOKEN_LAYER_H * micro_tokens * executed_layers * hidden
    if checkpointing:
        act *= 0.25
    return int(state + act + 12 * micro_tokens * vocab)


# ── hyperparameters ────────────────────────────────────────────────────
def deepseek_hparams(compute_flops: float) -> tuple[float, float]:
    """DeepSeek LLM (arXiv 2401.02954) eq. for η_opt and B_opt (tokens)."""
    c = max(float(compute_flops), 1.0)
    return 0.3118 * c ** -0.1250, 0.2920 * c ** 0.3271


def flops_per_token(hidden: int, executed_layers: int, seq_len: int, active_ffn_ratio: float = 1.0) -> float:
    """``M = 72·L·h² + 12·L·h·s`` (DeepSeek's non-embedding FLOPs/token).

    ``72·L·h² = 6 · 12·L·h²`` counts forward+backward over the 4h² attention
    and 8h² FFN parameters of a standard layer (FFN 4h, 2 matrices).  For a
    MoE or SwiGLU layer the FFN share is scaled by ``active_ffn_ratio`` =
    active FFN parameters / 8h²; L counts executed (virtual) layers.
    """
    L, h = executed_layers, hidden
    return (24 + 48 * active_ffn_ratio) * L * h * h + 12 * L * h * seq_len


def _pow2_floor(x: float) -> int:
    return 1 << max(0, int(math.floor(math.log2(max(1.0, x)))))


# ── plan ───────────────────────────────────────────────────────────────
def _seq_len(req: PlanRequest, prof: DatasetProfile) -> tuple[int, str]:
    if req.seq_len:
        return int(req.seq_len), "user"
    if prof.seq_cap_tokens:
        return int(prof.seq_cap_tokens), "shard manifest max length (measured)"
    if prof.seq_p95_tokens:
        s = 1 << max(7, int(math.ceil(math.log2(prof.seq_p95_tokens * 1.15))))
        return int(min(2048, s)), "dataset p95 × 1.15 → power of two (estimated)"
    return 512, "default"


def _round_lr(x: float) -> float:
    return float(f"{x:.2g}")


def plan_training(req: PlanRequest, profile: DatasetProfile | None = None, *, verify: bool = True) -> dict:
    """Return ``{"config", "stage_overrides", "longrun_phases", "report"}``.

    ``config`` is a complete Config-tab dict (v3 kernel); ``stage_overrides``
    maps each stage to its own hyperparameters (the Train tab merges the
    selected stage over ``config``); ``longrun_phases`` are MLX long-run
    ``PhaseSpec`` dicts for pretrain + SFT.
    """
    level = req.memory_saving if req.memory_saving in MEMORY_POLICIES else "none"
    policy = MEMORY_POLICIES[level]
    prof = profile if profile is not None else profile_datasets(req.data_paths)
    notes, sources = list(prof.notes), {}
    vocab = int(req.vocab_size or prof.vocab_size or 16384)
    sources["vocab_size"] = "user" if req.vocab_size else ("tokenizer report" if prof.vocab_size else "default")
    depth = max(1, int(req.recurrent_depth))
    arch = search_architecture(req.target_total, use_moe=req.use_moe, num_experts=req.num_experts,
                               vocab_size=vocab, depth=depth, active_params=req.active_params,
                               active_experts=req.active_experts)
    h, L, groups = arch.hidden_size, arch.num_hidden_layers, arch.num_hidden_layers - 2
    seq_len, sources["max_seq_len"] = _seq_len(req, prof)
    executed = L + groups * (depth - 1)                       # virtual layers at full depth
    shared = 1 if req.use_moe else 0
    if req.use_moe:
        ffn_ratio = 3 * h * arch.ffn_width * (arch.num_experts_per_tok + shared) / (8 * h * h)
    else:
        ffn_ratio = 3 * h * arch.ffn_width / (8 * h * h)

    # Token budget D and DeepSeek-law LR / batch.
    if req.token_budget:
        tokens, sources["tokens"] = float(req.token_budget), "user"
    elif prof.pretrain_tokens >= 2.0 * arch.total:
        tokens, sources["tokens"] = prof.pretrain_tokens, f"dataset, one pass ({prof.pretrain_tokens_source})"
    else:
        # Too little (or no) pretrain data to size the run: plan for 20 tokens
        # per parameter and report the shortfall instead of fitting a toy set.
        tokens, sources["tokens"] = 20.0 * arch.total, "default 20 tokens/parameter (estimated)"
        if prof.pretrain_tokens:
            notes.append(f"dataset has ≈{prof.pretrain_tokens:.3g} pretrain tokens, < 2 per parameter; "
                         f"the plan assumes {tokens:.3g} tokens (multiple epochs or more data)")
    m = flops_per_token(h, executed, seq_len, ffn_ratio)
    compute = m * tokens
    eta, batch_tokens = deepseek_hparams(compute)
    if not DEEPSEEK_FIT_RANGE[0] <= compute <= DEEPSEEK_FIT_RANGE[1] * 10:
        notes.append(f"C = {compute:.2e} FLOPs is outside the DeepSeek fit range "
                     f"{DEEPSEEK_FIT_RANGE[0]:.0e}–{DEEPSEEK_FIT_RANGE[1]:.0e}; LR/batch are extrapolated and clamped")
    # Chronos clamps: aux losses + recurrence make very high LR fragile; the
    # ceiling shrinks with width (≈ μP-style 1/h, anchored at 3e-3 for h=384).
    lr = _round_lr(min(3e-3 * min(1.0, 384 / h), max(1e-4, eta)))
    batch_tokens = min(1 << 21, max(16384, _pow2_floor(batch_tokens)))

    ram_gb = float(req.ram_gb or detect_ram_gb())
    budget = ram_gb * 2**30 * _RAM_SHARE[level]
    routed = L * arch.num_experts * 3 * h * arch.ffn_width if req.use_moe else 0
    # Largest power-of-two micro-batch (≤ 16k tokens) whose estimated peak
    # fits the level's RAM share; accumulation makes up the optimizer batch.
    micro = max(seq_len, _pow2_floor(min(batch_tokens, 16384)))
    while micro > seq_len and training_peak_bytes(
            arch.total, micro, h, executed, vocab, checkpointing=policy["activation_checkpointing"],
            optimizer_paging=policy["optimizer_paging"], routed_params=routed) > budget:
        micro //= 2
    peak = training_peak_bytes(arch.total, micro, h, executed, vocab,
                               checkpointing=policy["activation_checkpointing"],
                               optimizer_paging=policy["optimizer_paging"], routed_params=routed)
    if peak > budget:
        notes.append(f"estimated training peak {peak / 2**30:.1f} GiB exceeds the {budget / 2**30:.1f} GiB "
                     f"target even at one sequence per micro-batch; raise memory saving or lower the target")
    rows = max(1, micro // seq_len)
    accum = max(1, batch_tokens // (rows * seq_len))
    batch_tokens = rows * seq_len * accum
    updates = max(1, int(tokens // batch_tokens))

    # Independent per-stage hyperparameters.  Ratios to the pretrain LR are
    # heuristics (the 50M recipe uses SFT = pretrain/3, half the batch);
    # GRPO β = 0.04 follows DeepSeekMath (arXiv 2402.03300 §4.2); other
    # coefficients are the Chronos trainer defaults.
    def stage(lr_value, rows_value, accum_value, **extra):
        out = dict(learning_rate=_round_lr(lr_value), batch_size=int(max(1, rows_value)),
                   accumulation_steps=int(max(1, accum_value)), max_seq_len=seq_len, grad_clip=1.0)
        out.update(extra)
        return out

    epochs, pre_steps = 1, 0
    if prof.pretrain_tokens and tokens > prof.pretrain_tokens:
        epochs = max(1, int(math.ceil(tokens / prof.pretrain_tokens)))
    if req.token_budget or tokens != prof.pretrain_tokens:
        pre_steps = updates * accum                       # Train tab counts micro-steps
    micro_steps = pre_steps or updates * accum

    def save_interval(steps: float) -> int:
        """≈20 checkpoints per stage, a multiple of 10 in [100, 100000] (the UI slider)."""
        return int(min(100_000, max(100, round(steps / 20, -1))))

    save_every = save_interval(micro_steps)
    sft_epochs = 1 if prof.sft_records >= 200_000 or not prof.sft_records else (2 if prof.sft_records >= 20_000 else 3)
    pair_rows = max(1, rows // 2)                         # chosen + rejected per pair
    rl_rows = max(1, rows // 4)
    num_gen = 8 if level == "none" else (4 if level in ("balanced", "aggressive") else 2)
    # Post-training stages are sized by their own records (one micro-step per
    # `rows` records); without records fall back to a fixed 500-step cadence.
    sft_save = save_interval(prof.sft_records * sft_epochs / rows) if prof.sft_records else 500
    pref_save = save_interval(prof.pref_records / pair_rows) if prof.pref_records else 500
    rl_save = save_interval(prof.rl_records / rl_rows) if prof.rl_records else 500
    gen_len = int(min(256, max(24, (prof.seq_p95_tokens or prof.seq_cap_tokens or seq_len) // 2)))
    overrides = {
        "pretrain": stage(lr, rows, accum, epochs=epochs, max_steps=pre_steps, weight_decay=0.1,
                          save_interval=save_every, lambda_router_anchor=0.0),
        "sft": stage(lr / 3, rows, max(1, accum // 2), epochs=sft_epochs, max_steps=0, weight_decay=0.1,
                     save_interval=sft_save, lambda_router_anchor=0.0),
        "dpo": stage(max(5e-6, lr / 20), pair_rows, max(1, accum // 2), epochs=1, max_steps=0, weight_decay=0.01,
                     beta=0.1, lambda_router_anchor=0.01, save_interval=pref_save),
        "orpo": stage(max(5e-6, lr / 10), pair_rows, max(1, accum // 2), epochs=1, max_steps=0, weight_decay=0.01,
                      lambda_or=0.1, lambda_router_anchor=0.01, save_interval=pref_save),
        "grpo": stage(min(2e-5, max(1e-6, lr / 100)), rl_rows, 1, epochs=1, max_steps=0, weight_decay=0.01,
                      beta=0.04, num_generations=num_gen, temperature=1.0, max_gen_len=gen_len,
                      lambda_router_anchor=0.01, save_interval=rl_save),
        "distill": stage(lr / 3, rows, max(1, accum // 2), epochs=1, max_steps=0, weight_decay=0.1,
                         alpha=0.7, temperature=4.0, lambda_router_anchor=0.0, save_interval=sft_save),
    }

    p95 = prof.seq_p95_tokens or prof.seq_cap_tokens
    # SWA window: the typical record must fit; long documents use MLA layers.
    window = min(seq_len, max(128, 1 << int(math.ceil(math.log2(max(2, p95 // 4)))))) if p95 \
        else max(128, seq_len // 4)
    per_expert_bytes = 3 * h * arch.ffn_width * 2                # bf16 page
    expert_mb = routed * 2 / 2**20
    ram_budget = None if policy["ram_fraction"] >= 1 or not req.use_moe else \
        round(max(per_expert_bytes * L * max(2, arch.num_experts_per_tok) / 2**20,
                  expert_mb * policy["ram_fraction"]), 1)
    resident_weights = (arch.total - routed * (1 - policy["resident_fraction"])) * 2
    model = dict(
        architecture_version=3, recurrent_attention_mode="shared_first", rope_layout="per_head",
        hidden_size=h, num_hidden_layers=L, num_attention_heads=arch.num_attention_heads,
        num_key_value_heads=arch.num_key_value_heads, kv_latent_dim=arch.kv_latent_dim, rope_dim=ROPE_DIM,
        sliding_window_size=int(window), use_hybrid_attention=True, vocab_size=vocab, tie_word_embeddings=True,
        use_moe=bool(req.use_moe), num_experts=arch.num_experts if req.use_moe else 1,
        num_experts_per_tok=arch.num_experts_per_tok, num_shared_experts=shared,
        moe_intermediate_size=arch.ffn_width if req.use_moe else 0,
        intermediate_size=0 if req.use_moe else arch.ffn_width,
        lookahead_steps=LOOKAHEAD_STEPS, layer_aware_lookahead=True, layer_aware_cache=True,
        recurrent_enabled=depth > 1, recurrent_groups=groups, recurrent_depth=depth, prelude_layers=1,
        coda_layers=1, min_recurrent_depth=max(1, depth // 2), max_inference_depth=depth,
        adaptive_halting=depth > 1, halting_threshold=0.95, depth_schedule="fixed_then_adaptive",
        depth_schedule_warmup_fraction=0.3, recurrent_strategy="scheduled", recurrent_latent_refinement=False,
        ponder_loss_coef=3e-3, recurrent_distill_coef=0.05, recurrent_stability_coef=1e-3,
        lambda_balance=5e-3 if req.use_moe else 0.0, lambda_temporal=1e-3, lambda_lookahead=0.1,
        lambda_lookahead_topk=0.05, fallback_mask_prob=0.08 if req.use_moe else 0.0, dropout=0.0,
        kv_cache_dtype=policy["kv_cache_dtype"], mla_cache_rope=policy["mla_cache_rope"],
        router_scope=policy["router_scope"], ram_budget_mb=ram_budget,
        activation_checkpointing=policy["activation_checkpointing"], optimizer_paging=policy["optimizer_paging"],
        vram_budget_gb=round(max(0.25, resident_weights / 2**30 * 1.25), 2), dtype=req.dtype,
    )
    config = {**model, **overrides["pretrain"], "stage_overrides": overrides}
    if prof.tokenizer_dir and os.path.isfile(os.path.join(prof.tokenizer_dir, "tokenizer.json")):
        # The tokenizer the dataset was profiled with is the one the model needs.
        config["tokenizer_path"] = prof.tokenizer_dir
        sources["tokenizer_path"] = "dataset tokenizer report"

    longrun = [
        dict(name="pretrain", budget_fraction=0.5, peak_lr=lr, tokens_per_step=int(batch_tokens),
             micro_tokens=int(micro), seq_len=seq_len, warmup_fraction=0.02, min_lr_ratio=0.1,
             weight_decay=0.1, grad_clip=1.0),
        dict(name="sft", budget_fraction=0.5, peak_lr=_round_lr(lr / 3), tokens_per_step=int(max(micro, batch_tokens // 2)),
             micro_tokens=int(micro), seq_len=seq_len, warmup_fraction=0.01, min_lr_ratio=0.05,
             weight_decay=0.1, grad_clip=1.0),
    ]
    counted = None
    if verify:
        from chronos.config_schema import model_config_kwargs
        from chronos.model.factory import count_parameters
        counted = count_parameters(model_config_kwargs(config))
        drift = abs(counted["total"] / arch.total - 1)
        if drift > 0.01:
            notes.append(f"closed-form count differs from the meta-device count by {drift:.2%}")
    total = counted["total"] if counted else arch.total
    active = counted["active"] if counted else arch.active
    union = counted["active_union_upper"] if counted else arch.union_upper
    report = {
        "request": asdict(req) | {"data_paths": list(req.data_paths)},
        "architecture": {
            "total_params": total, "active_params": active, "active_union_upper": union,
            "params_source": "measured (meta-device count)" if counted else "estimated (closed form)",
            "hidden_size": h, "physical_layers": L, "recurrent_groups": groups, "recurrent_depth": depth,
            "virtual_layers": executed, "experts": arch.num_experts, "top_k": arch.num_experts_per_tok,
            "expert_width": arch.ffn_width, "heads": f"{arch.num_attention_heads}/{arch.num_key_value_heads}",
        },
        "dataset": {k: v for k, v in asdict(prof).items() if k not in ("files", "notes")}
                   | {"files": len(prof.files)},
        "scaling": {
            "tokens": tokens, "tokens_source": sources["tokens"], "flops_per_token": m,
            "compute_flops": compute, "deepseek_lr": eta, "deepseek_batch_tokens": deepseek_hparams(compute)[1],
            "chosen_lr": lr, "chosen_batch_tokens": batch_tokens, "label": "estimated",
            "law": "η=0.3118·C^-0.125, B=0.2920·C^0.3271 (DeepSeek LLM, arXiv 2401.02954)",
            "tokens_per_param": round(tokens / max(1, total), 1),
        },
        "memory": {
            "level": level, "ram_gb": round(ram_gb, 1), "target_gb": round(budget / 2**30, 1),
            "micro_tokens": micro, "rows_per_micro": rows, "accumulation_steps": accum,
            "training_peak_gb": round(peak / 2**30, 2), "training_peak_label": "estimated",
            "routed_expert_mb_bf16": round(expert_mb, 1), "ram_budget_mb": ram_budget,
            "kv_cache_dtype": policy["kv_cache_dtype"], "router_scope": policy["router_scope"],
            "activation_checkpointing": policy["activation_checkpointing"],
            "optimizer_paging": policy["optimizer_paging"],
        },
        "sources": sources,
        "notes": notes,
    }
    return {"config": config, "stage_overrides": overrides, "longrun_phases": longrun, "report": report}


def stage_table(overrides: dict) -> tuple[list[str], list[list]]:
    """(headers, rows) for the UI editor: one row per stage, one column per key."""
    headers = ["stage"] + list(STAGE_KEYS)
    rows = [[name] + [overrides.get(name, {}).get(k, "") for k in STAGE_KEYS] for name in STAGES]
    return headers, rows


def overrides_from_table(rows) -> dict:
    """Inverse of :func:`stage_table`; blank cells are dropped, numbers coerced."""
    out = {}
    for row in rows or ():
        row = list(row)
        if not row or str(row[0]).strip() not in STAGES:
            continue
        entry = {}
        for key, value in zip(STAGE_KEYS, row[1:]):
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            number = float(value)
            entry[key] = int(number) if key in _INT_KEYS else number
        out[str(row[0]).strip()] = entry
    return out


_INT_KEYS = {"batch_size", "accumulation_steps", "epochs", "max_steps", "max_seq_len", "save_interval",
             "num_generations", "max_gen_len"}
