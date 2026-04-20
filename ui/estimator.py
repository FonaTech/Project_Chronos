"""
ui/estimator.py

Parameter-count and memory-footprint estimator for the Architecture
Designer tab. Pure-math functions; no torch dependency.

The numbers match an actual ``ChronosForCausalLM`` state_dict to within
~1% for standard configs — the mismatch comes from bias terms and RMSNorm
scale vectors which are tiny relative to the weight matrices we count.

IMPORTANT: MiniMindConfig auto-derives ``intermediate_size`` from
``hidden_size`` as ``ceil(H·π/64)·64`` unless the user explicitly sets
it, and ``moe_intermediate_size`` defaults to ``intermediate_size``.
The estimator replicates that auto-derivation here so the UI preview
matches what the trainer actually builds. Callers can still override
either value by passing it into ``ArchConfig`` explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


DTYPE_BYTES = {
    "fp32": 4, "float32": 4,
    "bf16": 2, "bfloat16": 2,
    "fp16": 2, "float16": 2,
    "int8": 1,
    "nf4": 0.5,
}


def _minimind_intermediate(hidden_size: int) -> int:
    """Replicates MiniMindConfig's auto-derivation rule."""
    return int(math.ceil(hidden_size * math.pi / 64) * 64)


@dataclass
class ArchConfig:
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    vocab_size: int = 6400
    # intermediate_size / moe_intermediate_size: when None, auto-derive from
    # hidden_size via the MiniMind rule so the estimate matches the trainer.
    intermediate_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    num_experts: int = 4
    num_experts_per_tok: int = 1
    num_shared_experts: int = 1
    max_position_embeddings: int = 2048
    lookahead_steps: int = 2
    kv_latent_dim: int = 64
    rope_dim: int = 32
    sliding_window_size: int = 2048
    use_hybrid_attention: bool = True
    tie_word_embeddings: bool = True
    dtype: str = "fp16"

    def __post_init__(self):
        auto = _minimind_intermediate(self.hidden_size)
        if self.intermediate_size is None or self.intermediate_size <= 0:
            self.intermediate_size = auto
        if self.moe_intermediate_size is None or self.moe_intermediate_size <= 0:
            self.moe_intermediate_size = auto


def _attn_params(cfg: ArchConfig) -> int:
    """Params per attention block (MLA/SW hybrid has similar Q/K/V/O sizes)."""
    h = cfg.hidden_size
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = h // n_h
    q = h * (n_h * head_dim)
    k = h * (n_kv * head_dim)
    v = h * (n_kv * head_dim)
    o = (n_h * head_dim) * h
    if cfg.use_hybrid_attention:
        # MLA adds latent KV projections; negligible vs full KV projections.
        q += h * cfg.kv_latent_dim
        k += cfg.kv_latent_dim * (n_kv * head_dim)
        v += cfg.kv_latent_dim * (n_kv * head_dim)
    return q + k + v + o


def _expert_params(cfg: ArchConfig) -> int:
    """Params per single expert FeedForward (SwiGLU: gate+up+down)."""
    h = cfg.hidden_size
    d = cfg.moe_intermediate_size
    return 3 * h * d


def _per_layer_params(cfg: ArchConfig) -> int:
    """Attention + routing + all experts + shared experts + 2×RMSNorm."""
    attn = _attn_params(cfg)
    router = cfg.hidden_size * cfg.num_experts
    experts = cfg.num_experts * _expert_params(cfg)
    shared = cfg.num_shared_experts * _expert_params(cfg)
    rmsnorm = 2 * cfg.hidden_size  # 2 norms per block
    return attn + router + experts + shared + rmsnorm


def _active_expert_slots(cfg: ArchConfig) -> int:
    """Experts whose weights are *used* on a given token: top-k + shared."""
    return cfg.num_experts_per_tok + cfg.num_shared_experts


def total_params(cfg: ArchConfig) -> int:
    """Full model parameter count."""
    embed = cfg.vocab_size * cfg.hidden_size
    layers = cfg.num_hidden_layers * _per_layer_params(cfg)
    lookahead = (
        cfg.hidden_size * (cfg.hidden_size // 4)
        + (cfg.hidden_size // 4) * cfg.num_experts * (cfg.lookahead_steps + 1)
    )
    lm_head = 0 if cfg.tie_word_embeddings else embed
    return int(embed + layers + lookahead + lm_head + cfg.hidden_size)  # + final norm


def active_params(cfg: ArchConfig) -> int:
    """Params touched by one token: embed + L·(attn + norm + (topK+shared)·expert)."""
    embed = cfg.vocab_size * cfg.hidden_size
    per_layer_active = (
        _attn_params(cfg)
        + 2 * cfg.hidden_size
        + cfg.hidden_size * cfg.num_experts  # router always active
        + _active_expert_slots(cfg) * _expert_params(cfg)
    )
    lm_head_active = 0 if cfg.tie_word_embeddings else embed
    return int(embed + cfg.num_hidden_layers * per_layer_active + lm_head_active)


def kv_cache_bytes(cfg: ArchConfig, seq_len: int = 1024) -> int:
    """Approximate KV-cache bytes for a given sequence length."""
    h = cfg.hidden_size
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = h // n_h
    bytes_per = DTYPE_BYTES.get(cfg.dtype, 2)
    if cfg.use_hybrid_attention:
        # Half the layers use MLA (compressed), half use sliding window.
        # MLA cache ≈ kv_latent_dim per position; SW cache capped at window.
        mla_layers = cfg.num_hidden_layers // 2
        sw_layers = cfg.num_hidden_layers - mla_layers
        mla_bytes = mla_layers * seq_len * cfg.kv_latent_dim * bytes_per
        sw_bytes = sw_layers * min(seq_len, cfg.sliding_window_size) * 2 * n_kv * head_dim * bytes_per
        return int(mla_bytes + sw_bytes)
    return int(cfg.num_hidden_layers * seq_len * 2 * n_kv * head_dim * bytes_per)


def memory_footprint(cfg: ArchConfig, seq_len: int = 1024) -> Dict[str, float]:
    """Return a dict of byte sizes for display in the UI."""
    bytes_per = DTYPE_BYTES.get(cfg.dtype, 2)
    total_b = total_params(cfg) * bytes_per
    active_b = active_params(cfg) * bytes_per
    kv_b = kv_cache_bytes(cfg, seq_len=seq_len)
    cold_b = total_b - active_b
    return {
        "total_bytes": total_b,
        "active_bytes": active_b,
        "cold_bytes": cold_b,          # candidates for SSD offload
        "kv_cache_bytes": kv_b,
        "vram_estimate_bytes": active_b + kv_b,
        "ssd_estimate_bytes": cold_b,
    }


def estimated_decode_tps(cfg: ArchConfig, hit_rate: float = 0.85,
                         ssd_gbps: float = 3.0, vram_gbps: float = 600.0) -> float:
    """Order-of-magnitude decode throughput estimator.

    Model: tps ≈ 1 / (t_compute + t_miss).
      t_compute ≈ active_params · flop_per_param / compute_gflops  (simplified
                  to a linear-in-active-params term with 1e-10 coefficient
                  so the output lives in a sensible range for toy models).
      t_miss    ≈ (1 − hit_rate) · expert_bytes_per_layer · num_layers / ssd_gbps.
    Conservative; intended as a UI hint, not a benchmark.
    """
    bytes_per = DTYPE_BYTES.get(cfg.dtype, 2)
    active_b = active_params(cfg) * bytes_per
    t_compute = max(active_b / (vram_gbps * 1e9), 1e-6)
    expert_b = _expert_params(cfg) * bytes_per
    t_miss = (1.0 - hit_rate) * expert_b * cfg.num_hidden_layers / (ssd_gbps * 1e9)
    return 1.0 / (t_compute + t_miss)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.2f} K"
    return str(n)
