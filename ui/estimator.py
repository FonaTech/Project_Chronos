"""Static parameter and memory budgets matching the Chronos physical graph."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Optional


DTYPE_BYTES = {
    "fp32": 4, "float32": 4,
    "bf16": 2, "bfloat16": 2,
    "fp16": 2, "float16": 2,
    "int8": 1,
    "nf4": 0.5,
}


def _base_intermediate(hidden_size: int) -> int:
    """ChronosBaseConfig auto-derivation: ceil(h·π/64)·64."""
    return int(math.ceil(hidden_size * math.pi / 64) * 64)


@dataclass
class ArchConfig:
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    vocab_size: int = 6400
    # intermediate_size / moe_intermediate_size: when None, auto-derive from
    # hidden_size via the ChronosBaseConfig rule so the estimate matches the trainer.
    intermediate_size: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    head_dim: Optional[int] = None
    use_moe: bool = True
    layer_aware_lookahead: bool = False
    recurrent_latent_refinement: bool = True
    max_inference_depth: Optional[int] = None
    image_patch_size: int = 16
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
    recurrent_enabled: bool = True
    # None → derived like the model (layers − prelude − coda); a fixed default
    # here silently moved layers from the recurrent groups into the coda.
    recurrent_groups: Optional[int] = None
    architecture_version: int = 2
    recurrent_attention_mode: Optional[str] = None
    recurrent_depth: int = 1
    prelude_layers: int = 0
    coda_layers: int = 0
    min_recurrent_depth: int = 1
    training_paging: bool = False
    optimizer_paging: bool = False
    resident_experts: Optional[int] = None
    modalities: tuple[str, ...] = ()
    image_size: int = 224
    max_image_tokens: int = 196
    max_audio_tokens: int = 128

    def __post_init__(self):
        from chronos.config_schema import normalize_model_dimensions
        resolved = normalize_model_dimensions(vars(self))
        for key in vars(self).keys() & resolved.keys():
            setattr(self, key, resolved[key])

    @classmethod
    def from_config(cls, source):
        raw = dict(source) if isinstance(source, dict) else vars(source)
        values = {f.name: raw[f.name] for f in fields(cls) if f.name in raw}
        values["resident_experts"] = raw.get("offload_working_set_experts", raw.get("resident_experts"))
        return cls(**values)

    @property
    def attention_mode(self) -> str:
        from chronos.config_schema import resolve_recurrent_attention_mode
        return resolve_recurrent_attention_mode(vars(self))

    @property
    def executed_depth(self):
        return self.max_inference_depth if self.recurrent_enabled and self.recurrent_depth > 1 else 1


def _attn_params(cfg: ArchConfig, layer: int = 0) -> int:
    h, nh, nk, d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    if cfg.use_hybrid_attention and layer % 2 == 0:
        latent = cfg.kv_latent_dim
        return 2*h*nh*d + h*latent + 2*latent*nk*d + d + latent
    return 2*h*nh*d + 2*h*nk*d + 2*d


def _expert_params(cfg: ArchConfig) -> int:
    return 3 * cfg.hidden_size * cfg.moe_intermediate_size


def parameter_breakdown(cfg: ArchConfig) -> dict[str, int]:
    """Unique allocated tensors, including norms/biases; tied weights counted once."""
    h, layers = cfg.hidden_size, cfg.num_hidden_layers
    image = 3*h*cfg.image_patch_size**2 + 2*h if "image" in cfg.modalities else 0
    mid = max(32, h // 2)
    audio = 400*mid + mid + 5*mid*h + h + 2*h if "audio" in cfg.modalities else 0
    fusion = 4*h*h + 4*h + 2*h + 1 if cfg.modalities else 0
    predictor_hidden = max(1, h // 4)
    # The controller is instantiated even when recurrence is disabled.
    controller = (cfg.recurrent_depth + cfg.recurrent_groups)*h + 2*h + 1 + cfg.recurrent_groups
    return {
        "embedding_and_readout": h*cfg.vocab_size*(1 if cfg.tie_word_embeddings else 2),
        "attention": sum(_attn_params(cfg, i) for i in range(layers)),
        "block_and_final_norms": (2*layers + 1)*h,
        "routers": layers*h*cfg.num_experts if cfg.use_moe else 0,
        "routed_experts": layers*cfg.num_experts*_expert_params(cfg) if cfg.use_moe else 0,
        "shared_experts": layers*cfg.num_shared_experts*_expert_params(cfg) if cfg.use_moe else 0,
        "dense_ffn": layers*3*h*cfg.intermediate_size if not cfg.use_moe else 0,
        "lookahead": h*predictor_hidden + predictor_hidden*cfg.num_experts*(cfg.lookahead_steps+1)*(
            cfg.recurrent_groups if cfg.layer_aware_lookahead else 1),
        "recurrent_controller": controller,
        "image_encoder": image, "audio_encoder": audio, "modality_fusion": fusion,
    }


def total_params(cfg: ArchConfig) -> int:
    return sum(parameter_breakdown(cfg).values())


def active_param_bounds(cfg: ArchConfig) -> tuple[int, int]:
    """Route-dependent unique weight budget, assuming all configured media present.

    Embedding/readout tables use the usual full-table parameter convention.
    Repeated passes can select DIFFERENT experts: report the union bounds,
    rather than claiming top-k physical experts always cover a recurrent token.
    """
    parts = parameter_breakdown(cfg)
    base = sum(parts.values()) - parts["routed_experts"]
    if not (cfg.recurrent_enabled and cfg.recurrent_depth > 1):
        base -= parts["recurrent_controller"]
    if not cfg.use_moe:
        return base, base
    k, e = cfg.num_experts_per_tok, cfg.num_experts
    low = base + cfg.num_hidden_layers*k*_expert_params(cfg)
    extra = cfg.recurrent_groups*(min(e, k*cfg.executed_depth)-k)*_expert_params(cfg)
    return low, low + extra


def active_params(cfg: ArchConfig) -> int:
    """Lower bound when recurrent passes reuse the same top-k experts."""
    return active_param_bounds(cfg)[0]


def compute_params(cfg: ArchConfig) -> int:
    """Weight-use count, including repeats; NOT FLOPs or unique storage."""
    work = active_params(cfg)
    repeats = cfg.executed_depth - 1
    h = cfg.hidden_size
    ffn = ((cfg.num_experts_per_tok + cfg.num_shared_experts)*_expert_params(cfg) + h*cfg.num_experts
           if cfg.use_moe else 3*h*cfg.intermediate_size)
    mode = cfg.attention_mode
    for layer in range(cfg.prelude_layers, cfg.prelude_layers + cfg.recurrent_groups):
        if mode == "first_only":
            attn = 0                                   # latent-only middle iterations
        elif mode == "shared_first":                   # fresh Q and output projection, iteration-0 K/V
            attn = 2*h*cfg.num_attention_heads*cfg.head_dim + h
        else:                                          # per_iteration recomputes the whole block
            attn = _attn_params(cfg, layer) + h
        work += repeats*(ffn + h + attn)
    return work


def kv_cache_bytes(cfg: ArchConfig, seq_len: int = 1024, batch_size: int = 1) -> int:
    """Persistent cache only; MLA stores c_kv, sliding layers store K and V."""
    size = 0
    for layer in range(cfg.num_hidden_layers):
        if cfg.use_hybrid_attention and layer % 2 == 0:
            size += seq_len*cfg.kv_latent_dim
        else:
            length = min(seq_len, cfg.sliding_window_size) if cfg.use_hybrid_attention else seq_len
            size += length*2*cfg.num_key_value_heads*cfg.head_dim
    return int(size*batch_size*DTYPE_BYTES.get(cfg.dtype, 4))


def memory_footprint(cfg: ArchConfig, seq_len: int = 1024) -> Dict[str, float]:
    """Static budgets, excluding activations, allocator pools and IO staging peaks.

    Inference and Adam training are separate scenarios. On unified memory the
    same physical allocation must not be added again as both VRAM and RAM.
    """
    byte_width = DTYPE_BYTES.get(cfg.dtype, 4)  # auto: conservative FP32 scenario
    parts = parameter_breakdown(cfg)
    total = sum(parts.values())
    base = total - parts["routed_experts"]
    resident = cfg.resident_experts
    if resident is None:
        resident = max(cfg.num_experts_per_tok, cfg.num_experts // 2)
    resident = max(0, min(cfg.num_experts, int(resident)))
    resident_params = cfg.num_hidden_layers*resident*_expert_params(cfg) if cfg.use_moe else 0
    resident_weights = (base + resident_params)*byte_width
    cold = (parts["routed_experts"] - resident_params)*byte_width
    kv = kv_cache_bytes(cfg, seq_len)
    return {
        "total_bytes": total*byte_width, "active_bytes": active_params(cfg)*byte_width,
        "active_params": active_params(cfg), "compute_params": compute_params(cfg),
        "base_weight_bytes": base*byte_width, "resident_expert_bytes": resident_params*byte_width,
        "resident_weight_bytes": resident_weights, "resident_experts_per_layer": resident,
        "cold_bytes": cold, "kv_cache_bytes": kv,
        "vram_estimate_bytes": resident_weights + kv,
        "ram_estimate_bytes": resident_params*byte_width,  # example one warm working set
        "ssd_estimate_bytes": parts["routed_experts"]*byte_width,
        "optimizer_bytes": total*8, "gradient_bytes": total*byte_width,
        "training_state_bytes": total*(2*byte_width + 8),
        "mlx_master_weight_bytes": total*4 if byte_width < 4 else 0,
        "mlx_accumulated_gradient_bytes": total*4,
        "mlx_training_state_bytes": total*(byte_width + 4 + 8 + (4 if byte_width < 4 else 0)),
        "paged_optimizer_bytes": parts["routed_experts"]*8 if cfg.optimizer_paging else 0,
        "modality_encoder_bytes": (parts["image_encoder"] + parts["audio_encoder"] + parts["modality_fusion"])*byte_width,
    }


def estimated_decode_tps(cfg: ArchConfig, hit_rate: float = 0.85,
                         ssd_gbps: float = 3.0, vram_gbps: float = 600.0) -> float:
    """Illustrative bandwidth-only model, not a hardware benchmark."""
    byte_width = DTYPE_BYTES.get(cfg.dtype, 4)
    compute = compute_params(cfg)*byte_width / max(vram_gbps*1e9, 1)
    misses = ((1-hit_rate)*_expert_params(cfg)*byte_width*cfg.num_hidden_layers*cfg.num_experts_per_tok
              / max(ssd_gbps*1e9, 1)) if cfg.use_moe else 0
    return 1 / max(compute + misses, 1e-6)


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
