"""Canonical Chronos v2 configuration mapping.

Every boundary (UI, CLI, checkpoint, HF config, GGUF metadata) must use this
module rather than maintaining another positional list of model fields.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
from typing import Any, Mapping


SCHEMA_VERSION = 3
ARCHITECTURE_VERSION = 3
LEGACY_ARCHITECTURE_VERSION = 2

MODEL_FIELDS = (
    "architecture_version", "hidden_size", "num_hidden_layers", "vocab_size",
    "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
    "max_position_embeddings", "tie_word_embeddings", "dropout", "rms_norm_eps",
    "rope_theta", "flash_attn", "bos_token_id", "eos_token_id",
    "recurrent_enabled", "recurrent_groups", "recurrent_depth", "min_recurrent_depth",
    "prelude_layers", "coda_layers", "adaptive_halting", "halting_threshold",
    "halting_epsilon", "max_inference_depth", "ponder_loss_coef",
    "recurrent_distill_coef", "recurrent_stability_coef",
    "recurrent_contraction_coef", "recurrent_contraction_budget",
    "recurrent_stability_threshold", "recurrent_dt_init", "recurrent_dt_min",
    "recurrent_dt_max", "recurrent_latent_refinement", "recurrent_attention_mode",
    "recurrent_strategy",
    "recurrent_depth_jitter", "recurrent_parallel_refine",
    "recurrent_gradient_checkpoint_groups", "depth_schedule",
    "depth_schedule_warmup_fraction",
    "use_moe", "num_experts", "num_experts_per_tok", "num_shared_experts",
    "moe_intermediate_size", "norm_topk_prob", "router_aux_loss_coef",
    "lookahead_steps", "layer_aware_lookahead", "layer_aware_cache",
    "lambda_balance", "lambda_temporal", "lambda_lookahead",
    "lambda_lookahead_topk", "lambda_lookahead_union", "lambda_router_locality",
    "lambda_cluster_locality", "cluster_locality_clusters", "cluster_locality_window", "cluster_locality_mode",
    "lambda_router_anchor", "fallback_mask_prob", "offload_miss_policy_default",
    "use_hybrid_attention", "kv_latent_dim", "rope_dim", "rope_layout", "sliding_window_size",
    "kv_cache_dtype", "kv_quant_group_size", "mla_decode", "mla_cache_rope", "router_scope",
    "draft_exit_kl_coef", "draft_exit_depth", "max_resident_experts", "ram_budget_mb",
    "vram_budget_gb", "pinned_memory_max_fraction", "prefetch_depth",
    "recommended_resident_experts", "offload_working_set_experts", "storage_format",
    "cluster_manifest_path", "training_paging", "optimizer_paging",
    "optimizer_page_dir", "page_size_mb", "activation_checkpointing", "ce_chunk_size", "attention_block_size",
    "reset_router_at_documents", "document_masking", "recompute_policy", "recompute_scope", "lookahead_input_detach", "temporal_grad_span", "moe_chunk_tokens", "mla_upproj_recompute",
    "reset_lookahead_at_documents", "rope_train_scaling",
    "modalities", "auto_detect_modalities", "image_size", "image_patch_size",
    "max_image_tokens", "audio_sample_rate", "max_audio_seconds", "max_audio_tokens",
    "modality_dropout", "freeze_modality_encoders", "image_encoder_type",
    "audio_encoder_type", "modality_fusion_heads", "media_error_policy",
    "modality_adapters", "modality_alignment_coef",
)

RUNTIME_FIELDS = (
    "dtype", "learning_rate", "batch_size", "accumulation_steps", "epochs",
    "max_steps", "save_interval", "log_interval", "save_dir", "weight_decay",
    "grad_clip", "cpu_threads", "cpu_budget_percent", "num_workers", "reward_spec",
    "max_gen_len", "num_generations", "temperature", "beta", "alpha", "lambda_or",
)

CANONICAL_FIELDS = frozenset(MODEL_FIELDS + RUNTIME_FIELDS + ("max_seq_len", "extra_metadata"))


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def canonical_config_dict(source: Any, *, preserve_unknown: bool = True) -> dict:
    """Return a versioned, JSON-safe configuration with no silent drops."""
    if isinstance(source, Mapping):
        raw = dict(source)
    elif hasattr(source, "to_dict"):
        raw = dict(source.to_dict())
    else:
        raw = dict(vars(source))
    if "max_seq_len" in raw and "max_position_embeddings" not in raw:
        raw["max_position_embeddings"] = raw["max_seq_len"]
    known = {key: _plain(raw[key]) for key in CANONICAL_FIELDS if key in raw}
    extra = dict(known.pop("extra_metadata", {}) or {})
    if preserve_unknown:
        for key, value in raw.items():
            if key not in CANONICAL_FIELDS and not key.startswith("_"):
                extra[key] = _plain(value)
    # A config without an explicit version predates v3 metadata; treat it
    # as legacy so its recurrent numerics never change silently.
    known["architecture_version"] = int(known.get("architecture_version", LEGACY_ARCHITECTURE_VERSION))
    known["schema_version"] = SCHEMA_VERSION
    known["extra_metadata"] = extra
    return known


def model_config_kwargs(source: Any) -> dict:
    canonical = canonical_config_dict(source)
    out = {key: canonical[key] for key in MODEL_FIELDS if key in canonical}
    out["extra_metadata"] = canonical.get("extra_metadata", {})
    for key in ("intermediate_size", "moe_intermediate_size"):
        if out.get(key) in (None, 0, ""):
            out.pop(key, None)
    return out


def normalize_model_dimensions(source: Mapping[str, Any]) -> dict:
    """Resolve auto widths and the physical recurrent partition without Torch."""
    out = dict(source)
    h, layers = int(out["hidden_size"]), int(out["num_hidden_layers"])
    if h < 1 or layers < 1:
        raise ValueError("hidden_size and num_hidden_layers must be positive")
    heads = int(out["num_attention_heads"])
    if heads < 1:
        raise ValueError("num_attention_heads must be positive")
    out["head_dim"] = int(out.get("head_dim") or h // heads)
    out["num_key_value_heads"] = int(out.get("num_key_value_heads") or heads)
    if out["head_dim"] < 1 or heads % out["num_key_value_heads"]:
        raise ValueError("attention heads must divide evenly by KV heads, with head_dim >= 1")
    out["intermediate_size"] = int(out.get("intermediate_size") or math.ceil(h*math.pi/64)*64)
    out["moe_intermediate_size"] = int(out.get("moe_intermediate_size") or out["intermediate_size"])
    if out.get("use_moe", True) and not 1 <= int(out["num_experts_per_tok"]) <= int(out["num_experts"]):
        raise ValueError("num_experts_per_tok must be between 1 and num_experts")
    p = max(0, min(int(out.get("prelude_layers", 0)), layers-1))
    c = max(0, min(int(out.get("coda_layers", 0)), layers-p-1))
    g = max(1, min(int(out.get("recurrent_groups") or layers-p-c), layers-p-c))
    depth = max(1, int(out.get("recurrent_depth", 1)))
    minimum = max(1, min(int(out.get("min_recurrent_depth", 1)), depth))
    out.update(prelude_layers=p, recurrent_groups=g, coda_layers=layers-p-g,
               recurrent_depth=depth, min_recurrent_depth=minimum,
               max_inference_depth=max(minimum, int(out.get("max_inference_depth") or depth)))
    return out


def config_hash(source: Any) -> str:
    payload = json.dumps(canonical_config_dict(source), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ui_to_canonical(ui_state: Mapping[str, Any]) -> dict:
    return canonical_config_dict(ui_state)


def canonical_to_ui(config: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict:
    out = dict(defaults)
    canonical = canonical_config_dict(config)
    out.update({k: v for k, v in canonical.items() if k in out or k in CANONICAL_FIELDS})
    out.pop("schema_version", None)
    return out


RECURRENT_ATTENTION_MODES = ("shared_first", "per_iteration", "first_only")


def resolve_recurrent_attention_mode(source: Any) -> str:
    """Return the recurrent attention contract for a config or mapping.

    v3 configs carry ``recurrent_attention_mode`` explicitly.  v2 configs
    predate it and are resolved from ``recurrent_latent_refinement`` so their
    numerics never change: True → ``first_only`` (latent-only middle steps),
    False → ``per_iteration``.
    """
    get = source.get if isinstance(source, Mapping) else (lambda k, d=None: getattr(source, k, d))
    mode = get("recurrent_attention_mode", None)
    if mode:
        mode = str(mode).lower()
        if mode not in RECURRENT_ATTENTION_MODES:
            raise ValueError(f"recurrent_attention_mode must be one of {RECURRENT_ATTENTION_MODES}, got {mode!r}")
        return mode
    if int(get("architecture_version", LEGACY_ARCHITECTURE_VERSION) or LEGACY_ARCHITECTURE_VERSION) >= 3:
        return "shared_first"
    return "first_only" if bool(get("recurrent_latent_refinement", True)) else "per_iteration"


class RecurrentCacheLayout:
    """Map (physical layer, loop iteration) to a KV-cache slot.

    ``shared_first``/``first_only`` keep one slot per physical layer.
    ``per_iteration`` gives each recurrent iteration its own slot, ordered
    prelude, group0·step0..stepD-1, group1·…, coda.
    """

    def __init__(self, prelude: int, groups: int, coda: int, depth: int, mode: str):
        self.prelude, self.groups, self.coda = int(prelude), int(groups), int(coda)
        self.depth, self.mode = max(1, int(depth)), mode
        self.per_iteration = mode == "per_iteration"
        physical = self.prelude + self.groups + self.coda
        self.num_slots = (self.prelude + self.groups * self.depth + self.coda) if self.per_iteration else physical

    def slot(self, layer: int, step: int = 0) -> int:
        if not self.per_iteration or layer < self.prelude:
            return int(layer)
        if layer < self.prelude + self.groups:
            return self.prelude + (layer - self.prelude) * self.depth + int(step)
        return self.prelude + self.groups * self.depth + (layer - self.prelude - self.groups)

    def physical_layer(self, slot: int) -> int:
        if not self.per_iteration or slot < self.prelude:
            return int(slot)
        recurrent_end = self.prelude + self.groups * self.depth
        if slot < recurrent_end:
            return self.prelude + (slot - self.prelude) // self.depth
        return self.prelude + self.groups + (slot - recurrent_end)


def recurrent_cache_layout(config: Any, depth: int | None = None) -> RecurrentCacheLayout:
    get = config.get if isinstance(config, Mapping) else (lambda k, d=None: getattr(config, k, d))
    return RecurrentCacheLayout(
        get("prelude_layers", 0), get("recurrent_groups", 1), get("coda_layers", 0),
        depth if depth is not None else get("max_inference_depth", None) or get("recurrent_depth", 1),
        resolve_recurrent_attention_mode(config),
    )
