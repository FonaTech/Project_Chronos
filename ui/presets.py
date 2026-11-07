"""
ui/presets.py

Default + preset configurations matching minimind's MoE defaults.

Each preset is a complete dict that can be:
  1. Loaded into the Config tab sliders via the "Load Preset" button.
  2. Used as the initial value when the user clicks "Reset to MiniMind".
  3. Persisted to / loaded from JSON via the Save/Load Config buttons.

The single source of truth for the *minimind* preset is what
``MiniMindConfig(use_moe=True)`` actually produces — we mirror those values
here so the UI shows the same model the trainer would build.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List


# ── MiniMind MoE — exact defaults from model/model_minimind.py ─────
# These values match MiniMindConfig(hidden_size=768, num_hidden_layers=8,
# use_moe=True) field-for-field, with Chronos-specific extensions left at
# their conservative defaults (no router-anchor by default).
MINIMIND_MOE_DEFAULTS: Dict = {'architecture_version': 2,
 'recurrent_enabled': True,
 'recurrent_groups': 2,
 'recurrent_depth': 4,
 'min_recurrent_depth': 2,
 'prelude_layers': 1,
 'coda_layers': 1,
 'adaptive_halting': True,
 'halting_threshold': 0.99,
 'halting_epsilon': 0.01,
 'max_inference_depth': 4,
 'ponder_loss_coef': 0.01,
 'recurrent_distill_coef': 0.05,
 'recurrent_stability_coef': 0.001,
 'recurrent_stability_threshold': 0.1,
 'recurrent_dt_init': 0.25,
 'recurrent_dt_min': 0.05,
 'recurrent_dt_max': 1.0,
 'recurrent_latent_refinement': True,
 'recurrent_strategy': 'scheduled',
 'recurrent_depth_jitter': 0.0,
 'recurrent_parallel_refine': False,
 'depth_schedule': 'fixed_then_adaptive',
 'depth_schedule_warmup_fraction': 0.25,
 'layer_aware_lookahead': True,
 'layer_aware_cache': True,
 'training_paging': False,
 'optimizer_paging': False,
 'optimizer_page_dir': './optimizer_pages',
 'page_size_mb': 64,
 'prefetch_depth': 2,
 'activation_checkpointing': False,
 'modalities': [],
 'auto_detect_modalities': True,
 'image_size': 224,
 'image_patch_size': 16,
 'max_image_tokens': 196,
 'audio_sample_rate': 16000,
 'max_audio_seconds': 30.0,
 'max_audio_tokens': 128,
 'modality_dropout': 0.05,
 'freeze_modality_encoders': False,
 'image_encoder_type': 'patch',
 'audio_encoder_type': 'conv',
 'modality_fusion_heads': 4,
 'media_error_policy': 'warn',
 'modality_adapters': {},
 'modality_alignment_coef': 0.0,
 'hidden_size': 768,
 'num_hidden_layers': 8,
 'num_attention_heads': 8,
 'num_key_value_heads': 4,
 'vocab_size': 6400,
 'intermediate_size': 0,
 'moe_intermediate_size': 0,
 'max_seq_len': 512,
 'tie_word_embeddings': True,
 'num_experts': 4,
 'num_experts_per_tok': 2,
 'num_shared_experts': 1,
 'kv_latent_dim': 64,
 'rope_dim': 32,
 'sliding_window_size': 2048,
 'use_hybrid_attention': True,
 'lookahead_steps': 2,
 'lambda_balance': 0.0005,
 'lambda_temporal': 0.001,
 'lambda_lookahead': 0.1,
 'lambda_lookahead_topk': 0.05,
 'lambda_router_anchor': 0.0,
 'lambda_cluster_locality': 0.3,
 'cluster_locality_clusters': 0,
 'cluster_locality_window': 32,
 'fallback_mask_prob': 0.0,
 'vram_budget_gb': 4.0,
 'pinned_memory_max_fraction': 0.25,
 'storage_format': 'safetensors',
 'learning_rate': 0.0005,
 'batch_size': 16,
 'accumulation_steps': 8,
 'epochs': 2,
 'max_steps': 0,
 'save_interval': 1000,
 'weight_decay': 0.01,
 'grad_clip': 1.0,
 'log_interval': 10,
 'reward_spec': 'toy',
 'max_gen_len': 24,
 'num_generations': 4,
 'temperature': 1.0,
 'beta': 0.1,
 'alpha': 0.7,
 'lambda_or': 0.1,
 'save_dir': './out',
 'dtype': 'auto',
 'cpu_threads': 'auto',
 'cpu_budget_percent': 100,
 'num_workers': 'auto'}


CONFIG_INPUT_ORDER = ['hidden_size',
 'num_hidden_layers',
 'num_experts',
 'num_experts_per_tok',
 'num_shared_experts',
 'lookahead_steps',
 'kv_latent_dim',
 'sliding_window_size',
 'num_attention_heads',
 'num_key_value_heads',
 'rope_dim',
 'moe_intermediate_size',
 'vocab_size',
 'dtype',
 'tie_word_embeddings',
 'lambda_balance',
 'lambda_temporal',
 'lambda_lookahead',
 'lambda_lookahead_topk',
 'lambda_router_anchor',
 'lambda_cluster_locality',
 'cluster_locality_clusters',
 'cluster_locality_window',
 'fallback_mask_prob',
 'vram_budget_gb',
 'pinned_memory_max_fraction',
 'use_hybrid_attention',
 'storage_format',
 'learning_rate',
 'batch_size',
 'accumulation_steps',
 'max_seq_len',
 'epochs',
 'max_steps',
 'save_interval',
 'log_interval',
 'save_dir',
 'weight_decay',
 'grad_clip',
 'cpu_threads',
 'cpu_budget_percent',
 'num_workers',
 'reward_spec',
 'max_gen_len',
 'num_generations',
 'temperature',
 'beta',
 'alpha',
 'lambda_or',
 'recurrent_enabled',
 'recurrent_groups',
 'recurrent_depth',
 'min_recurrent_depth',
 'adaptive_halting',
 'halting_threshold',
 'max_inference_depth',
 'ponder_loss_coef',
 'recurrent_distill_coef',
 'recurrent_stability_coef',
 'recurrent_latent_refinement',
 'depth_schedule',
 'recurrent_strategy',
 'recurrent_depth_jitter',
 'depth_schedule_warmup_fraction',
 'layer_aware_cache',
 'prefetch_depth',
 'training_paging',
 'optimizer_paging',
 'page_size_mb',
 'optimizer_page_dir',
 'modalities']


# ── Chronos v3 tiers ───────────────────────────────────────────────
# Generated by the adaptive planner (chronos.model.planner) and stored as JSON
# so importing the UI never runs the search.  ``python -m ui.presets``
# regenerates the file from V3_TIER_REQUESTS.
V3_PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets_v3.json")

V3_TIER_REQUESTS = [
    ("Chronos-v3 Tiny (smoke / CI)", "1.5M smoke test; trains in minutes on CPU.",
     dict(target_total=1.5e6, num_experts=4, active_experts=2, vocab_size=6400, seq_len=128, recurrent_depth=2,
          memory_saving="none", token_budget=2e6, ram_gb=16)),
    ("Chronos-v3 50M (16k vocab)", "50M MoE (36 experts, top-2) for a 16k Chronos tokenizer; no memory saving.",
     dict(target_total=50e6, num_experts=36, active_experts=2, vocab_size=16384, seq_len=512,
          memory_saving="none", ram_gb=32)),
    ("Chronos-v3 200M (32 GB)", "General 200M MoE, balanced memory saving, minimind 6400 vocab.",
     dict(target_total=200e6, num_experts=32, active_experts=2, vocab_size=6400, seq_len=1024,
          memory_saving="balanced", ram_gb=32)),
    ("Chronos-v3 500M (32 GB)", "500M MoE, aggressive memory saving (int8 KV, shared routing, checkpointing).",
     dict(target_total=500e6, num_experts=48, active_experts=3, vocab_size=6400, seq_len=1024,
          memory_saving="aggressive", ram_gb=32)),
    ("Chronos-v3 1B (64 GB)", "1B MoE, 64 fine-grained experts, aggressive memory saving.",
     dict(target_total=1e9, num_experts=64, active_experts=4, vocab_size=6400, seq_len=2048,
          memory_saving="aggressive", ram_gb=64)),
]


def build_v3_presets() -> dict:
    """Run the planner for every tier; returns the presets_v3.json document."""
    from chronos.model.planner import PlanRequest, plan_training
    presets, meta = {}, {}
    for name, blurb, request in V3_TIER_REQUESTS:
        plan = plan_training(PlanRequest(**request))
        cfg = dict(plan["config"])
        presets[name] = {k: v for k, v in cfg.items() if MINIMIND_MOE_DEFAULTS.get(k, object()) != v}
        arch, mem = plan["report"]["architecture"], plan["report"]["memory"]
        meta[name] = dict(
            summary=blurb, request={k: (list(v) if isinstance(v, tuple) else v) for k, v in request.items()},
            total_params=arch["total_params"], active_params=arch["active_params"],
            active_union_upper=arch["active_union_upper"], params_label=arch["params_source"],
            training_peak_gb=mem["training_peak_gb"], training_peak_label="estimated",
            notes=plan["report"]["notes"])
    return {"_about": "Generated by chronos.model.planner.plan_training (inputs in _meta.request). Parameter "
                      "counts are measured on the meta device; memory figures are estimates. Regenerate with "
                      "`python -m ui.presets` instead of hand-editing.",
            "_meta": meta, "presets": presets}


def _load_v3_presets() -> tuple[Dict[str, Dict], Dict[str, Dict]]:
    try:
        with open(V3_PRESETS_PATH, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}, {}
    return ({name: {**MINIMIND_MOE_DEFAULTS, **cfg} for name, cfg in doc.get("presets", {}).items()},
            dict(doc.get("_meta", {})))


V3_PRESETS, V3_PRESET_META = _load_v3_presets()
DEFAULT_PRESET = "Chronos-v3 200M (32 GB)" if "Chronos-v3 200M (32 GB)" in V3_PRESETS else "MiniMind-MoE (default)"

PRESETS: Dict[str, Dict] = {
    **V3_PRESETS,
    # Legacy v2 shape kept for minimind compatibility (first_only recurrence).
    "MiniMind-MoE (default)": dict(MINIMIND_MOE_DEFAULTS),
}

# Pre-v3 preset names (saved sessions, docs) resolve to the closest v3 tier.
_PRESET_ALIASES = {
    "Recommended-CN (≈120M)": "Chronos-v3 200M (32 GB)",
    "Recommended-CN (≈192M)": "Chronos-v3 200M (32 GB)",
    "Tiny (smoke / CI)": "Chronos-v3 Tiny (smoke / CI)",
    "Small (256H × 4L × 4E)": "Chronos-v3 50M (16k vocab)",
    "Chronos-v3 50M (en\u2192zh recipe)": "Chronos-v3 50M (16k vocab)",   # 0.3.1a1 tier name
    "Medium (512H × 8L × 8E)": "Chronos-v3 200M (32 GB)",
    "Large (1024H × 16L × 16E)": "Chronos-v3 1B (64 GB)",
}


def preset_names() -> List[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> Dict:
    """Return a copy of the named preset (avoids accidental mutation)."""
    name = _PRESET_ALIASES.get(name, name)
    cfg = dict(MINIMIND_MOE_DEFAULTS)
    cfg.update(json.loads(json.dumps(PRESETS.get(name, {}))))     # nested stage_overrides copied too
    return cfg


def values_in_input_order(cfg: Dict) -> List:
    """Return preset values in the same order as Config tab's all_inputs.

    For keys missing from `cfg`, falls back to MINIMIND_MOE_DEFAULTS. If a
    value is ``None`` (missing in both), it is replaced by a safe-default
    per field type so Gradio's slider/number widgets don't reject it.
    """
    # Hardcoded fallbacks for fields whose MiniMind default is None or absent.
    # Values chosen to match the slider minima where applicable.
    SAFE = {
        "hidden_size": 768, "num_hidden_layers": 8,
        "num_experts": 4, "num_experts_per_tok": 2, "num_shared_experts": 1,
        "lookahead_steps": 2, "kv_latent_dim": 64, "sliding_window_size": 2048,
        "num_attention_heads": 8, "num_key_value_heads": 4, "rope_dim": 32,
        "moe_intermediate_size": 0, "vocab_size": 6400,
        "dtype": "auto", "tie_word_embeddings": True,
        "lambda_balance": 5e-4, "lambda_temporal": 1e-3,
        "lambda_lookahead": 0.1, "lambda_lookahead_topk": 0.05, "lambda_router_anchor": 0.0,
        "lambda_cluster_locality": 0.3, "cluster_locality_clusters": 0, "cluster_locality_window": 32,
        "fallback_mask_prob": 0.0,
        "vram_budget_gb": 4.0, "pinned_memory_max_fraction": 0.25,
        "use_hybrid_attention": True, "storage_format": "safetensors",
        "learning_rate": 5e-4, "batch_size": 16, "accumulation_steps": 8,
        "max_seq_len": 512, "epochs": 2, "max_steps": 0, "save_interval": 1000,
        "save_dir": "./out",
        "cpu_threads": "auto", "cpu_budget_percent": 100,
        "num_workers": "auto",
        "weight_decay": 0.01, "grad_clip": 1.0, "log_interval": 10,
        "reward_spec": "toy",
        "max_gen_len": 24, "num_generations": 4, "temperature": 1.0,
        "beta": 0.1, "alpha": 0.7, "lambda_or": 0.1,
        "recurrent_enabled": True, "recurrent_groups": 2,
        "recurrent_depth": 4, "min_recurrent_depth": 2,
        "adaptive_halting": True, "halting_threshold": 0.99,
        "max_inference_depth": 4, "ponder_loss_coef": 1e-2,
        "recurrent_distill_coef": 0.05, "recurrent_stability_coef": 1e-3,
        "recurrent_latent_refinement": True,
        "depth_schedule": "fixed_then_adaptive", "recurrent_strategy": "scheduled",
        "recurrent_depth_jitter": 0.0, "depth_schedule_warmup_fraction": 0.25,
        "layer_aware_cache": True, "prefetch_depth": 2,
        "training_paging": False, "optimizer_paging": False, "page_size_mb": 64,
        "optimizer_page_dir": "./optimizer_pages", "modalities": "",
    }
    out = []
    for k in CONFIG_INPUT_ORDER:
        v = cfg.get(k) if k in cfg else MINIMIND_MOE_DEFAULTS.get(k)
        if v is None:
            v = SAFE.get(k, 0)
        if k == "dtype":
            normalized = str(v).strip().lower()
            v = {
                "fp32": "float32",
                "bf16": "bfloat16",
                "fp16": "float16",
                "half": "float16",
                "full": "float32",
                "": "auto",
            }.get(normalized, normalized)
        if k == "modalities" and isinstance(v, (list, tuple)):
            v = ",".join(v)
        out.append(v)
    return out


def save_config(cfg: Dict, path: str) -> str:
    """Write cfg as pretty JSON. Returns the absolute path written."""
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return abs_path


def load_config(path: str) -> Dict:
    """Read JSON and return the dict. Raises FileNotFoundError if missing."""
    with open(os.path.abspath(path)) as f:
        loaded = json.load(f)
    cfg = dict(MINIMIND_MOE_DEFAULTS)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration JSON must be an object")
    if "max_position_embeddings" in loaded and "max_seq_len" not in loaded:
        loaded["max_seq_len"] = loaded["max_position_embeddings"]
    cfg.update(loaded)
    return cfg


if __name__ == "__main__":                      # python -m ui.presets → regenerate presets_v3.json
    with open(V3_PRESETS_PATH, "w", encoding="utf-8") as fh:
        json.dump(build_v3_presets(), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {V3_PRESETS_PATH}")
