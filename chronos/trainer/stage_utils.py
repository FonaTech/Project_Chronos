"""Shared helpers for standalone Chronos stage scripts."""
from __future__ import annotations

import argparse
import os

from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_controlled,
    resolve_checkpoint_path,
    resolve_checkpoint_tokenizer,
)
from chronos.model.config import ChronosConfig
from chronos.expert_clusters import DEFAULT_LAMBDA, DEFAULT_WINDOW
from chronos.config_schema import MODEL_FIELDS


TOPOLOGY_ARG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_experts",
    "num_experts_per_tok",
    "num_shared_experts",
    "lookahead_steps",
    "moe_intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "rope_dim",
    "kv_latent_dim",
    "vocab_size",
    "recurrent_groups",
    "recurrent_depth",
    "min_recurrent_depth",
    "prelude_layers",
    "coda_layers",
    "max_inference_depth",
    "halting_threshold",
    "adaptive_halting",
    "recurrent_latent_refinement",
    "training_paging",
    "optimizer_paging",
    "optimizer_page_dir",
    "page_size_mb",
    "prefetch_depth",
)


def add_topology_args(parser, *, defaults: bool = False):
    """Add topology CLI args.

    Alignment scripts use ``defaults=False`` so omitted args do not override
    the upstream checkpoint. Pretraining can pass ``defaults=True`` because it
    creates the first topology.
    """
    d = None if not defaults else 0
    parser.add_argument("--hidden_size", type=int, default=(512 if defaults else None))
    parser.add_argument("--num_hidden_layers", type=int, default=(8 if defaults else None))
    parser.add_argument("--num_experts", type=int, default=(4 if defaults else None))
    parser.add_argument("--num_experts_per_tok", type=int, default=(2 if defaults else None))
    parser.add_argument("--num_shared_experts", type=int, default=(1 if defaults else None))
    parser.add_argument("--lookahead_steps", type=int, default=(2 if defaults else None))
    parser.add_argument("--moe_intermediate_size", type=int, default=d)
    parser.add_argument("--num_attention_heads", type=int, default=(8 if defaults else None))
    parser.add_argument("--num_key_value_heads", type=int, default=(4 if defaults else None))
    parser.add_argument("--rope_dim", type=int, default=(32 if defaults else None))
    parser.add_argument("--kv_latent_dim", type=int, default=(64 if defaults else None))
    parser.add_argument("--vocab_size", type=int, default=(6400 if defaults else None))
    parser.add_argument("--tokenizer_path", default=None,
                        help="Tokenizer directory (tokenizer.json). Default: the checkpoint's own tokenizer "
                             "(<ckpt dir>/tokenizer or its config metadata), else minimind.")
    parser.add_argument("--recurrent_groups", type=int, default=(2 if defaults else None))
    parser.add_argument("--recurrent_depth", type=int, default=(4 if defaults else None))
    parser.add_argument("--min_recurrent_depth", type=int, default=(2 if defaults else None))
    parser.add_argument("--prelude_layers", type=int, default=(1 if defaults else None))
    parser.add_argument("--coda_layers", type=int, default=(1 if defaults else None))
    parser.add_argument("--max_inference_depth", type=int, default=(4 if defaults else None))
    parser.add_argument("--halting_threshold", type=float, default=(0.99 if defaults else None))
    parser.add_argument(
        "--adaptive_halting", action=argparse.BooleanOptionalAction,
        default=(True if defaults else None),
    )
    parser.add_argument(
        "--recurrent_latent_refinement", action=argparse.BooleanOptionalAction,
        default=(True if defaults else None),
    )
    parser.add_argument(
        "--training_paging", action=argparse.BooleanOptionalAction,
        default=(False if defaults else None),
    )
    parser.add_argument(
        "--optimizer_paging", action=argparse.BooleanOptionalAction,
        default=(False if defaults else None),
    )
    parser.add_argument("--optimizer_page_dir", default=("./optimizer_pages" if defaults else None))
    parser.add_argument("--page_size_mb", type=int, default=(64 if defaults else None))
    parser.add_argument("--prefetch_depth", type=int, default=(2 if defaults else None))
    # Additional kernel settings use one typed registry for CLI registration;
    # checkpoint-backed stages leave omitted values unset.
    fields = {
        "recurrent_strategy": (str, "scheduled"),
        "recurrent_depth_jitter": (float, 0.),
        "depth_schedule": (str, "fixed_then_adaptive"),
        "depth_schedule_warmup_fraction": (float, .25),
        "ponder_loss_coef": (float, .01),
        "recurrent_distill_coef": (float, .05),
        "recurrent_stability_coef": (float, .001),
        "recurrent_stability_threshold": (float, .1),
        "recurrent_dt_init": (float, .25),
        "recurrent_dt_min": (float, .05),
        "recurrent_dt_max": (float, 1.),
    }
    for name, (kind, value) in fields.items():
        parser.add_argument("--"+name, type=kind, default=value if defaults else None)
    for name, value in (("recurrent_enabled", True), ("layer_aware_lookahead", True),
                        ("layer_aware_cache", True), ("activation_checkpointing", False)):
        parser.add_argument("--"+name, action=argparse.BooleanOptionalAction, default=value if defaults else None)


def topology_overrides_from_args(args) -> dict:
    out = {}
    for key in MODEL_FIELDS:
        value = getattr(args, key, None)
        if value is None:
            continue
        if key == "moe_intermediate_size" and value == 0:
            continue
        out[key] = value
    return out


def build_pretrain_config(args) -> ChronosConfig:
    kwargs = topology_overrides_from_args(args)
    kwargs.update({
        "max_position_embeddings": getattr(args, "max_seq_len", 512),
        "lambda_balance": getattr(args, "lambda_balance", 5e-4),
        "lambda_temporal": getattr(args, "lambda_temporal", 3e-3),
        "lambda_lookahead": getattr(args, "lambda_lookahead", 0.1),
        "lambda_lookahead_topk": getattr(args, "lambda_lookahead_topk", 0.15),
        "lambda_lookahead_union": getattr(args, "lambda_lookahead_union", 0.05),
        "lambda_router_locality": getattr(args, "lambda_router_locality", 0.02),
        "lambda_cluster_locality": getattr(args, "lambda_cluster_locality", DEFAULT_LAMBDA),
        "cluster_locality_clusters": getattr(args, "cluster_locality_clusters", 0),
        "cluster_locality_window": getattr(args, "cluster_locality_window", DEFAULT_WINDOW),
        "vram_budget_gb": getattr(args, "vram_budget_gb", 4.0),
        "use_moe": True,
        "architecture_version": 2,
        "recurrent_enabled": getattr(args, "recurrent_enabled", True),
        "adaptive_halting": getattr(args, "adaptive_halting", True),
        "halting_threshold": getattr(args, "halting_threshold", 0.99),
        "max_inference_depth": getattr(args, "max_inference_depth", None),
        "recurrent_latent_refinement": getattr(args, "recurrent_latent_refinement", True),
        "layer_aware_lookahead": getattr(args, "layer_aware_lookahead", True),
        "layer_aware_cache": getattr(args, "layer_aware_cache", True),
        "training_paging": getattr(args, "training_paging", False),
        "optimizer_paging": getattr(args, "optimizer_paging", False),
        "optimizer_page_dir": getattr(args, "optimizer_page_dir", "./optimizer_pages"),
        "page_size_mb": getattr(args, "page_size_mb", 64),
        "prefetch_depth": getattr(args, "prefetch_depth", 2),
    })
    fallback_mask_prob = getattr(args, "fallback_mask_prob", None)
    if fallback_mask_prob is not None:
        kwargs["fallback_mask_prob"] = fallback_mask_prob
    return ChronosConfig(**kwargs)


def build_config_from_upstream(
    args,
    *,
    default_stem: str,
    max_positions: int,
    lambda_router_anchor: float = 0.0,
) -> tuple[ChronosConfig, str, list[str]]:
    hidden = getattr(args, "hidden_size", None)
    requested = getattr(args, "from_weight", default_stem) or default_stem
    explicit = None
    if requested.endswith(".pth") or os.path.sep in requested:
        explicit = requested
        requested = os.path.splitext(os.path.basename(requested))[0]
    ckpt = resolve_checkpoint_path(
        getattr(args, "save_dir", "out"),
        requested,
        hidden_size=hidden,
        explicit_path=explicit,
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"required upstream checkpoint not found: {ckpt}. "
            "Run the prior stage first or pass --from_weight/--save_dir."
        )
    overrides = topology_overrides_from_args(args)
    overrides["lambda_router_anchor"] = lambda_router_anchor
    for key in (
        "lambda_temporal",
        "lambda_lookahead",
        "lambda_lookahead_topk",
        "lambda_lookahead_union",
        "lambda_router_locality",
        "lambda_cluster_locality",
        "cluster_locality_window",
        "fallback_mask_prob",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    cfg, sources = chronos_config_from_checkpoint(
        ckpt,
        overrides=overrides,
        require_unsniffable=True,
    )
    cfg.max_position_embeddings = max(int(getattr(cfg, "max_position_embeddings", 0) or 0), int(max_positions))
    return cfg, ckpt, sources


def load_required_checkpoint(model, checkpoint_path: str, device: str) -> dict:
    # Always deserialize checkpoints on CPU first. Loading directly to MPS/CUDA
    # can create a large transient device-memory spike before the model copy
    # settles.
    state = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    load_state_dict_controlled(model, state)
    return state


def load_stage_tokenizer(args, config, checkpoint_path: str | None = None):
    """Load the tokenizer that matches ``checkpoint_path`` (or ``--tokenizer_path``).

    Raises an error when the tokenizer has more entries than ``config.vocab_size``.
    A larger tokenizer would emit ids the embedding cannot hold. The minimind
    tokenizer (6400 entries) used on a 16k-vocabulary checkpoint fails the other
    way: it trains silently on the wrong ids.
    """
    from transformers import AutoTokenizer
    path = resolve_checkpoint_tokenizer(checkpoint_path, getattr(args, "tokenizer_path", None))
    tokenizer = AutoTokenizer.from_pretrained(path)
    vocab = int(getattr(config, "vocab_size", 0) or 0)
    if vocab and len(tokenizer) > vocab:
        raise ValueError(f"tokenizer at {path} has {len(tokenizer)} entries but the model vocab_size is {vocab}")
    if checkpoint_path and vocab and len(tokenizer) < vocab // 2:
        raise ValueError(
            f"tokenizer at {path} has {len(tokenizer)} entries for a {vocab}-entry checkpoint; "
            "pass the checkpoint's own tokenizer with --tokenizer_path")
    print(f"[tokenizer] {path} ({len(tokenizer)} entries, model vocab {vocab})")
    return tokenizer
