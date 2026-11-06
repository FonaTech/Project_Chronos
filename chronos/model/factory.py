"""Parameter-budget model factory.

``fit_config(target_total, target_active, ...)`` searches the Chronos design
space (width, expert count, expert width) for a valid configuration whose
*allocated* parameter counts are closest to the targets.  Counts come from
instantiating the real model on the ``meta`` device, so they always match
what training allocates — no separate formula to drift.

Active parameters follow the usual MoE convention: everything except the
routed experts, plus ``top_k`` routed experts per MoE layer.  Because loop
iterations may route differently, the factory also reports the union upper
bound ``min(E, top_k·depth)`` experts for recurrent layers.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch


@dataclass
class FitResult:
    config: dict
    total: int
    active: int
    active_union_upper: int
    embedding: int
    per_expert: int
    score: float


def count_parameters(config_kwargs: dict) -> dict:
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    cfg = ChronosConfig(**config_kwargs)
    with torch.device("meta"):
        model = ChronosForCausalLM(cfg)
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p)); total += p.numel()
    routed, per_expert, moe_layers = 0, 0, 0
    for module in model.modules():
        if isinstance(module, ChronosMOEFeedForward):
            moe_layers += 1
            per_expert = sum(p.numel() for p in module.experts[0].parameters())
            routed += per_expert * len(module.experts)
    k, E = int(cfg.num_experts_per_tok), int(cfg.num_experts)
    active = total - routed + moe_layers * k * per_expert
    union_extra = int(cfg.recurrent_groups) * (min(E, k * int(cfg.recurrent_depth)) - k) * per_expert
    return {"total": total, "active": active, "active_union_upper": active + union_extra,
            "embedding": int(model.model.embed_tokens.weight.numel()), "per_expert": per_expert,
            "virtual_layers": int(cfg.virtual_num_hidden_layers)}


def fit_config(target_total: float, target_active: float, *, vocab_size: int,
               base: dict | None = None, hidden_sizes=(320, 352, 384, 416, 448, 512),
               expert_counts=tuple(range(8, 49, 4)), expert_widths=(160, 192, 224, 256, 288, 320),
               head_dim: int = 64) -> FitResult:
    """Grid search minimizing relative error on (total, active)."""
    base = dict(base or {})
    best = None
    for h, e, i in itertools.product(hidden_sizes, expert_counts, expert_widths):
        heads = max(1, h // head_dim)
        kv = 2 if heads % 2 == 0 else 1
        kw = {**base, "hidden_size": h, "num_attention_heads": heads, "num_key_value_heads": kv,
              "num_experts": e, "moe_intermediate_size": i, "vocab_size": vocab_size}
        try:
            c = count_parameters(kw)
        except (ValueError, AssertionError):
            continue
        score = abs(c["total"] / target_total - 1) + 2.0 * abs(c["active"] / target_active - 1)
        if best is None or score < best.score:
            best = FitResult(kw, c["total"], c["active"], c["active_union_upper"], c["embedding"],
                             c["per_expert"], score)
    if best is None:
        raise ValueError("no valid configuration in the search space")
    return best


ENZH_DEMO_BASE = dict(
    architecture_version=3, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
    recurrent_depth=4, min_recurrent_depth=1, max_inference_depth=4,
    recurrent_attention_mode="shared_first", adaptive_halting=True,
    depth_schedule="fixed_then_adaptive", depth_schedule_warmup_fraction=0.3,
    halting_threshold=0.95, num_experts_per_tok=2, num_shared_experts=1,
    kv_latent_dim=128, rope_dim=32, rope_layout="per_head", sliding_window_size=128,
    max_position_embeddings=1024,
    layer_aware_lookahead=True, layer_aware_cache=True, lookahead_steps=2,
    dropout=0.0, fallback_mask_prob=0.08, rope_theta=1e6,
    bos_token_id=1, eos_token_id=2, tie_word_embeddings=True,
)
