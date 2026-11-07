"""Next-stage parameter recommendations shipped with a bundle.

Every value carries a ``source``:

* ``recipe``   – copied from the run's own RunSpec (what produced these weights);
* ``planner``  – ``chronos.model.planner`` output for the same topology
  (``ui/presets_v3.json``);
* ``rule``     – a stated derivation from one of the above (the rule is
  written next to the value), not a measurement.

Nothing here is a measured optimum.  The tiers trade wall time for quality
on a single Apple-silicon or CUDA machine; ``chronos bundle finetune --tier``
turns a tier into a ``chronos longrun`` spec.
"""
from __future__ import annotations

import copy
import json
import os

# Wall-clock budgets per tier (hours) and the reason for each.
TIERS = {
    "smoke": {"hours": 0.05, "why": "proves the pipeline end to end on the bundled examples (minutes)"},
    "quick": {"hours": 2.0, "why": "a small domain set (≈10k–100k pairs); one evening"},
    "standard": {"hours": 12.0, "why": "a medium set (≈0.1M–1M pairs); overnight"},
    "full": {"hours": 36.0, "why": "the SFT half of the original 72 h recipe"},
}

# Phase keys a longrun PhaseSpec accepts.
PHASE_KEYS = ("name", "budget_fraction", "peak_lr", "tokens_per_step", "micro_tokens", "seq_len",
              "warmup_fraction", "min_lr_ratio", "weight_decay", "grad_clip")


def _phase(recipe: dict, name: str) -> dict | None:
    for phase in recipe.get("phases", []):
        if phase.get("name") == name:
            return {k: phase[k] for k in PHASE_KEYS if k in phase}
    return None


def _labelled(values: dict, source: str, note: str | None = None) -> dict:
    out = {"values": values, "source": source}
    if note:
        out["note"] = note
    return out


def _planner_stages(planner_preset: dict | None) -> dict:
    stages = (planner_preset or {}).get("stage_overrides") or {}
    return {k: dict(v) for k, v in stages.items()}


def continued_pretrain_phase(recipe: dict) -> dict:
    """Re-warm the pretrain phase at 0.3× its peak LR.

    The bundled weights finished (or are partway through) a cosine decay;
    restarting at the full peak would undo part of it.  0.3× is a rule of
    thumb, not a tuned value."""
    phase = _phase(recipe, "pretrain") or {}
    phase = dict(phase, budget_fraction=1.0, peak_lr=round(float(phase.get("peak_lr", 1.5e-3)) * 0.3, 8),
                 warmup_fraction=max(0.01, float(phase.get("warmup_fraction", 0.02))))
    return phase


def sft_phase(recipe: dict) -> dict:
    phase = _phase(recipe, "sft")
    if phase is None:                     # a pretrain-only recipe: fall back to the 72 h SFT values
        phase = {"name": "sft", "peak_lr": 5e-4, "tokens_per_step": 65536, "micro_tokens": 8192,
                 "warmup_fraction": 0.01, "min_lr_ratio": 0.05, "weight_decay": 0.1, "grad_clip": 1.0}
    return dict(phase, budget_fraction=1.0)


def tier_phases(recipe: dict, tier: str, mode: str = "sft") -> list[dict]:
    """PhaseSpec dicts for one tier.

    ``mode``: ``sft`` (translation fine-tune), ``pretrain`` (continued
    pretraining on new text) or ``pretrain+sft`` (40 % / 60 % split)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose one of {sorted(TIERS)}")
    sft, cpt = sft_phase(recipe), continued_pretrain_phase(recipe)
    if tier == "smoke":
        # Tiny steps so a CPU finishes in minutes; LR as in the recipe.
        sft = dict(sft, tokens_per_step=8192, micro_tokens=2048, warmup_fraction=0.1)
        cpt = dict(cpt, tokens_per_step=8192, micro_tokens=2048, seq_len=min(int(cpt.get("seq_len", 512)), 256),
                   warmup_fraction=0.1)
    elif tier == "quick":
        # Small data: halve the step size so a 2 h budget still makes enough
        # updates, and warm up longer (rule: ≥ ~100 warmup steps).
        sft = dict(sft, tokens_per_step=max(16384, int(sft["tokens_per_step"]) // 2), warmup_fraction=0.03)
        cpt = dict(cpt, tokens_per_step=max(16384, int(cpt.get("tokens_per_step", 131072)) // 2),
                   warmup_fraction=0.03)
    if mode == "sft":
        return [sft]
    if mode == "pretrain":
        return [cpt]
    if mode == "pretrain+sft":
        return [dict(cpt, budget_fraction=0.4), dict(sft, budget_fraction=0.6)]
    raise ValueError(f"unknown mode {mode!r}; choose sft, pretrain or pretrain+sft")


def median_record_tokens(shards_dir: str, max_files: int = 8) -> float | None:
    """Median SFT record length (tokens) from the ``.idx`` offsets under ``<shards>/sft/train``.

    Reads at most ``max_files`` index files (offsets only, no token data).
    Returns None when the shards have no SFT records (packed pretrain text
    has no record boundaries)."""
    import glob
    import numpy as np
    files = sorted(glob.glob(os.path.join(shards_dir, "sft", "train", "*.idx")))[:max_files]
    lengths = [np.diff(np.fromfile(f, dtype=np.int64)) for f in files]
    lengths = [x for x in lengths if x.size]
    if not lengths:
        return None
    return float(np.median(np.concatenate(lengths)))


def cluster_locality_config(model_config: dict, shards_dir: str | None, *, enabled: bool = True,
                            lambda_cluster: float | None = None, window: int | None = None,
                            clusters: int | None = None, seq_len: int | None = None) -> dict:
    """Cluster-locality fields for a fine-tune and where each value comes from.

    On by default at ``DEFAULT_LAMBDA`` (measured sweep on a synthetic
    multi-topic task, from-scratch training; its effect on a fine-tune of
    weights trained without the constraint has not been measured).  Off for
    dense models and when there are fewer than two clusters."""
    from chronos.expert_clusters import DEFAULT_LAMBDA, recommended_window
    from chronos.model.config import ChronosConfig
    # The values training will use: longrun builds ChronosConfig(**model_config).
    cfg = ChronosConfig(**{k: v for k, v in model_config.items() if v is not None})
    e = int(getattr(cfg, "num_experts", 0) or 0)
    k = int(getattr(cfg, "num_experts_per_tok", 1) or 1)
    seq = int(seq_len or getattr(cfg, "max_position_embeddings", 512) or 512)
    usable = bool(getattr(cfg, "use_moe", True)) and e > 2 * k
    lam = 0.0 if not (enabled and usable) else float(DEFAULT_LAMBDA if lambda_cluster is None else lambda_cluster)
    if lam < 0:
        raise ValueError("lambda_cluster must be >= 0")
    median = median_record_tokens(shards_dir) if (shards_dir and window is None) else None
    win = int(window) if window is not None else recommended_window(median, seq)
    if win < 0:
        raise ValueError("cluster_window must be >= 0 (0 = whole sequence)")
    values = {"lambda_cluster_locality": lam, "cluster_locality_window": win,
              "cluster_locality_clusters": int(clusters or 0)}
    if not enabled:
        why = "off (--no-cluster-locality)"
    elif not usable:
        why = "off: dense model or fewer than two clusters"
    elif lambda_cluster is None:
        why = f"default {DEFAULT_LAMBDA} (measured sweep, synthetic task; not measured for fine-tuning)"
    else:
        why = "user value"
    if window is not None:
        wsrc = "user value"
    elif median is not None:
        wsrc = f"rule: power of two nearest the median SFT record ({median:.0f} tokens), clamped to [16, {seq}]"
    else:
        wsrc = "default (no SFT record lengths in the data)"
    return {"values": values, "source": {"lambda_cluster_locality": why, "cluster_locality_window": wsrc,
                                         "cluster_locality_clusters": "user value" if clusters else "ceil(E / 2k)"}}


def longrun_spec(bundle_dir: str, recipe: dict, *, run_dir: str, shards_dir: str, tier: str,
                 mode: str = "sft", budget_hours: float | None = None, seed: int | None = None,
                 cluster_locality: bool = True, lambda_cluster: float | None = None,
                 cluster_window: int | None = None, clusters: int | None = None) -> dict:
    """A ``chronos longrun`` RunSpec (as a dict) that starts from the bundle weights.

    The cluster-locality constraint is on by default (``cluster_locality_config``)."""
    from .builder import load_bundle_manifest
    manifest = load_bundle_manifest(bundle_dir)
    bundle_dir = os.path.abspath(bundle_dir)
    smoke = tier == "smoke"
    model_config = copy.deepcopy(recipe["model_config"])
    phases = tier_phases(recipe, tier, mode)
    seq_len = min((int(p["seq_len"]) for p in phases if p.get("seq_len")), default=None)
    # Decide from the weights being fine-tuned: the recipe may omit fields that
    # took class defaults when it ran.
    topology = dict(model_config)
    try:
        from chronos.model.checkpoint import read_checkpoint_config
        saved, _ = read_checkpoint_config(os.path.join(bundle_dir, manifest["model"]["checkpoint"]))
        topology.update({k: saved[k] for k in ("num_experts", "num_experts_per_tok", "use_moe",
                                               "max_position_embeddings") if saved and k in saved})
    except (OSError, ValueError):
        pass
    locality = cluster_locality_config(topology, shards_dir, enabled=cluster_locality,
                                       lambda_cluster=lambda_cluster, window=cluster_window, clusters=clusters,
                                       seq_len=seq_len)
    model_config.update(locality["values"])
    spec = {
        "run_dir": os.path.abspath(run_dir),
        "shards_dir": os.path.abspath(shards_dir),
        "tokenizer_dir": os.path.join(bundle_dir, manifest["tokenizer"]["dir"]),
        "model_config": model_config,
        "phases": phases,
        "budget_hours": float(budget_hours if budget_hours is not None else TIERS[tier]["hours"]),
        "eval_reserve_fraction": 0.1 if smoke else 0.04,
        "log_every_steps": 2 if smoke else 20,
        "validate_every_s": 60.0 if smoke else 1800.0,
        "checkpoint_every_s": 120.0 if smoke else 1800.0,
        "bleu_samples": 8 if smoke else 100,
        "final_eval_samples": 16 if smoke else 1012,
        "probe_samples": 4 if smoke else 16,
        "tensor_stats_every_s": 0.0 if smoke else 1800.0,
        "seed": int(seed if seed is not None else recipe.get("seed", 1234)),
        "dtype": recipe.get("dtype", "bfloat16"),
        "schedule": "time",
        "init_checkpoint": os.path.join(bundle_dir, manifest["model"]["checkpoint"]),
        "export_prefix": manifest["name"].replace("-", "_") + "_ft",
        "extra": {"raw_root": os.path.join(bundle_dir, "eval"), "from_bundle": manifest["name"],
                  "tier": tier, "mode": mode, "cluster_locality_source": locality["source"]},
    }
    return spec


def build_recommendations(recipe: dict, planner_preset: dict | None, decode: dict | None = None) -> dict:
    """The ``recommended.json`` payload."""
    planner = _planner_stages(planner_preset)
    tiers = {}
    for tier, meta in TIERS.items():
        tiers[tier] = {
            "hours": meta["hours"], "why": meta["why"],
            "phases": {mode: tier_phases(recipe, tier, mode) for mode in ("sft", "pretrain", "pretrain+sft")},
            "source": "rule",
            "rule": "SFT phase = recipe SFT phase; continued pretrain = recipe pretrain phase at 0.3x peak LR; "
                    "smoke uses 8k-token steps, quick halves tokens_per_step and warms up 3%",
        }
    torch_stages = {}
    for stage, values in planner.items():
        torch_stages[stage] = _labelled(values, "planner",
                                        "per-stage values for train_chronos_<stage>.py / the Web UI; "
                                        "batch_size counts sequences of max_seq_len tokens")
    out = {
        "about": "Next-stage parameters for this bundle. 'recipe' = what produced these weights; "
                 "'planner' = chronos.model.planner for the same topology; 'rule' = derived as stated; "
                 "'measured' = measured on the bundled weights (conditions given). None is a tuned optimum.",
        "mlx_longrun": {
            "tiers": tiers,
            "recipe_phases": _labelled({p["name"]: {k: p[k] for k in PHASE_KEYS if k in p}
                                        for p in recipe.get("phases", [])}, "recipe"),
            "model_lambdas": _labelled({k: v for k, v in recipe.get("model_config", {}).items()
                                        if k.startswith(("lambda_", "ponder", "fallback_mask"))}, "recipe"),
            "cluster_locality": _labelled(
                cluster_locality_config(recipe.get("model_config", {}), None)["values"], "rule",
                "on by default for `chronos bundle finetune` (--no-cluster-locality to disable); lambda from "
                "a measured sweep on a synthetic task (from-scratch training), window from the median SFT "
                "record length of the fine-tune data when it has one"),
        },
        "torch_stages": torch_stages,
        "data": _labelled({
            "pretrain_mix": {"mono": 0.5, "en_zh": 0.25, "zh_en": 0.25},
            "sft_max_len": 512,
            "clean": {"max_chars": 1024, "max_len_ratio": 3.0, "min_han_ratio": 0.3, "min_latin_ratio": 0.4,
                      "zh_fullwidth_punct": True},
            "min_pairs_hint": "below ~5k pairs prefer tier 'quick' with mode 'sft'; the 50M model overfits "
                              "small sets quickly — watch valid_ce in metrics.jsonl",
        }, "recipe", "defaults of chronos.data.corpus (build_corpus / build_shards) used for the bundled weights"),
    }
    if decode:
        out["decode"] = decode
    return out


def write_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path
