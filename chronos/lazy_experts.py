"""SSD-lazy routed experts: one implementation for the Web UI, bundle and server.

The page layout adapts to how the checkpoint was trained.  Weights trained
with the cluster-locality constraint (``lambda_cluster_locality > 0``) keep a
stream inside one training cluster, so they are paged one cluster file per
read; other weights are paged one expert per read, which never loads
experts the stream does not use (``chronos.expert_clusters.inference_clusters``).
The layout is part of the cache-directory name, so a cache written for one
layout is never reused for the other.

Residency (what stays loaded) is independent of the layout: ``"planned"``
evicts the page needed latest by the next-use planner, ``"window"`` keeps the
LRU order with the lookahead window protected.  Neither changes the output.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass

from chronos.expert_clusters import PAGING_MODES, inference_clusters, paging_tag, trained_with_clusters

RESIDENCY_MODES = ("planned", "window")
# Defaults shared by every entry point (the engine's own defaults).
DEFAULT_RESIDENCY = "planned"
DEFAULT_RESIDENCY_DECAY = 0.7
DEFAULT_DECODE_WAIT_S = 0.012
DEFAULT_VERIFY_WAIT_S = 0.02
DEFAULT_RAM_RATIO = 1.0


@dataclass
class LazyOptions:
    """User-facing lazy-expert settings; every field has a working default."""

    paging: str = "auto"                         # auto | cluster | expert
    residency: str = DEFAULT_RESIDENCY           # planned | window
    residency_decay: float = DEFAULT_RESIDENCY_DECAY
    decode_wait_s: float = DEFAULT_DECODE_WAIT_S
    verify_wait_s: float = DEFAULT_VERIFY_WAIT_S
    ram_ratio: float = DEFAULT_RAM_RATIO

    def validated(self) -> "LazyOptions":
        if self.paging not in PAGING_MODES:
            raise ValueError(f"paging must be one of {PAGING_MODES}, not {self.paging!r}")
        if self.residency not in RESIDENCY_MODES:
            raise ValueError(f"residency must be one of {RESIDENCY_MODES}, not {self.residency!r}")
        decay = float(self.residency_decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("residency_decay must be in [0, 1)")
        waits = float(self.decode_wait_s), float(self.verify_wait_s)
        if min(waits) < 0:
            raise ValueError("decode_wait_s and verify_wait_s must be >= 0")
        ratio = float(self.ram_ratio)
        if not ratio > 0:
            raise ValueError("ram_ratio must be > 0")
        return LazyOptions(self.paging, self.residency, decay, waits[0], waits[1], ratio)

    def generate_kwargs(self) -> dict:
        """Keyword arguments for ``ChronosMLXInferenceEngine.generate``."""
        return {"residency": self.residency, "residency_decay": self.residency_decay,
                "decode_wait_s": self.decode_wait_s, "verify_wait_s": self.verify_wait_s}

    def as_dict(self) -> dict:
        return asdict(self)


def page_layout(config, paging: str = "auto") -> dict:
    """Resolved layout for a checkpoint: clusters, tag and the reason (for UI/stats)."""
    clusters = inference_clusters(config, paging) or []
    tag = paging_tag(clusters) if clusters else "e"
    trained = trained_with_clusters(config)
    if paging == "auto":
        reason = "trained with cluster locality" if trained else "trained without cluster locality"
    else:
        reason = f"forced {paging}"
    return {"paging": paging, "resolved": "expert" if tag == "e" else "cluster", "clusters": clusters,
            "tag": tag, "num_pages": len(clusters), "trained_with_clusters": trained, "reason": reason}


# ── cache location ────────────────────────────────────────────────

def checkpoint_cache_dir(model_path: str, config, *, backend: str = "torch", tag: str = "e",
                         root: str | None = None) -> str:
    """Per-checkpoint expert cache directory.

    The key is path + size + mtime + core topology, so a changed checkpoint
    gets a fresh namespace.  ``tag`` is the page layout (``paging_tag``):
    per-expert caches keep their historical names (``ckpt_<key>`` for Torch,
    ``ckpt_<key>_mlx`` for MLX); cluster layouts add ``_<tag>``.
    """
    base = root or os.path.join(os.getcwd(), "expert_cache")
    if not model_path or not os.path.exists(model_path):
        name_dir = base
    else:
        try:
            st = os.stat(model_path)
        except OSError:
            name_dir = base
        else:
            topology = tuple(getattr(config, k, "") for k in (
                "vocab_size", "hidden_size", "num_hidden_layers", "num_experts", "moe_intermediate_size"))
            raw = f"{os.path.abspath(model_path)}:{st.st_size}:{st.st_mtime_ns}:{topology}"
            name_dir = os.path.join(base, "ckpt_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16])
    if backend == "mlx":
        name_dir += "_mlx"
    if tag and tag != "e":
        name_dir += f"_{tag}"
    return name_dir


# ── expert budget ─────────────────────────────────────────────────

def active_expert_count(config) -> int:
    """Routed working set per token: observed/recommended size, else top-k × layers."""
    num_experts = int(getattr(config, "num_experts", 1) or 1)
    for attr in ("observed_active_experts", "offload_working_set_experts", "recommended_working_set_experts"):
        value = getattr(config, attr, None)
        if value in (None, "", 0):
            continue
        try:
            return max(1, min(num_experts, int(value)))
        except (TypeError, ValueError):
            pass
    top_k = int(getattr(config, "num_experts_per_tok", 1) or 1)
    num_layers = int(getattr(config, "num_hidden_layers", 1) or 1)
    # top_k is per layer; the cache is keyed by global expert id, so one token
    # can touch up to top_k × layers distinct ids before repeats.
    return max(1, min(num_experts, top_k * max(1, num_layers)))


def expert_budget(config, ram_ratio: float | str | None = 1.0) -> dict:
    """Expert-page budget from the routed working set and the user ratio.

    The ratio produces real lower-budget runs for sweeps; output quality is
    preserved by on-demand loading, never by clamping to top-k.
    """
    active = active_expert_count(config)
    num_experts = int(getattr(config, "num_experts", active) or active)
    try:
        ratio = float(ram_ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    ratio = max(0.01, ratio)
    routing_top_k = int(getattr(config, "num_experts_per_tok", active) or active)
    hard_cap = max(active, int(math.floor(active * 1.25)))
    requested = max(1, int(math.ceil(active * ratio)))
    effective = max(1, min(num_experts, requested, hard_cap))
    return {
        "ideal_active_experts": active,
        "routing_top_k": routing_top_k,
        "num_moe_layers": int(getattr(config, "num_hidden_layers", 1) or 1),
        "requested_ram_load_ratio": round(ratio, 3),
        "requested_expert_budget": requested,
        "max_allowed_expert_budget": min(num_experts, hard_cap),
        "effective_expert_budget": effective,
        "effective_vram_expert_budget": effective,
        "effective_ram_expert_budget": effective,
        "min_vram_expert_budget": 1,
        "budget_clamped": effective != requested,
    }


def apply_torch_budget(engine, config, ram_ratio: float | str | None = 1.0) -> dict:
    """Set the Torch store's VRAM/RAM page capacity; a hard budget always wins."""
    budget = expert_budget(config, ram_ratio)
    vram_cap = int(budget["effective_vram_expert_budget"])
    ram_cap = int(budget["effective_ram_expert_budget"])
    store = getattr(getattr(engine, "cache_manager", None), "expert_store", None)
    if store is None:
        return budget
    hard_vram = getattr(store, "max_resident_pages", None)
    hard_ram = getattr(store, "ram_budget_pages", None)
    if hard_vram is not None:
        vram_cap = max(1, min(vram_cap, int(hard_vram)))
    if hard_ram is not None:
        ram_cap = max(1, min(ram_cap, int(hard_ram)))
    budget["effective_vram_expert_budget"], budget["effective_ram_expert_budget"] = vram_cap, ram_cap
    store.vram_capacity = vram_cap
    store.ram_capacity = ram_cap
    store.vram_lru.capacity = vram_cap
    store.ram_lru.capacity = ram_cap
    return budget


# ── live-expert placeholders ──────────────────────────────────────

def torch_experts_to_placeholders(model, config) -> None:
    """Drop the constructor's random experts before ``model.to(device)``."""
    from chronos.model.moe_chronos import ChronosMOEFeedForward, LazyExpertPlaceholder
    from chronos.precision import torch_model_dtype

    hidden = int(getattr(config, "hidden_size", 0) or 0)
    intermediate = int(getattr(config, "moe_intermediate_size", 0) or 0)
    if not hidden or not intermediate:
        return
    for layer in getattr(getattr(model, "model", None), "layers", []):
        moe = getattr(layer, "mlp", None)
        if not isinstance(moe, ChronosMOEFeedForward):
            continue
        for idx in range(len(moe.experts)):
            # Placeholder dtype follows the trunk; place_torch_model() updates it on move.
            moe.experts[idx] = LazyExpertPlaceholder(hidden, intermediate, torch_model_dtype(model))


def mlx_experts_to_placeholders(model) -> None:
    import mlx.core as mx
    from chronos.mlx.moe import ChronosMLXMOE, LazyFeedForwardMLX

    for layer in getattr(model, "layers", []):
        moe = getattr(layer, "mlp", None)
        if not isinstance(moe, ChronosMLXMOE):
            continue
        for idx in range(len(moe.experts)):
            moe.experts[idx] = LazyFeedForwardMLX()
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


# ── MLX expert cache ──────────────────────────────────────────────

def _store_torch_dtype(store):
    """Torch dtype of the MLX store's pages (= the MLX trunk compute dtype)."""
    import torch
    name = str(getattr(store, "page_dtype", "float32")).rsplit(".", 1)[-1]
    return getattr(torch, name)


def write_mlx_cache(store, get_tensor, keys, clusters=None) -> None:
    """Write the MLX store's cache: one ``cluster_<c>.ctsr`` per page + manifest.

    ``get_tensor(key)`` returns a checkpoint tensor (``model.layers.L.mlp.experts.E.P``);
    ``keys`` is the set of checkpoint keys.  ``clusters`` defaults to one page per expert.
    Only one page's tensors are held at a time.
    """
    from safetensors.torch import save_file  # BF16-capable
    from chronos.io.storage import ClusterManifest, KEY_SEP, MANIFEST_FILENAME

    num_experts, num_layers = int(store.num_experts), int(store.num_layers)
    if clusters is None:
        clusters = [[i] for i in range(num_experts)]
    os.makedirs(store.ssd_dir, exist_ok=True)
    page_dtype = _store_torch_dtype(store)
    keys = set(keys)
    manifest_clusters, expert_to_cluster = {}, {}
    for cid, members in enumerate(clusters):
        tensors = {}
        for eid in members:
            for li in range(num_layers):
                for pname in store._PARAM_NAMES:
                    key = f"model.layers.{li}.mlp.experts.{int(eid)}.{pname}"
                    if key in keys:
                        tensors[f"l{li}_e{int(eid)}{KEY_SEP}{pname}"] = (
                            get_tensor(key).detach().to(dtype=page_dtype, device="cpu").contiguous())
        if not tensors:
            continue
        file_name = f"cluster_{cid}.ctsr"
        save_file(tensors, os.path.join(store.ssd_dir, file_name))
        manifest_clusters[cid] = (file_name, [int(e) for e in members])
        for eid in members:
            expert_to_cluster[int(eid)] = cid
    if len(expert_to_cluster) != num_experts:
        raise RuntimeError(f"MLX lazy cache is incomplete: wrote {len(expert_to_cluster)}/{num_experts} experts")
    manifest = ClusterManifest(version=1, num_experts=num_experts, num_layers=num_layers,
                               storage_format="safetensors", clusters=manifest_clusters,
                               expert_to_cluster=expert_to_cluster)
    with open(os.path.join(store.ssd_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def write_mlx_cache_from_state_dict(store, state_dict: dict, clusters=None) -> None:
    write_mlx_cache(store, state_dict.__getitem__, state_dict.keys(), clusters)


def write_mlx_cache_from_reader(store, reader, clusters=None) -> None:
    write_mlx_cache(store, reader.get_tensor, reader.keys(), clusters)


def _manifest_matches(store, clusters) -> bool:
    manifest = getattr(store, "_cluster_manifest", None)
    if manifest is None:
        return False
    have = sorted(sorted(int(e) for e in members) for _f, members in manifest.clusters.values())
    return have == sorted(sorted(int(e) for e in c) for c in clusters)


def ensure_mlx_cache(store, cache_dir: str, *, reader=None, weights=None, clusters=None) -> bool:
    """Attach a complete cache with this layout at ``cache_dir`` or rebuild it.

    Returns True when an existing cache was reused.
    """
    import shutil

    if clusters is None:
        clusters = [[i] for i in range(int(store.num_experts))]
    if store.attach_cluster_manifest(cache_dir) and _manifest_matches(store, clusters):
        return True
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    store.ssd_dir = cache_dir
    store._cluster_manifest = None
    store._loaded_clusters.clear()
    store._warm.clear()
    store._warm_lru.clear()
    if reader is not None:
        write_mlx_cache_from_reader(store, reader, clusters)
    elif weights is not None:
        write_mlx_cache_from_state_dict(store, weights, clusters)
    else:
        store.offload_all_to_ssd(clusters=clusters)
    if not (store.attach_cluster_manifest(cache_dir) and _manifest_matches(store, clusters)):
        raise RuntimeError(f"failed to build a complete MLX lazy expert cache at {cache_dir}")
    return False


def build_mlx_lazy_engine(model_path: str, config, options: LazyOptions | None = None, *,
                          cache_root: str | None = None, dtype: str = "auto"):
    """Load a checkpoint or export into an MLX engine with SSD-lazy experts.

    Only the trunk is materialised; experts are written to the page cache
    (layout from ``options.paging`` and the checkpoint) and stream in on demand.
    Sets ``config.recommended_resident_experts`` / ``recommended_ram_experts``
    from the budget.  Returns ``(engine, info)``; the caller owns ``engine.stop()``.
    """
    from chronos.export import is_export_artifact, open_export_reader
    from chronos.mlx.inference import ChronosMLXInferenceEngine
    from chronos.mlx.model import ChronosMLXModel
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.precision import resolve_mlx

    opts = (options or LazyOptions()).validated()
    layout = page_layout(config, opts.paging)
    budget = expert_budget(config, opts.ram_ratio)
    config.recommended_resident_experts = int(budget["effective_vram_expert_budget"])
    config.recommended_ram_experts = int(budget["effective_ram_expert_budget"])
    cache_dir = checkpoint_cache_dir(model_path, config, backend="mlx", tag=layout["tag"], root=cache_root)

    reader = weights = None
    if model_path and is_export_artifact(model_path):
        reader = open_export_reader(model_path)
        base_only = reader.load_state_dict(include_experts=False)
    elif model_path and os.path.exists(model_path):
        weights = load_checkpoint_state_dict(model_path, map_location="cpu")
        base_only = {k: v for k, v in weights.items() if ".mlp.experts." not in k}
    else:
        raise FileNotFoundError("lazy-expert inference needs a checkpoint or export artifact")
    engine = None
    try:
        pt_model = ChronosForCausalLM(config)
        if base_only:
            load_state_dict_controlled(pt_model, base_only, allow_missing_substrings=(".mlp.experts.",))
        torch_experts_to_placeholders(pt_model, config)
        precision = resolve_mlx(dtype)
        mlx_model = ChronosMLXModel.from_chronos_pytorch(pt_model.eval(), config, include_experts=False,
                                                         dtype=precision.compute)
        mlx_experts_to_placeholders(mlx_model)
        del pt_model, base_only
        gc.collect()
        engine = ChronosMLXInferenceEngine(mlx_model, config, ssd_dir=cache_dir)
        reused = ensure_mlx_cache(engine.store, cache_dir, reader=reader, weights=weights,
                                  clusters=layout["clusters"])
    except BaseException:
        if engine is not None:
            engine.stop()
        raise
    finally:
        reader = weights = None
        gc.collect()
    info = {**budget, "cache_dir": cache_dir, "cache_reused": reused,
            "paging": layout["paging"], "paging_resolved": layout["resolved"], "paging_reason": layout["reason"],
            "num_pages": layout["num_pages"], "trained_with_clusters": layout["trained_with_clusters"],
            "lazy_options": opts.as_dict(), "precision": precision.as_dict()}
    return engine, info


def warm_ids(config, budget: dict) -> list[int]:
    """Experts warmed before the first token (the execution budget, lowest ids first)."""
    count = int(budget["effective_vram_expert_budget"])
    return list(range(min(int(getattr(config, "num_experts", 0) or 0), count)))
