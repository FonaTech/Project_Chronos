"""
chronos/runtime/inference_engine.py

End-to-end Chronos inference engine with:
- Async prefetch driven by LookaheadRouter
- Soft gating fallback via CacheManager availability mask
- Token/s throughput measurement
- Optional streamer support

Usage:
    engine = ChronosInferenceEngine(model, config, ssd_dir="./expert_cache")
    engine.setup()  # offload experts to SSD, warm up cache
    output_ids = engine.generate(input_ids, max_new_tokens=200)
    print(engine.last_stats)
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import time
from typing import Dict, Hashable, List, Optional

import torch
import torch.nn.functional as F

try:
    from chronos.runtime.metrics import safe_record as _metric
except Exception:  # pragma: no cover
    def _metric(_name, _value):
        return None

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.moe_chronos import ChronosMOEFeedForward
from chronos.runtime.cache_manager import CacheManager
from chronos.router.prefill_scheduler import PrefillScheduler


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def _backend_memory_snapshot(device) -> dict:
    dtype = str(device).split(":", 1)[0]
    try:
        from chronos.backend.mac_diagnostics import mps_memory_snapshot, mlx_memory_snapshot
    except Exception:
        return {}
    if dtype == "mps":
        return mps_memory_snapshot()
    if dtype == "mlx":
        return mlx_memory_snapshot()
    return {}


def _backend_memory_fields(snapshot: dict, suffix: str) -> dict:
    out = {}
    for key in (
        "mps_allocated_gb",
        "mps_driver_allocated_gb",
        "mlx_active_gb",
        "mlx_cache_gb",
        "mlx_peak_gb",
    ):
        if key in snapshot:
            prefix = key[:-3] if key.endswith("_gb") else key
            out[f"{prefix}_{suffix}_gb"] = round(float(snapshot[key]), 6)
    return out


class ChronosInferenceEngine:
    """
    Wraps ChronosForCausalLM with the full async prefetch + soft gating pipeline.

    The decode loop:
    1. Get availability mask from CacheManager
    2. Forward pass with mask → soft gating handles cache misses
    3. Extract LookaheadRouter probs from output
    4. Submit future expert predictions to prefetcher (non-blocking)
    5. Repeat
    """

    def __init__(
        self,
        model: ChronosForCausalLM,
        config: ChronosConfig,
        ssd_dir: str = "./expert_cache",
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.cache_manager = CacheManager(model, config, ssd_dir=ssd_dir)
        self.last_stats: dict = {}
        self._runtime_stats: dict = {}
        self._route_cache_scores: Dict[Hashable, float] = {}
        self._last_prefill_ram_expert_ids: List[int] = []

    def _moe_layers(self) -> List[ChronosMOEFeedForward]:
        return [
            l.mlp for l in self.model.model.layers
            if isinstance(l.mlp, ChronosMOEFeedForward)
        ]

    @staticmethod
    def _normalize_miss_policy(miss_policy: str | None) -> str:
        policy = (miss_policy or "on_demand").strip().lower().replace("-", "_")
        aliases = {
            "quality_safe": "sync_on_demand",
            "quality": "sync_on_demand",
            "safe": "sync_on_demand",
            "strict": "sync_on_demand",
            "sync": "sync_on_demand",
            "blocking": "sync_on_demand",
            "fallback": "fallback_diagnostic",
            "diagnostic": "fallback_diagnostic",
        }
        policy = aliases.get(policy, policy)
        return policy if policy in {"on_demand", "sync_on_demand", "fallback_diagnostic"} else "on_demand"

    def _install_moe_runtime_hooks(self, miss_policy: str):
        store = self.cache_manager.expert_store

        def cache_key(layer_id: int, expert_id: int):
            return (
                store.expert_key(layer_id, expert_id)
                if bool(getattr(self.config, "layer_aware_cache", False))
                else int(expert_id)
            )

        def load_one(layer_id: int, expert_id: int) -> dict:
            expert_id = int(expert_id)
            key = cache_key(layer_id, expert_id)
            already_vram = store.vram_lru.contains(key)
            already_ram = store.ram_lru.contains(key)
            t0 = time.monotonic()
            ok = False
            try:
                if already_vram or already_ram or miss_policy == "sync_on_demand":
                    promoted = self.cache_manager.ensure_resident([key])
                    ok = store.vram_lru.contains(key)
                else:
                    # README architecture: cold SSD misses must not enter the
                    # decode critical path. Queue the exact missing expert for
                    # RAM prefetch and let shared fallback cover this token.
                    self.cache_manager.prefetch_experts_to_ram([key])
                    promoted = []
                    ok = False
            finally:
                elapsed = time.monotonic() - t0
                self._runtime_stats["on_demand_loads"] += 0 if already_vram else 1
                self._runtime_stats["on_demand_load_time_s"] += elapsed
                if already_vram:
                    self._runtime_stats["resident_vram_hits"] += 1
                elif already_ram:
                    self._runtime_stats["resident_ram_hits"] += 1
                else:
                    if miss_policy == "on_demand":
                        self._runtime_stats["async_cold_miss_prefetches"] = (
                            int(self._runtime_stats.get("async_cold_miss_prefetches", 0)) + 1
                        )
                    else:
                        self._runtime_stats["sync_ssd_loads"] = (
                            int(self._runtime_stats.get("sync_ssd_loads", 0)) + 1
                        )
                if ok:
                    self._runtime_stats["on_demand_loaded_experts"].append(key)
                    if promoted:
                        self._runtime_stats["on_demand_promoted_experts"].extend(promoted)
            return {"ok": ok, "already_vram": already_vram, "already_ram": already_ram}

        def touch_one(layer_id: int, expert_id: int) -> None:
            store.touch_expert(cache_key(layer_id, int(expert_id)))

        for layer_id, moe in enumerate(self._moe_layers()):
            moe.runtime_miss_policy = miss_policy
            moe.runtime_on_demand_loader = (
                (lambda expert_id, li=layer_id: load_one(li, expert_id))
                if miss_policy in {"on_demand", "sync_on_demand"} else None
            )
            moe.runtime_touch_expert = lambda expert_id, li=layer_id: touch_one(li, expert_id)

    def _clear_moe_runtime_hooks(self):
        for moe in self._moe_layers():
            moe.runtime_miss_policy = "fallback_diagnostic"
            moe.runtime_on_demand_loader = None
            moe.runtime_touch_expert = None

    def _consume_moe_runtime_stats(self) -> dict:
        out = {
            "on_demand_experts": [],
            "fallback_experts": [],
            "on_demand_weight_mass": 0.0,
            "fallback_executed_weight_mass": 0.0,
        }
        for moe in self._moe_layers():
            out["on_demand_experts"].extend(getattr(moe, "last_on_demand_experts", []) or [])
            out["fallback_experts"].extend(getattr(moe, "last_fallback_experts", []) or [])
            out["on_demand_weight_mass"] += float(getattr(moe, "last_on_demand_weight_mass", 0.0) or 0.0)
            out["fallback_executed_weight_mass"] += float(getattr(moe, "last_fallback_weight_mass", 0.0) or 0.0)
        return out

    def _recent_route_scores(
        self,
        current_ids: List[Hashable],
        predicted_ids: List[Hashable],
        ram_predicted_ids: Optional[List[Hashable]] = None,
    ) -> Dict[Hashable, float]:
        """Decayed route scores used as a sticky eviction priority.

        Higher scores stay hotter in both RAM and VRAM. This intentionally
        spans tokens; replacing it with only the current step turns tiny
        budgets into a promote/evict loop.
        """
        def normalized(key):
            if isinstance(key, (tuple, list)):
                return tuple(int(value) for value in key)
            return int(key)

        decayed = {
            normalized(eid): float(score) * 0.85
            for eid, score in self._route_cache_scores.items()
            if float(score) * 0.85 >= 0.05
        }
        scores = decayed
        for eid in current_ids:
            key = normalized(eid)
            scores[key] = scores.get(key, 0.0) + 3.0
        for eid in predicted_ids:
            key = normalized(eid)
            scores[key] = scores.get(key, 0.0) + 2.0
        for eid in ram_predicted_ids or []:
            key = normalized(eid)
            scores[key] = scores.get(key, 0.0) + 1.0
        self._route_cache_scores = scores
        return scores

    def _router_topk(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the MoE top-k ids/weights using the model's routing policy."""
        top_k = int(getattr(self.config, "num_experts_per_tok", 1) or 1)
        top_k = max(1, min(top_k, probs.shape[-1]))
        topk_weight, topk_idx = torch.topk(probs, k=top_k, dim=-1, sorted=False)
        if bool(getattr(self.config, "norm_topk_prob", False)):
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weight, topk_idx

    @staticmethod
    def _ordered_ids_from_scores(scores: torch.Tensor, capacity: int) -> List[int]:
        if scores.numel() == 0 or capacity <= 0 or not torch.any(scores > 0):
            return []
        ordered = torch.argsort(scores, descending=True).tolist()
        return [int(eid) for eid in ordered if float(scores[eid].item()) > 0.0][:capacity]

    def _ordered_sticky_ids(self, capacity: int) -> List[Hashable]:
        if capacity <= 0 or not self._route_cache_scores:
            return []
        return [
            eid
            for eid, _score in sorted(
                self._route_cache_scores.items(),
                key=lambda item: (-float(item[1]), repr(item[0])),
            )
        ][:capacity]

    def _promotion_ready_ids(self, expert_ids: List[Hashable]) -> List[Hashable]:
        """Only return IDs already staged in RAM/VRAM.

        Calling ensure_resident on a cold SSD expert during decode would
        synchronously read from disk. The predictive path keeps SSD IO in
        the background and promotes only warm experts.
        """
        store = self.cache_manager.expert_store
        ready = []
        normalized = (
            tuple(int(value) for value in eid) if isinstance(eid, (tuple, list)) else int(eid)
            for eid in expert_ids or []
        )
        for eid in dict.fromkeys(normalized):
            if store.vram_lru.contains(eid) or store.ram_lru.contains(eid):
                ready.append(eid)
        return ready

    def setup(self, warm_expert_ids: Optional[List[int]] = None):
        """
        One-time setup: serialize experts to SSD, start prefetcher, warm cache.
        Call once before the first generate().
        """
        print("[Chronos] Offloading expert weights to SSD...")
        self.cache_manager.expert_store.offload_all_to_ssd()
        self.cache_manager.start()
        print("[Chronos] Warming up VRAM cache...")
        self.cache_manager.warm_up(warm_expert_ids)
        print(f"[Chronos] Ready. Cache stats: {self.cache_manager.stats()}")

    def setup_from_state_dict(
        self,
        state_dict: dict,
        warm_expert_ids: Optional[List[int]] = None,
    ):
        """Setup path for real lazy expert loading on CPU/MPS.

        The caller loads base/non-expert weights into the live model, then
        hands the full checkpoint here so ExpertStore can shard only expert
        tensors to SSD and replace live experts with placeholders.
        """
        print("[Chronos] Offloading expert weights to SSD from checkpoint...")
        self.cache_manager.expert_store.offload_from_state_dict(state_dict)
        self.cache_manager.start()
        print("[Chronos] Warming up VRAM cache...")
        self.cache_manager.warm_up(warm_expert_ids)
        print(f"[Chronos] Ready. Cache stats: {self.cache_manager.stats()}")

    def setup_from_state_reader(
        self,
        reader,
        warm_expert_ids: Optional[List[int]] = None,
    ):
        """Setup lazy expert loading from a safetensors/GGUF export reader."""
        print("[Chronos] Offloading expert weights to SSD from export reader...")
        self.cache_manager.expert_store.offload_from_state_reader(reader)
        self.cache_manager.start()
        print("[Chronos] Warming up VRAM cache...")
        self.cache_manager.warm_up(warm_expert_ids)
        print(f"[Chronos] Ready. Cache stats: {self.cache_manager.stats()}")

    def teardown(self):
        self.cache_manager.stop()

    @staticmethod
    def _virtual_router_probs(moe, last_token: bool = False) -> Optional[torch.Tensor]:
        """Router probs of every dispatch of ``moe`` in the last forward.

        A recurrent group's MoE runs once per loop iteration and each
        iteration may pick different experts, so prefetch/residency must
        cover the union over all D iterations (virtual layers), not only
        the last one.  Returns ``[B, S*D, E]`` (``[B, D, E]`` for
        ``last_token``) on the CPU, or None before the first forward.
        """
        probs = list(getattr(moe, "iteration_router_probs", None) or ())
        if not probs:
            probs = [moe.last_router_probs] if moe.last_router_probs is not None else []
        if not probs:
            return None
        if last_token:
            probs = [p[:, -1:, :] for p in probs]
        return torch.cat([p.detach() for p in probs], dim=1).cpu()

    def _get_current_expert_ids(self, capacity: Optional[int] = None) -> List[int]:
        """Extract active top-k expert IDs from every MoE layer.

        Older code only looked at layer 0, which made the lazy path report
        false cache hits whenever later layers routed to cold experts.
        """
        moe_layers = self._moe_layers()
        num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        if not moe_layers or num_experts <= 0:
            return []
        scores = torch.zeros(num_experts, dtype=torch.float32)
        for moe in moe_layers:
            probs = self._virtual_router_probs(moe, last_token=True)   # [B, D, E]
            if probs is None:
                continue
            weights, ids = self._router_topk(probs.reshape(-1, probs.shape[-1]))
            scores.index_add_(0, ids.reshape(-1), weights.reshape(-1).to(scores.dtype))
        if capacity is None:
            capacity = num_experts
        return self._ordered_ids_from_scores(scores, min(int(capacity), num_experts))

    def _predict_expert_ids(
        self,
        lookahead_probs: Optional[torch.Tensor],
        future_offset: int = 1,
        capacity: Optional[int] = None,
        all_future: bool = False,
    ) -> List[int]:
        """Select globally hot expert IDs from LookaheadRouter predictions."""
        if lookahead_probs is None:
            return []
        probs = lookahead_probs.detach().cpu()
        if probs.dim() not in (4, 5) or probs.shape[-1] == 0:
            return []
        if probs.dim() == 5 and not bool(getattr(self.config, "layer_aware_cache", False)):
            probs = probs.mean(dim=3)
        num_experts = int(probs.shape[-1])
        layer_aware = probs.dim() == 5 and bool(getattr(self.config, "layer_aware_cache", False))
        if layer_aware:
            # Return physical-layer page keys.  The recurrent group is encoded
            # by the prediction axis and ExpertStore validates the mapping.
            group_axis = probs.shape[-2]
            layer_map = getattr(self.cache_manager.expert_store, "_layer_to_group", {})
            if all_future:
                future = probs[:, -1, 1:, :, :]
                offsets = range(future.shape[1])
            else:
                if probs.shape[2] <= future_offset:
                    return []
                future = probs[:, -1, future_offset:future_offset + 1, :, :]
                offsets = [0]
            selected = []
            for offset in offsets:
                for group in range(group_axis):
                    step_probs = future[:, offset, group, :]
                    weights, ids = self._router_topk(step_probs)
                    scale = 1.0 / float(offset + 1) if all_future else 1.0
                    for eid, weight in zip(ids.reshape(-1).tolist(), weights.reshape(-1).tolist()):
                        selected.append(((group, int(eid)), float(weight) * scale))
            score_map = {}
            for (group, eid), score in selected:
                score_map[(group, eid)] = score_map.get((group, eid), 0.0) + score
            capacity = num_experts if capacity is None else int(capacity)
            ordered = sorted(score_map, key=lambda key: (-score_map[key], key[0], key[1]))
            out = []
            for group, eid in ordered[:capacity]:
                layers = [li for li, gi in layer_map.items() if gi == group]
                out.extend((int(li), int(group), int(eid)) for li in layers)
            return out
        scores = torch.zeros(num_experts, dtype=torch.float32)
        if all_future:
            if probs.shape[2] <= 1:
                return []
            future = probs[:, -1, 1:, :]  # [B, K, E]
            K = future.shape[1]
            for offset in range(K):
                step_probs = future[:, offset, :]
                weights, ids = self._router_topk(step_probs)
                weight_scale = 1.0 / float(offset + 1)
                scores.index_add_(
                    0,
                    ids.reshape(-1),
                    weights.reshape(-1).to(scores.dtype) * weight_scale,
                )
        else:
            if probs.shape[2] <= future_offset:
                return []
            step_probs = probs[:, -1, future_offset, :]  # [B, E]
            weights, ids = self._router_topk(step_probs)
            scores.index_add_(0, ids.reshape(-1), weights.reshape(-1).to(scores.dtype))
        if capacity is None:
            capacity = num_experts
        return self._ordered_ids_from_scores(scores, min(int(capacity), num_experts))

    def _route_cache_stats(
        self,
        masks: Optional[List[torch.Tensor]],
        *,
        include_all_tokens: bool,
    ) -> Dict[str, object]:
        """Measure real per-layer top-k cache coverage for the last forward."""
        moe_layers = self._moe_layers()
        totals = {
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_weight_mass": 0.0,
            "fallback_weight_mass": 0.0,
            "total_route_weight_mass": 0.0,
            "activated_expert_ids": [],
            "activated_expert_keys": [],
            "per_layer": [],
        }
        if not moe_layers:
            return totals

        activated: set[int] = set()
        activated_keys: set[tuple[int, int, int]] = set()
        store = self.cache_manager.expert_store
        for layer_idx, moe in enumerate(moe_layers):
            # Every loop iteration is a real dispatch: count all of them.
            probs = self._virtual_router_probs(moe, last_token=not include_all_tokens)
            if probs is None:
                continue
            flat_probs = probs.reshape(-1, probs.shape[-1])
            weights, ids = self._router_topk(flat_probs)
            flat_ids = ids.reshape(-1).to(torch.long)
            flat_weights = weights.reshape(-1).to(torch.float32)
            if flat_ids.numel() == 0:
                continue

            if masks is None:
                available = torch.ones(probs.shape[-1], dtype=torch.bool)
            elif len(masks) == 0:
                available = torch.zeros(probs.shape[-1], dtype=torch.bool)
            else:
                raw_mask = masks[min(layer_idx, len(masks) - 1)].detach().cpu()
                available = raw_mask.to(dtype=torch.bool)
            if available.numel() < probs.shape[-1]:
                padded = torch.zeros(probs.shape[-1], dtype=torch.bool)
                padded[:available.numel()] = available
                available = padded

            hit_mask = available[flat_ids]
            hits = int(hit_mask.sum().item())
            total = int(flat_ids.numel())
            misses = total - hits
            hit_mass = float(flat_weights[hit_mask].sum().item())
            miss_mass = float(flat_weights[~hit_mask].sum().item())
            total_mass = hit_mass + miss_mass

            activated.update(int(eid) for eid in flat_ids.unique().tolist())
            group_idx = int(getattr(store, "_layer_to_group", {}).get(layer_idx, -1))
            activated_keys.update(
                (int(layer_idx), group_idx, int(eid))
                for eid in flat_ids.unique().tolist()
            )
            totals["cache_hits"] += hits
            totals["cache_misses"] += misses
            totals["hit_weight_mass"] += hit_mass
            totals["fallback_weight_mass"] += miss_mass
            totals["total_route_weight_mass"] += total_mass
            totals["per_layer"].append({
                "layer": layer_idx,
                "hit_rate": round(hits / max(total, 1), 4),
                "fallback_weight_rate": round(miss_mass / max(total_mass, 1e-12), 4),
                "hits": hits,
                "misses": misses,
                "activated_experts": sorted(int(eid) for eid in flat_ids.unique().tolist()),
                "loaded_experts": sorted(int(eid) for eid in available.nonzero().flatten().tolist()),
            })

        totals["activated_expert_ids"] = sorted(activated)
        totals["activated_expert_keys"] = sorted(activated_keys)
        return totals

    def _plan_prefill_experts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modality_inputs: Optional[dict] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> tuple[List[int], List[int]]:
        """Probe prompt routing with shared fallback, then load hot experts.

        The live model contains placeholder experts after lazy setup. A cold
        first prefill would therefore run entirely through shared fallback and
        can diverge from the real model before decode starts. This probe pass
        does not materialize expert modules; it only records gate choices, then
        returns the most frequent expert IDs bounded by the active cache budget.
        The caller reruns the real prefill after those experts are promoted.
        """
        if not getattr(self.config, "use_moe", True):
            return [], []
        num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        if num_experts <= 0:
            return [], []

        cold_mask = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
        masks = [cold_mask] * len(self.cache_manager.expert_store.moe_layers)
        saved_hooks = [
            (
                moe.runtime_miss_policy,
                moe.runtime_on_demand_loader,
                moe.runtime_touch_expert,
            )
            for moe in self._moe_layers()
        ]
        try:
            for moe in self._moe_layers():
                moe.runtime_miss_policy = "fallback_diagnostic"
                moe.runtime_on_demand_loader = None
                moe.runtime_touch_expert = None
            probe_start = {
                "on_demand_loads": int(self._runtime_stats.get("on_demand_loads", 0)),
                "on_demand_load_time_s": float(self._runtime_stats.get("on_demand_load_time_s", 0.0)),
                "sync_ssd_loads": int(self._runtime_stats.get("sync_ssd_loads", 0)),
                "resident_vram_hits": int(self._runtime_stats.get("resident_vram_hits", 0)),
                "resident_ram_hits": int(self._runtime_stats.get("resident_ram_hits", 0)),
                "loaded_len": len(self._runtime_stats.get("on_demand_loaded_experts", [])),
                "promoted_len": len(self._runtime_stats.get("on_demand_promoted_experts", [])),
            }
            self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                available_expert_masks=masks,
                modality_inputs=modality_inputs,
                modality_mask=modality_mask,
            )
            self._runtime_stats["on_demand_loads"] = probe_start["on_demand_loads"]
            self._runtime_stats["on_demand_load_time_s"] = probe_start["on_demand_load_time_s"]
            self._runtime_stats["sync_ssd_loads"] = probe_start["sync_ssd_loads"]
            self._runtime_stats["resident_vram_hits"] = probe_start["resident_vram_hits"]
            self._runtime_stats["resident_ram_hits"] = probe_start["resident_ram_hits"]
            self._runtime_stats["on_demand_loaded_experts"] = self._runtime_stats["on_demand_loaded_experts"][:probe_start["loaded_len"]]
            self._runtime_stats["on_demand_promoted_experts"] = self._runtime_stats["on_demand_promoted_experts"][:probe_start["promoted_len"]]
        finally:
            for moe, saved in zip(self._moe_layers(), saved_hooks):
                moe.runtime_miss_policy, moe.runtime_on_demand_loader, moe.runtime_touch_expert = saved

        scores = torch.zeros(num_experts, dtype=torch.float32)
        for layer in self.model.model.layers:
            moe = getattr(layer, "mlp", None)
            if not isinstance(moe, ChronosMOEFeedForward):
                continue
            probs = self._virtual_router_probs(moe)
            if probs is None:
                continue
            weights, expert_ids = self._router_topk(probs.reshape(-1, num_experts))
            scores.index_add_(
                0,
                expert_ids.reshape(-1).to(torch.long),
                weights.reshape(-1).to(scores.dtype),
            )

        if not torch.any(scores > 0):
            return [], []
        vram_capacity = max(1, min(
            int(getattr(self.cache_manager.expert_store, "vram_capacity", 1) or 1),
            num_experts,
        ))
        ram_capacity = max(vram_capacity, min(
            int(getattr(self.cache_manager.expert_store, "ram_capacity", vram_capacity) or vram_capacity),
            num_experts,
        ))
        ram_ids = self._ordered_ids_from_scores(scores, ram_capacity)
        return ram_ids[:vram_capacity], ram_ids

    # no_grad, not inference_mode: experts paged in here are reused by
    # generate(); inference tensors would poison later autograd-tracked calls.
    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        draft_depth: int = 1,
        gamma: int = 4,
        gamma_max: int = 8,
        adaptive_gamma: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
        eos_token_id=None,
        miss_policy: str = "on_demand",
        generator: Optional[torch.Generator] = None,
        on_token=None,
    ) -> torch.Tensor:
        """Recurrent-depth self-speculative decoding with lazy experts.

        Batch size 1.  Each draft and verify forward runs with the current
        expert availability masks; before it, the lookahead router's
        prediction from the previous forward is prefetched to RAM and the
        promotable part moved to VRAM, so drafts warm the experts their own
        verify pass needs.  Greedy output equals :meth:`generate` greedy
        output token for token.  ``per_iteration`` / non-recurrent models
        decode plainly (``draft_depth`` = full depth).
        """
        from chronos.runtime.speculative import speculative_generate, speculative_supported
        input_ids = input_ids.to(self.device)
        miss_policy = self._normalize_miss_policy(miss_policy)
        full_depth = int(getattr(self.config, "max_inference_depth", None) or self.config.recurrent_depth)
        use_spec = speculative_supported(self.config) and int(draft_depth) < full_depth
        self._route_cache_scores = {}
        self._runtime_stats = {
            "on_demand_loads": 0, "on_demand_load_time_s": 0.0, "sync_ssd_loads": 0,
            "async_cold_miss_prefetches": 0, "resident_vram_hits": 0, "resident_ram_hits": 0,
            "on_demand_loaded_experts": [], "on_demand_promoted_experts": [],
        }
        cm = self.cache_manager
        store = cm.expert_store
        state = {"lookahead": None, "hits": 0, "misses": 0, "forwards": 0, "fallback_mass": 0.0,
                 "on_demand": 0, "prefill_s": None}
        t0 = time.monotonic()
        rss0 = _rss_gb()
        backend0 = _backend_memory_snapshot(self.device)
        self._install_moe_runtime_hooks(miss_policy)

        def forward_fn(ids, cache, depth):
            if state["lookahead"] is not None:
                predicted = self._predict_expert_ids(state["lookahead"], all_future=True,
                                                     capacity=int(store.ram_capacity))
                cm.prefetch_experts_to_ram(predicted)
                ready = self._promotion_ready_ids(predicted[: int(store.vram_capacity)])
                if ready:
                    cm.ensure_resident(ready)
            masks = cm.availability_masks_all_layers()
            out = self.model(ids.to(self.device), past_key_values=cache, use_cache=True,
                             recurrent_depth=depth, available_expert_masks=masks)
            state["lookahead"] = out.lookahead_probs
            route = self._route_cache_stats(masks, include_all_tokens=cache is None)
            state["hits"] += int(route["cache_hits"])
            state["misses"] += int(route["cache_misses"])
            rt = self._consume_moe_runtime_stats()
            state["fallback_mass"] += float(rt["fallback_executed_weight_mass"])
            state["on_demand"] += len(rt["on_demand_experts"])
            state["forwards"] += 1
            if state["prefill_s"] is None:
                state["prefill_s"] = time.monotonic() - t0
            return out

        try:
            prefill_ids, prefill_ram_ids = self._plan_prefill_experts(input_ids, None, None, None)
            if prefill_ram_ids:
                store.prefetch_to_ram(prefill_ram_ids)
            if prefill_ids:
                cm.ensure_resident(prefill_ids)
            out, spec = speculative_generate(
                self.model, input_ids, max_new_tokens=int(max_new_tokens),
                draft_depth=int(draft_depth) if use_spec else full_depth, gamma=int(gamma),
                gamma_max=int(gamma_max), adaptive_gamma=bool(adaptive_gamma), do_sample=bool(do_sample),
                temperature=max(float(temperature), 1e-6), top_k=int(top_k or 0), top_p=float(top_p or 1.0),
                eos_token_id=eos_token_id, generator=generator, on_token=on_token, forward_fn=forward_fn)
        finally:
            self._clear_moe_runtime_hooks()
        elapsed = time.monotonic() - t0
        generated = int(out.shape[1] - input_ids.shape[1])
        total = state["hits"] + state["misses"]
        runtime_stats = dict(self._runtime_stats)
        prefill_s = float(state["prefill_s"] or 0.0)
        self.last_stats = {
            "tokens_generated": generated,
            "elapsed_s": round(elapsed, 3),
            "tokens_per_sec": round(generated / max(elapsed, 1e-6), 2),
            "prefill_time_s": round(prefill_s, 3),
            "decode_time_s": round(max(0.0, elapsed - prefill_s), 3),
            "rss_before_prefill_gb": round(rss0, 3),
            "rss_after_decode_gb": round(_rss_gb(), 3),
            "decode_rss_delta_gb": round(_rss_gb() - rss0, 3),
            **_backend_memory_fields(backend0, "before_prefill"),
            **_backend_memory_fields(_backend_memory_snapshot(self.device), "after_decode"),
            "cache_hits": state["hits"], "cache_misses": state["misses"],
            "cache_hit_rate": round(state["hits"] / max(total, 1), 4),
            "resident_hit_rate": round(state["hits"] / max(total, 1), 4),
            "fallback_weight_mass": round(state["fallback_mass"], 4),
            "on_demand_loads": int(runtime_stats.get("on_demand_loads", 0)),
            "sync_ssd_loads": int(runtime_stats.get("sync_ssd_loads", 0)),
            "async_cold_miss_prefetches": int(runtime_stats.get("async_cold_miss_prefetches", 0)),
            "resident_ram_hits": int(runtime_stats.get("resident_ram_hits", 0)),
            "miss_policy": miss_policy,
            "quality_safe": miss_policy == "sync_on_demand",
            "forwards": state["forwards"],
            "prefill_expert_ids": prefill_ids,
            "prefill_ram_expert_ids": prefill_ram_ids,
            "speculative": spec.as_dict() | {"enabled": use_spec, "full_depth": full_depth,
                                             "draft_depth": int(draft_depth) if use_spec else full_depth,
                                             "gamma_final": spec.gamma_history[-1] if spec.gamma_history else int(gamma)},
        }
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.85,
        top_p: float = 0.85,
        top_k: int = 50,
        eos_token_id: int = 2,
        streamer=None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        scheduler: Optional[PrefillScheduler] = None,
        miss_policy: str = "on_demand",
        modality_inputs: Optional[dict] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        miss_policy = self._normalize_miss_policy(miss_policy)
        self._route_cache_scores = {}
        self._last_prefill_ram_expert_ids = []
        self._runtime_stats = {
            "on_demand_loads": 0,
            "on_demand_load_time_s": 0.0,
            "sync_ssd_loads": 0,
            "async_cold_miss_prefetches": 0,
            "resident_vram_hits": 0,
            "resident_ram_hits": 0,
            "on_demand_loaded_experts": [],
            "on_demand_promoted_experts": [],
        }
        self._install_moe_runtime_hooks(miss_policy)

        try:
            # ── Prefill phase: front-load expert IO before generation ──────
            prefill_t0 = time.monotonic()
            prefill_mem0 = _rss_gb()
            _io0 = dict(getattr(self.cache_manager.expert_store, "_io_stats", {}) or {})
            prefill_backend_mem0 = _backend_memory_snapshot(self.device)
            if scheduler is not None:
                scheduler.prepare(input_ids, device=str(self.device))
                scheduler.wait()  # block until SSD→VRAM load complete
                prefill_expert_ids: List[int] = []
                prefill_ram_expert_ids: List[int] = []
            else:
                prefill_expert_ids, prefill_ram_expert_ids = self._plan_prefill_experts(
                    input_ids, attention_mask, modality_inputs, modality_mask
                )
                self._last_prefill_ram_expert_ids = prefill_ram_expert_ids
                if prefill_ram_expert_ids:
                    self.cache_manager.expert_store.prefetch_to_ram(prefill_ram_expert_ids)
                if prefill_expert_ids:
                    self.cache_manager.ensure_resident(prefill_expert_ids)

            past_key_values = None
            finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
            if streamer:
                streamer.put(input_ids.cpu())

            tokens_generated = 0
            cache_hits, cache_misses = 0, 0
            activated_expert_ids: List[int] = []
            submitted_prefetch_ids: List[int] = []
            promoted_expert_ids: List[int] = []
            predicted_current_expert_ids: List[int] = []
            on_demand_expert_ids: List[int] = []
            fallback_expert_ids: List[int] = []
            route_layer_stats: List[dict] = []
            hit_weight_mass = 0.0
            fallback_weight_mass = 0.0
            total_route_weight_mass = 0.0
            on_demand_weight_mass = 0.0
            fallback_executed_weight_mass = 0.0
            prediction_hits = 0
            prediction_total = 0
            cluster_prediction_hits = 0
            cluster_prediction_total = 0
            # M3: holds last step's lookahead predictions; we submit them as
            # prefetch BEFORE this step's forward so the H2D stream overlaps
            # with compute. None on the first iteration (no predictions yet).
            prev_lookahead_probs = None
            prev_current_ids: List[int] = []
            prefill_time_s = 0.0
            prefill_mem1 = prefill_mem0
            prefill_backend_mem1 = prefill_backend_mem0
            prefill_done_t = prefill_t0

            for step in range(max_new_tokens):
            # ── M3 pipeline: fire async prefetch BEFORE forward ──
                predicted_ids: List[int] = []
                ram_predicted_ids: List[int] = []
                if scheduler is None:
                    if prev_lookahead_probs is not None:
                        store = self.cache_manager.expert_store
                        vram_capacity = int(getattr(store, "vram_capacity", 1) or 1)
                        ram_capacity = int(getattr(store, "ram_capacity", vram_capacity) or vram_capacity)
                        ram_predicted_ids = self._predict_expert_ids(
                            prev_lookahead_probs,
                            capacity=ram_capacity,
                            all_future=True,
                        )
                        predicted_ids = self._predict_expert_ids(
                            prev_lookahead_probs,
                            future_offset=1,
                            capacity=vram_capacity,
                        )
                        predicted_current_expert_ids.extend(predicted_ids)
                        # Predicted pages must survive this step's evictions.
                        protect = getattr(store, "set_protected_experts", None)
                        if protect is not None:
                            protect(list(dict.fromkeys(predicted_ids + prev_current_ids)))
                        submitted_prefetch_ids.extend(
                            self.cache_manager.prefetch_experts_to_ram(ram_predicted_ids)
                        )
                        submitted_prefetch_ids.extend(
                            self.cache_manager.prefetch_for_next_step(prev_lookahead_probs)
                        )
                        sticky_ids = self._ordered_sticky_ids(vram_capacity)
                        promote_candidates = list(dict.fromkeys(
                            predicted_ids + sticky_ids + prev_current_ids
                        ))
                        ready_promote = self._promotion_ready_ids(promote_candidates)
                        if ready_promote:
                            promoted_expert_ids.extend(
                                self.cache_manager.ensure_resident(ready_promote[:vram_capacity])
                            )
                    elif prev_current_ids:
                        ready_promote = self._promotion_ready_ids(prev_current_ids)
                        if ready_promote:
                            promoted_expert_ids.extend(
                                self.cache_manager.ensure_resident(ready_promote)
                            )

            # Get availability masks AFTER ensure_resident so the mask
            # reflects experts that just became available.
                if scheduler is not None:
                    masks = scheduler.avail_masks(device=str(self.device))
                else:
                    masks = self.cache_manager.availability_masks_all_layers()

                from chronos.model.model_chronos import cache_sequence_length
                past_len = cache_sequence_length(past_key_values) if past_key_values else 0
                result = self.model(
                    input_ids[:, past_len:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    available_expert_masks=masks,
                    modality_inputs=modality_inputs,
                    modality_mask=modality_mask,
                )
                outputs, lookahead_probs = result.legacy_tuple()

                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
                    )

                route_stats = self._route_cache_stats(
                    masks,
                    include_all_tokens=(past_key_values is None),
                )
                cache_hits += int(route_stats["cache_hits"])
                cache_misses += int(route_stats["cache_misses"])
                hit_weight_mass += float(route_stats["hit_weight_mass"])
                fallback_weight_mass += float(route_stats["fallback_weight_mass"])
                total_route_weight_mass += float(route_stats["total_route_weight_mass"])
                route_layer_stats = route_stats["per_layer"]
                current_ids = (
                    route_stats["activated_expert_keys"]
                    if bool(getattr(self.config, "layer_aware_cache", False))
                    else route_stats["activated_expert_ids"]
                )
                activated_expert_ids.extend(route_stats["activated_expert_ids"])
                rt = self._consume_moe_runtime_stats()
                on_demand_expert_ids.extend(rt["on_demand_experts"])
                fallback_expert_ids.extend(rt["fallback_experts"])
                on_demand_weight_mass += float(rt["on_demand_weight_mass"])
                fallback_executed_weight_mass += float(rt["fallback_executed_weight_mass"])
                if predicted_ids:
                    current_set = set(current_ids)
                    prediction_total += len(current_set)
                    prediction_hits += len(current_set.intersection(predicted_ids))
                    cluster_storage = getattr(
                        self.cache_manager.expert_store, "_cluster_storage", None
                    )
                    if cluster_storage is not None:
                        manifest = cluster_storage.manifest
                        def _eid(key):
                            return int(key[2]) if isinstance(key, tuple) else int(key)
                        predicted_clusters = {
                            manifest.expert_to_cluster[_eid(key)]
                            for key in predicted_ids
                            if _eid(key) in manifest.expert_to_cluster
                        }
                        current_clusters = [
                            manifest.expert_to_cluster[_eid(key)]
                            for key in current_set
                            if _eid(key) in manifest.expert_to_cluster
                        ]
                        cluster_prediction_total += len(current_clusters)
                        cluster_prediction_hits += sum(
                            int(cid in predicted_clusters) for cid in current_clusters
                        )
                if scheduler is None:
                    self.cache_manager.expert_store.reprioritize_resident_experts(
                        self._recent_route_scores(current_ids, predicted_ids, ram_predicted_ids)
                    )

                # Stash predictions for the NEXT iteration's prefetch.
                prev_lookahead_probs = lookahead_probs
                prev_current_ids = current_ids

                # Sampling
                logits = outputs.logits[:, -1, :] / temperature
                if repetition_penalty != 1.0:
                    for i in range(input_ids.shape[0]):
                        logits[i, torch.unique(input_ids[i])] /= repetition_penalty
                if top_k > 0:
                    k = min(int(top_k), int(logits.shape[-1]))
                    logits[logits < torch.topk(logits, k)[0][..., -1, None]] = -float('inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    mask_p = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                    mask_p[..., 1:] = mask_p[..., :-1].clone()
                    mask_p[..., 0] = 0
                    logits[mask_p.scatter(1, sorted_indices, mask_p)] = -float('inf')

                next_token = (
                    torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                    if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
                )
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        next_token.new_full((next_token.shape[0], 1), eos_token_id),
                        next_token,
                    )

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                past_key_values = outputs.past_key_values
                tokens_generated += 1

                if streamer:
                    streamer.put(next_token.cpu())

                if eos_token_id is not None:
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                    if finished.all():
                        if step == 0:
                            prefill_done_t = time.monotonic()
                            prefill_mem1 = _rss_gb()
                            prefill_backend_mem1 = _backend_memory_snapshot(self.device)
                            prefill_time_s = prefill_done_t - prefill_t0
                        break

                if step == 0:
                    prefill_done_t = time.monotonic()
                    prefill_mem1 = _rss_gb()
                    prefill_backend_mem1 = _backend_memory_snapshot(self.device)
                    prefill_time_s = prefill_done_t - prefill_t0

            # ── Post-generation: update frequency heuristic ───────────────
            if scheduler is not None:
                scheduler.record_activation(list(set(activated_expert_ids)))

            end_t = time.monotonic()
            end_mem = _rss_gb()
            end_backend_mem = _backend_memory_snapshot(self.device)
            if tokens_generated == 0:
                prefill_done_t = end_t
                prefill_mem1 = end_mem
                prefill_backend_mem1 = end_backend_mem
                prefill_time_s = end_t - prefill_t0
            elapsed = end_t - prefill_t0
            decode_time_s = max(0.0, end_t - prefill_done_t)
            total_cache = cache_hits + cache_misses
            resident_hit_rate = cache_hits / max(total_cache, 1)
            fallback_weight_rate = fallback_weight_mass / max(total_route_weight_mass, 1e-12)
            executed_fallback_rate = fallback_executed_weight_mass / max(total_route_weight_mass, 1e-12)
            prediction_hit_rate = prediction_hits / max(prediction_total, 1)
            cluster_recall = cluster_prediction_hits / max(cluster_prediction_total, 1)
            runtime_stats = dict(self._runtime_stats)
            promoted_expert_ids.extend(runtime_stats.get("on_demand_promoted_experts", []))
            backend_fields = {}
            backend_fields.update(_backend_memory_fields(prefill_backend_mem0, "before_prefill"))
            backend_fields.update(_backend_memory_fields(prefill_backend_mem1, "after_prefill"))
            backend_fields.update(_backend_memory_fields(end_backend_mem, "after_decode"))
            recurrent = getattr(self.model.model, "last_recurrent_state", None) or {}
            depth_used = recurrent.get("depth_used")
            average_ponder_steps = (
                float(depth_used.float().mean().item()) if depth_used is not None else 1.0
            )
            early_exit = recurrent.get("early_exit_ratio")
            early_exit_ratio = (
                float(early_exit.item()) if isinstance(early_exit, torch.Tensor)
                else float(early_exit or 0.0)
            )
            contraction = recurrent.get("contraction")
            contraction_ratio = (
                float(contraction.item()) if isinstance(contraction, torch.Tensor)
                else float(contraction or 0.0)
            )
            on_demand_loads = int(runtime_stats.get("on_demand_loads", 0))
            resident_ram_hits = int(runtime_stats.get("resident_ram_hits", 0))
            cold_misses = int(runtime_stats.get("async_cold_miss_prefetches", 0)) + int(
                runtime_stats.get("sync_ssd_loads", 0)
            )
            self.last_stats = {
                "tokens_generated": tokens_generated,
                "elapsed_s": round(elapsed, 3),
                "tokens_per_sec": round(tokens_generated / max(elapsed, 1e-6), 2),
                "prefill_time_s": round(prefill_time_s, 3),
                "decode_time_s": round(decode_time_s, 3),
                "prefill_rss_delta_gb": round(prefill_mem1 - prefill_mem0, 3),
                "decode_rss_delta_gb": round(end_mem - prefill_mem1, 3),
                "rss_before_prefill_gb": round(prefill_mem0, 3),
                "rss_after_prefill_gb": round(prefill_mem1, 3),
                "rss_after_decode_gb": round(end_mem, 3),
                **backend_fields,
                "cache_hit_rate": round(resident_hit_rate, 4),
                "resident_hit_rate": round(resident_hit_rate, 4),
                "resident_vram_hit_rate": round(resident_hit_rate, 4),
                "ram_promotion_hit_rate": round(resident_ram_hits / max(on_demand_loads, 1), 4),
                "ssd_cold_miss_rate": round(cold_misses / max(on_demand_loads, 1), 4),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "fallback_weight_rate": round(executed_fallback_rate, 4),
                "resident_miss_weight_rate": round(fallback_weight_rate, 4),
                "fallback_weight_mass": round(fallback_executed_weight_mass, 4),
                "resident_miss_weight_mass": round(fallback_weight_mass, 4),
                "hit_weight_mass": round(hit_weight_mass, 4),
                "total_route_weight_mass": round(total_route_weight_mass, 4),
                "on_demand_weight_mass": round(on_demand_weight_mass, 4),
                "on_demand_weight_rate": round(on_demand_weight_mass / max(total_route_weight_mass, 1e-12), 4),
                "on_demand_loads": int(runtime_stats.get("on_demand_loads", 0)),
                "on_demand_load_time_s": round(float(runtime_stats.get("on_demand_load_time_s", 0.0)), 4),
                "sync_ssd_loads": int(runtime_stats.get("sync_ssd_loads", 0)),
                "async_cold_miss_prefetches": int(runtime_stats.get("async_cold_miss_prefetches", 0)),
                "resident_vram_hits": int(runtime_stats.get("resident_vram_hits", 0)),
                "resident_ram_hits": int(runtime_stats.get("resident_ram_hits", 0)),
                "quality_safe": miss_policy == "sync_on_demand",
                "miss_policy": miss_policy,
                "prediction_hit_rate": round(prediction_hit_rate, 4),
                "prediction_topk_recall": round(prediction_hit_rate, 4),
                "cluster_recall": round(cluster_recall, 4),
                "prediction_hits": prediction_hits,
                "prediction_total": prediction_total,
                "cluster_prediction_hits": cluster_prediction_hits,
                "cluster_prediction_total": cluster_prediction_total,
                "average_ponder_steps": round(average_ponder_steps, 4),
                "early_exit_ratio": round(early_exit_ratio, 4),
                "recurrent_contraction": round(contraction_ratio, 6),
                "route_layer_stats": route_layer_stats,
                "prefill_expert_ids": prefill_expert_ids,
                "prefill_ram_expert_ids": prefill_ram_expert_ids,
                "activated_expert_ids": sorted(set(activated_expert_ids)),
                "predicted_current_expert_ids": sorted(set(predicted_current_expert_ids), key=repr),
                "prefetch_submitted_ids": sorted(set(submitted_prefetch_ids), key=repr),
                "promoted_expert_ids": sorted(set(promoted_expert_ids), key=repr),
                "on_demand_expert_ids": sorted(set(on_demand_expert_ids), key=repr),
                "fallback_expert_ids": sorted(set(fallback_expert_ids), key=repr),
                "prefetch_submitted_total": len(submitted_prefetch_ids),
                "promoted_total": len(promoted_expert_ids),
                **self.cache_manager.stats(),
            }
            store = self.cache_manager.expert_store
            pages_read = int(getattr(store, "_io_stats", {}).get("ssd_page_reads", 0)) - int(_io0.get("ssd_page_reads", 0))
            page_bytes = int(getattr(store, "_expert_bytes", 0) or 0)
            ssd_mb = pages_read * page_bytes / 2 ** 20
            self.last_stats.update({
                "ssd_pages_read": pages_read,
                "ssd_read_mb": round(ssd_mb, 3),
                "ssd_read_mb_per_s": round(ssd_mb / max(elapsed, 1e-6), 3),
                "resident_experts": int(self.last_stats.get("vram_experts", 0)),
            })
            if hasattr(store, "set_protected_experts"):
                store.set_protected_experts(())
            # Publish a compact rolling series for the IO Monitor without
            # retaining per-request payloads in the hot path.
            for metric_name, metric_value in (
                ("tokens_per_sec", self.last_stats["tokens_per_sec"]),
                ("resident_vram_hit_rate", self.last_stats["resident_vram_hit_rate"]),
                ("ram_promotion_hit_rate", self.last_stats["ram_promotion_hit_rate"]),
                ("ssd_cold_miss_rate", self.last_stats["ssd_cold_miss_rate"]),
                ("prediction_topk_recall", self.last_stats["prediction_topk_recall"]),
                ("cluster_recall", self.last_stats["cluster_recall"]),
                ("average_ponder_steps", self.last_stats["average_ponder_steps"]),
                ("early_exit_ratio", self.last_stats["early_exit_ratio"]),
                ("recurrent_contraction", self.last_stats["recurrent_contraction"]),
                ("fallback_weight_mass", self.last_stats["fallback_weight_mass"]),
            ):
                _metric(metric_name, metric_value)

            if streamer:
                streamer.end()
            return input_ids
        finally:
            self._clear_moe_runtime_hooks()
