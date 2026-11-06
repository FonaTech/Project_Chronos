"""Crash-safe SSD paging primitives for expert training and optimizer state.

This module intentionally has no dependency on a particular accelerator.  It
provides atomic tensor pages, bounded in-memory LRU, and an async prefetch
interface that PyTorch/MPS/CUDA trainers can share.
"""
from __future__ import annotations

import collections
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


class PagedTrainingController:
    """Route-aware expert residency controller for training.

    The controller deliberately keeps scheduling separate from the optimizer:
    optimizer moments are persisted by :class:`PagedOptimizerState`, while
    this object selects a bounded working set for the *next* microbatch from
    the previous batch's router probabilities.  The first batch is all
    resident (stable bootstrap); subsequent batches pass per-layer masks to
    Chronos' shared-fallback MoE path.  This gives a real bounded activation
    and expert-compute path without cloning a full ``state_dict`` on CPU.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.enabled = bool(getattr(config, "training_paging", False))
        self.num_experts = int(getattr(config, "num_experts", 0) or 0)
        self.num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        budget = getattr(config, "offload_working_set_experts", None)
        if budget is None:
            budget = getattr(config, "recommended_resident_experts", None)
        if budget is None:
            budget = max(1, self.num_experts // 2) if self.num_experts else 0
        self.active_budget = max(1, min(int(budget), self.num_experts)) if self.num_experts else 0
        self._masks = None
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        self.weight_store = None
        if self.enabled:
            page_root = os.path.join(
                str(getattr(config, "optimizer_page_dir", "./optimizer_pages")),
                "expert_weights",
            )
            page_mb = max(1, int(getattr(config, "page_size_mb", 64) or 64))
            # Weight pages keep the parameter's own (compute) dtype; the FP32
            # truth lives in the optimizer masters (precision_training), so an
            # FP32 copy here would only double SSD traffic.
            self.weight_store = TieredTensorStore(
                page_root, capacity=max(1, min(64, page_mb // 4)), dtype=None
            )
        # Checkpoint helpers can materialize paged weights without importing a
        # specific trainer class.
        setattr(model, "_chronos_paged_training_controller", self)
        self._step = 0
        self._stats = {"paged_steps": 0, "active_experts": 0,
                       "route_tokens": 0, "cold_route_tokens": 0,
                       "fallback_weight_mass": 0.0}

    def masks_for_forward(self, device=None, *, allow_evict: bool = True):
        if not self.enabled or self.num_experts <= 0:
            return None
        if self._masks is None:
            # Bootstrap with a full mask so router probabilities are learned
            # before any expert is made cold.
            mask = torch.ones(self.num_experts, dtype=torch.bool, device=device)
            masks = [mask.clone() for _ in range(self.num_layers)]
        else:
            masks = [mask.to(device=device) if device is not None else mask
                     for mask in self._masks]
        self._apply_residency(masks, allow_evict=allow_evict)
        return masks

    @torch.no_grad()
    def _apply_residency(self, masks, *, allow_evict: bool = True):
        """Move cold expert parameter pages off accelerator memory.

        Expert modules stay registered (optimizer references remain valid),
        but their ``Parameter.data`` lives on CPU while the route mask ensures
        they are not executed.  Active pages are brought back to the training
        device just before forward.  CPU runs are naturally a no-op.
        """
        if not self.enabled or self.device.type == "cpu":
            return
        backbone = getattr(self.model, "model", self.model)
        for layer_idx, layer in enumerate(getattr(backbone, "layers", [])):
            moe = getattr(layer, "mlp", None)
            experts = getattr(moe, "experts", None)
            if experts is None or layer_idx >= len(masks):
                continue
            mask = masks[layer_idx].detach().to("cpu", dtype=torch.bool)
            for eid, expert in enumerate(experts):
                active = eid < mask.numel() and bool(mask[eid])
                # Do not evict a page whose accumulated gradient has not yet
                # been consumed by optimizer.step().  This preserves gradient
                # accumulation semantics while still shrinking residency at
                # update boundaries.
                has_pending_grad = any(p.grad is not None for p in expert.parameters())
                target = (
                    self.device
                    if active or has_pending_grad or not allow_evict
                    else torch.device("cpu")
                )
                params = dict(expert.named_parameters())
                if not params:
                    continue
                group = self._layer_group(layer_idx)
                key = PageKey(layer_idx, group, eid, "weight")
                materialized = all(param.numel() > 0 for param in params.values())
                if target != torch.device("cpu") and not materialized:
                    if self.weight_store is None:
                        raise RuntimeError("training weight paging store is not initialized")
                    state = self.weight_store.get(key, device=target)
                    if state is None:
                        raise RuntimeError(f"missing paged expert weight: {key.filename()}")
                    for name, param in params.items():
                        if name not in state:
                            raise RuntimeError(f"missing tensor {name} in {key.filename()}")
                        param.data = state[name].to(device=target, dtype=param.dtype)
                elif target == torch.device("cpu") and allow_evict and materialized:
                    # Persist updated weights before releasing accelerator and
                    # host storage.  Empty registered Parameters keep optimizer
                    # identity stable while consuming no model-sized memory.
                    if self.weight_store is None:
                        raise RuntimeError("training weight paging store is not initialized")
                    self.weight_store.put(key, {name: p.data for name, p in params.items()})
                    for param in params.values():
                        param.data = torch.empty(0, device="cpu", dtype=param.dtype)

    def _layer_group(self, layer_idx: int) -> int:
        prelude = int(getattr(self.config, "prelude_layers", 0) or 0)
        groups = int(getattr(self.config, "recurrent_groups", 0) or 0)
        return layer_idx - prelude if prelude <= layer_idx < prelude + groups else -1

    @torch.no_grad()
    def materialize_all(self) -> None:
        """Restore every SSD expert page for full checkpoints/evaluation."""
        if not self.enabled:
            return
        masks = [torch.ones(self.num_experts, dtype=torch.bool, device=self.device)
                 for _ in range(self.num_layers)]
        self._apply_residency(masks, allow_evict=False)

    @torch.no_grad()
    def observe(self):
        if not self.enabled or self.num_experts <= 0:
            return
        layers = []
        active_total = 0
        cold_mass = 0.0
        for layer_idx, layer in enumerate(getattr(getattr(self.model, "model", self.model), "layers", [])):
            moe = getattr(layer, "mlp", None)
            probs = getattr(moe, "last_router_probs", None)
            if probs is None:
                layers.append(torch.ones(self.num_experts, dtype=torch.bool))
                continue
            scores = probs.float().reshape(-1, self.num_experts).mean(dim=0)
            k = min(self.active_budget, self.num_experts)
            ids = torch.topk(scores, k=k, dim=-1).indices
            mask = torch.zeros(self.num_experts, dtype=torch.bool)
            mask[ids.cpu()] = True
            layers.append(mask)
            active_total += int(mask.sum())
            self._stats["route_tokens"] += int(probs.shape[0] * probs.shape[1])
            if self._masks is not None and layer_idx < len(self._masks):
                previous = self._masks[layer_idx]
                top_ids = torch.topk(probs.float().reshape(-1, self.num_experts),
                                     k=min(int(getattr(self.config, "num_experts_per_tok", 1)), self.num_experts),
                                     dim=-1).indices
                cold_mass += float((~previous[top_ids]).float().mean().item())
        if layers:
            self._masks = layers
            self._stats["active_experts"] = active_total
            self._stats["paged_steps"] += 1
            self._step += 1
            self._stats["cold_route_tokens"] = int(cold_mass * max(self._stats["route_tokens"], 1))
            self._stats["fallback_weight_mass"] = cold_mass
            try:
                from chronos.runtime.metrics import safe_record
                safe_record("training_active_experts", active_total)
                safe_record("training_cold_route_fraction", cold_mass / max(len(layers), 1))
            except Exception:
                pass

    def stats(self) -> dict:
        return {**self._stats, "enabled": self.enabled,
                "active_budget": self.active_budget,
                "weight_store": self.weight_store.summary() if self.weight_store else None}


@dataclass(frozen=True)
class PageKey:
    layer: int
    group: int
    expert: int
    kind: str = "weight"

    def filename(self) -> str:
        return f"{self.kind}_l{self.layer}_g{self.group}_e{self.expert}.pt"


class TieredTensorStore:
    """Bounded CPU page cache with atomic SSD persistence."""

    def __init__(self, root_dir: str, capacity: int = 8, *, dtype=None):
        """``dtype=None`` keeps each tensor's dtype; otherwise convert once on put."""
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self.capacity = max(1, int(capacity))
        self.dtype = dtype
        self._cache: "collections.OrderedDict[PageKey, dict]" = collections.OrderedDict()
        self._lock = threading.RLock()
        self.stats = {
            "loads": 0, "stores": 0, "hits": 0, "misses": 0,
            "errors": 0, "evictions": 0,
        }

    def path(self, key: PageKey) -> str:
        return os.path.join(self.root_dir, key.filename())

    def has(self, key: PageKey) -> bool:
        with self._lock:
            return key in self._cache or os.path.exists(self.path(key))

    def put(self, key: PageKey, state: Dict[str, torch.Tensor], *, persist: bool = True) -> None:
        normalized = {
            str(name): tensor.detach().to(device="cpu", dtype=self.dtype or tensor.dtype).contiguous()
            for name, tensor in state.items()
        }
        with self._lock:
            self._cache[key] = normalized
            self._cache.move_to_end(key)
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
                self.stats["evictions"] += 1
        if persist:
            self._atomic_save(key, normalized)
            self.stats["stores"] += 1

    def get(self, key: PageKey, *, device: Optional[torch.device] = None) -> Optional[Dict[str, torch.Tensor]]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.stats["hits"] += 1
                state = self._cache[key]
            else:
                state = None
        if state is None:
            path = self.path(key)
            if not os.path.exists(path):
                self.stats["misses"] += 1
                return None
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.stats["loads"] += 1
                with self._lock:
                    self._cache[key] = state
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.capacity:
                        self._cache.popitem(last=False)
                        self.stats["evictions"] += 1
            except Exception:
                self.stats["errors"] += 1
                return None
        if device is not None:
            return {name: value.to(device=device, non_blocking=True) for name, value in state.items()}
        return dict(state)

    def _atomic_save(self, key: PageKey, state: dict) -> None:
        target = self.path(key)
        fd, tmp = tempfile.mkstemp(prefix=".chronos-page-", suffix=".pt", dir=self.root_dir)
        os.close(fd)
        try:
            torch.save(state, tmp)
            os.replace(tmp, target)
        except Exception:
            self.stats["errors"] += 1
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def prefetch(self, keys: Iterable[PageKey], device=None) -> Dict[PageKey, Optional[dict]]:
        return {key: self.get(key, device=device) for key in keys}

    def clear_memory(self) -> None:
        with self._lock:
            self._cache.clear()

    def summary(self) -> dict:
        with self._lock:
            resident = len(self._cache)
        return {**self.stats, "resident_pages": resident, "capacity": self.capacity,
                "root_dir": self.root_dir}


class PagedOptimizerState:
    """Adam moments persisted per expert page."""

    def __init__(self, root_dir: str, capacity: int = 8):
        self.store = TieredTensorStore(root_dir, capacity=capacity, dtype=torch.float32)

    def load(self, key: PageKey, params: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        payload = self.store.get(PageKey(key.layer, key.group, key.expert, "adam"))
        if payload is None:
            return {
                "exp_avg": {name: torch.zeros_like(value, dtype=torch.float32) for name, value in params.items()},
                "exp_avg_sq": {name: torch.zeros_like(value, dtype=torch.float32) for name, value in params.items()},
            }
        names = list(params)
        loaded = {
            "exp_avg": {name: payload.get(f"avg.{name}", torch.zeros_like(params[name], dtype=torch.float32)) for name in names},
            "exp_avg_sq": {name: payload.get(f"sq.{name}", torch.zeros_like(params[name], dtype=torch.float32)) for name in names},
        }
        step = payload.get("meta.step")
        if step is not None:
            loaded["step"] = int(step.item())
        return loaded

    def save(self, key: PageKey, state: Dict[str, Dict[str, torch.Tensor]], *, step: int = 0) -> None:
        payload = {}
        for name, value in state.get("exp_avg", {}).items():
            payload[f"avg.{name}"] = value
        for name, value in state.get("exp_avg_sq", {}).items():
            payload[f"sq.{name}"] = value
        payload["meta.step"] = torch.tensor(int(step), dtype=torch.int64)
        self.store.put(PageKey(key.layer, key.group, key.expert, "adam"), payload)
