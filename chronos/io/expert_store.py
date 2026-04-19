"""
chronos/io/expert_store.py

Three-tier expert weight storage: VRAM (hot) → RAM pinned (warm) → SSD (cold).

Responsibilities:
- Load expert weights from SSD into RAM pinned memory (async-friendly)
- Promote RAM-resident weights to VRAM on demand
- Evict VRAM weights via LRU when budget is exceeded
- Track per-expert residency state for soft-gating mask generation
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import contextlib
import os
import threading
import collections
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LRUCache:
    """Thread-safe LRU cache tracking which expert IDs are in a tier."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def contains(self, key: int) -> bool:
        with self._lock:
            return key in self._cache

    def touch(self, key: int):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)

    def put(self, key: int) -> Optional[int]:
        """Insert key; returns evicted key if capacity exceeded, else None."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return None
            evicted = None
            if len(self._cache) >= self.capacity:
                evicted, _ = self._cache.popitem(last=False)
            self._cache[key] = True
            return evicted

    def remove(self, key: int):
        with self._lock:
            self._cache.pop(key, None)

    def keys(self) -> List[int]:
        with self._lock:
            return list(self._cache.keys())

    def __len__(self):
        with self._lock:
            return len(self._cache)


class ExpertStore:
    """
    Manages three-tier storage for MoE expert weights.

    Tier 0 — VRAM: active experts as nn.Module parameters on GPU
    Tier 1 — RAM:  pinned-memory tensors ready for fast H2D transfer
    Tier 2 — SSD:  full expert weights saved as .pt files

    Usage:
        store = ExpertStore(model, config, ssd_dir="./expert_cache")
        store.offload_all_to_ssd()          # serialize experts to SSD
        store.prefetch_to_ram([0, 2])       # load experts 0,2 into RAM
        store.promote_to_vram(0)            # move expert 0 RAM→VRAM
        mask = store.vram_availability_mask()  # [num_experts] bool tensor
    """

    def __init__(self, model, config, ssd_dir: str = "./expert_cache"):
        self.config = config
        self.ssd_dir = ssd_dir
        self.device = next(model.parameters()).device
        os.makedirs(ssd_dir, exist_ok=True)

        # Dedicated CUDA stream for H2D weight transfers.
        # Runs concurrently with the default compute stream so weight copies
        # never stall matrix multiplications on the current token.
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self._h2d_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(device=self.device)
        else:
            self._h2d_stream = None

        # Collect all MoE layers and their expert modules
        from chronos.model.moe_chronos import ChronosMOEFeedForward
        self.moe_layers: List[ChronosMOEFeedForward] = [
            layer.mlp for layer in model.model.layers
            if isinstance(layer.mlp, ChronosMOEFeedForward)
        ]
        self.num_experts = config.num_experts
        self.num_layers = len(self.moe_layers)

        # Estimate VRAM capacity in number of experts
        expert_bytes = self._expert_size_bytes()
        vram_bytes = config.vram_budget_gb * (1024 ** 3)
        # Reserve 50% headroom for shared experts + attention KV cache
        self.vram_capacity = max(1, int(vram_bytes * 0.5 / max(expert_bytes, 1)))

        # Pinned RAM capacity: bounded by physical RAM safety limit
        # Never exceed pinned_memory_max_fraction of total physical RAM
        max_fraction = getattr(config, 'pinned_memory_max_fraction', 0.25)
        physical_ram_bytes = self._physical_ram_bytes()
        safe_pinned_bytes = physical_ram_bytes * max_fraction
        # Each expert occupies expert_bytes * num_layers in RAM
        per_expert_ram = max(expert_bytes * self.num_layers, 1)
        ram_capacity_by_safety = max(1, int(safe_pinned_bytes / per_expert_ram))
        # Also cap at 4x VRAM capacity (original heuristic)
        self.ram_capacity = min(self.vram_capacity * 4, ram_capacity_by_safety)

        self.vram_lru = LRUCache(capacity=self.vram_capacity)
        self.ram_lru = LRUCache(capacity=self.ram_capacity)

        # RAM pinned buffers: {expert_id: {layer_idx: tensor_dict}}
        self._ram_buffers: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._ram_lock = threading.Lock()

    @staticmethod
    def _physical_ram_bytes() -> int:
        """Query available (not total) RAM at call time for dynamic safety."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            # Use available memory so we respect current system load
            return vm.available
        except ImportError:
            return 8 * (1024 ** 3)  # conservative 8 GB fallback

    def _safe_ram_capacity(self) -> int:
        """Recompute RAM capacity based on current available memory."""
        max_fraction = getattr(self.config, 'pinned_memory_max_fraction', 0.25)
        available_bytes = self._physical_ram_bytes()
        safe_bytes = available_bytes * max_fraction
        per_expert_ram = max(self._expert_size_bytes() * self.num_layers, 1)
        return max(1, min(self.vram_capacity * 4, int(safe_bytes / per_expert_ram)))

    def _expert_size_bytes(self) -> int:
        if not self.moe_layers:
            return 0
        expert = self.moe_layers[0].experts[0]
        return sum(p.numel() * 2 for p in expert.parameters())  # fp16

    def _ssd_path(self, expert_id: int, layer_idx: int) -> str:
        return os.path.join(self.ssd_dir, f"expert_l{layer_idx}_e{expert_id}.pt")

    # ── SSD operations ────────────────────────────────────────────

    def offload_all_to_ssd(self):
        """Serialize all expert weights to SSD as fp16 .pt files."""
        for li, moe in enumerate(self.moe_layers):
            for ei, expert in enumerate(moe.experts):
                path = self._ssd_path(ei, li)
                if not os.path.exists(path):
                    state = {k: v.half().cpu() for k, v in expert.state_dict().items()}
                    torch.save(state, path)

    def _load_from_ssd(self, expert_id: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Load expert weights for all layers from SSD into CPU tensors."""
        layer_states = {}
        for li in range(self.num_layers):
            path = self._ssd_path(expert_id, li)
            if os.path.exists(path):
                layer_states[li] = torch.load(path, map_location="cpu")
        return layer_states

    # ── RAM operations ────────────────────────────────────────────

    def prefetch_to_ram(self, expert_ids: List[int]):
        """
        Synchronously load expert weights from SSD into pinned RAM.
        Call this from a background thread for async prefetch.
        """
        for eid in expert_ids:
            if self.ram_lru.contains(eid):
                self.ram_lru.touch(eid)
                continue
            # Dynamically check available RAM before each prefetch
            dynamic_cap = self._safe_ram_capacity()
            self.ram_lru.capacity = dynamic_cap

            layer_states = self._load_from_ssd(eid)
            if not layer_states:
                continue
            pinned = {}
            for li, state in layer_states.items():
                pinned[li] = {}
                for k, v in state.items():
                    try:
                        pinned[li][k] = v.pin_memory() if not v.is_pinned() else v
                    except RuntimeError:
                        pinned[li][k] = v.cpu()
            with self._ram_lock:
                evicted = self.ram_lru.put(eid)
                if evicted is not None:
                    self._ram_buffers.pop(evicted, None)
                self._ram_buffers[eid] = pinned

    # ── VRAM operations ───────────────────────────────────────────

    def promote_to_vram(self, expert_id: int) -> bool:
        """
        Move expert from RAM → VRAM. Returns True if successful.
        If expert not in RAM, loads from SSD first (blocking).

        The H2D copy runs on a dedicated CUDA stream (_h2d_stream) so it
        never stalls the default compute stream mid-token.  The caller must
        synchronize (_h2d_stream.synchronize()) before the expert is used.
        """
        if not self.ram_lru.contains(expert_id):
            self.prefetch_to_ram([expert_id])

        with self._ram_lock:
            if expert_id not in self._ram_buffers:
                return False
            layer_states = self._ram_buffers[expert_id]

        evicted = self.vram_lru.put(expert_id)
        if evicted is not None:
            self._evict_from_vram(evicted)

        ctx = (
            torch.cuda.stream(self._h2d_stream)
            if self._h2d_stream is not None
            else _null_context()
        )
        with ctx:
            for li, state in layer_states.items():
                expert = self.moe_layers[li].experts[expert_id]
                expert.load_state_dict(
                    {k: v.to(self.device, non_blocking=True) for k, v in state.items()},
                    strict=False,
                )
        return True

    def sync_h2d(self):
        """Wait for all pending H2D weight transfers to complete."""
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()

    def _evict_from_vram(self, expert_id: int):
        """Write VRAM expert back to RAM buffer before eviction."""
        for li, moe in enumerate(self.moe_layers):
            if expert_id < len(moe.experts):
                state = {k: v.half().cpu().pin_memory()
                         for k, v in moe.experts[expert_id].state_dict().items()}
                with self._ram_lock:
                    if expert_id not in self._ram_buffers:
                        self._ram_buffers[expert_id] = {}
                    self._ram_buffers[expert_id][li] = state
        self.vram_lru.remove(expert_id)

    # ── Mask generation ───────────────────────────────────────────

    def vram_availability_mask(self) -> torch.Tensor:
        """Returns [num_experts] bool tensor: True if expert is in VRAM."""
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        for eid in self.vram_lru.keys():
            if eid < self.num_experts:
                mask[eid] = True
        return mask

    def stats(self) -> dict:
        available_ram_gb = self._physical_ram_bytes() / (1024 ** 3)
        pinned_used_gb = (
            len(self.ram_lru) * self._expert_size_bytes() * self.num_layers / (1024 ** 3)
        )
        dynamic_cap = self._safe_ram_capacity()
        return {
            "vram_experts": len(self.vram_lru),
            "vram_capacity": self.vram_capacity,
            "ram_experts": len(self.ram_lru),
            "ram_capacity_dynamic": dynamic_cap,
            "expert_size_kb": self._expert_size_bytes() // 1024,
            "pinned_ram_used_gb": round(pinned_used_gb, 3),
            "available_ram_gb": round(available_ram_gb, 1),
            "pinned_ram_fraction": round(pinned_used_gb / max(available_ram_gb, 1), 4),
            "h2d_stream": "dedicated" if self._h2d_stream is not None else "default",
        }


@contextlib.contextmanager
def _null_context():
    yield
