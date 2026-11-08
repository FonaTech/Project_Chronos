"""
chronos/io/expert_store.py

Three-tier expert weight storage: VRAM (hot) → RAM pinned (warm) → SSD (cold).

Responsibilities:
- Load expert weights from SSD into RAM pinned memory (async-friendly)
- Promote RAM-resident weights to VRAM on demand
- Evict VRAM weights via LRU when budget is exceeded
- Track per-expert residency state for soft-gating mask generation
"""
import contextlib
import os
import threading
import collections
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from chronos.io.storage import ClusterStorage, MANIFEST_FILENAME
from chronos.io.io_simulator import maybe_sleep as _maybe_sleep_ssd

try:
    from chronos.runtime.metrics import safe_record as _metric
except Exception:
    def _metric(name, value):  # fallback
        pass


def expert_budget_pages(config, page_bytes: int):
    """(max_resident_pages, ram_budget_pages) from the hard budget fields.

    ``max_resident_experts`` counts fast-tier pages directly; ``ram_budget_mb``
    is converted with the page size.  Either may be None (no hard cap).
    """
    resident = getattr(config, "max_resident_experts", None)
    budget_mb = getattr(config, "ram_budget_mb", None)
    resident = None if resident in (None, "", 0) else max(1, int(resident))
    ram = None
    if budget_mb not in (None, "", 0):
        ram = max(1, int(float(budget_mb) * 2 ** 20 // max(int(page_bytes), 1)))
    return resident, ram


def _process_rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


class LRUCache:
    """Thread-safe LRU cache tracking which expert IDs are in a tier."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()
        # Keys the predictor says are about to be used: evicted only when
        # every resident key is protected (the hard capacity always holds).
        self.protected: frozenset = frozenset()

    def _victim_locked(self):
        for key in self._cache:
            if key not in self.protected:
                return key
        return next(iter(self._cache))

    def victim(self):
        """Key the next insertion would evict (coldest unprotected), or None."""
        with self._lock:
            return self._victim_locked() if self._cache else None

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
                evicted = self._victim_locked()
                self._cache.pop(evicted)
            self._cache[key] = True
            return evicted

    def remove(self, key: int):
        with self._lock:
            self._cache.pop(key, None)

    def keys(self) -> List[int]:
        with self._lock:
            return list(self._cache.keys())

    def reorder_by_score(self, scores: Dict[int, float]):
        """Order entries from coldest to hottest so LRU evicts low scores."""
        with self._lock:
            ordered = sorted(
                self._cache.keys(),
                key=lambda key: (float(scores.get(key, 0.0)), repr(key)),
            )
            self._cache = collections.OrderedDict((key, True) for key in ordered)

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
        self.model = model
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
        self.layer_aware = bool(getattr(config, "layer_aware_cache", False))
        self._layer_to_group = self._build_layer_group_map()

        # Estimate VRAM capacity in number of experts
        self._expert_dtype_cached = self._expert_dtype()
        expert_bytes = self._expert_size_bytes()
        self._expert_bytes = expert_bytes
        page_bytes = expert_bytes * (1 if self.layer_aware else self.num_layers)
        total_pages = self.num_experts * (self.num_layers if self.layer_aware else 1)
        vram_bytes = config.vram_budget_gb * (1024 ** 3)
        # Reserve 50% headroom for shared experts + attention KV cache
        self.vram_capacity = min(total_pages, max(1, int(vram_bytes * 0.5 / max(page_bytes, 1))))

        # Pinned RAM capacity: bounded by physical RAM safety limit
        # Never exceed pinned_memory_max_fraction of total physical RAM
        max_fraction = getattr(config, 'pinned_memory_max_fraction', 0.25)
        physical_ram_bytes = self._physical_ram_bytes()
        safe_pinned_bytes = physical_ram_bytes * max_fraction
        # Each expert occupies expert_bytes * num_layers in RAM
        per_expert_ram = max(expert_bytes if self.layer_aware else expert_bytes * self.num_layers, 1)
        ram_capacity_by_safety = max(1, int(safe_pinned_bytes / per_expert_ram))
        # Also cap at 4x VRAM capacity (original heuristic)
        self.ram_capacity = min(total_pages, self.vram_capacity * 4, ram_capacity_by_safety)
        # Hard user budget (max_resident_experts / ram_budget_mb) on top of
        # the heuristics; counted in cache pages, at least one page each.
        self.max_resident_pages, self.ram_budget_pages = expert_budget_pages(config, page_bytes)
        if self.max_resident_pages is not None:
            self.vram_capacity = min(self.vram_capacity, self.max_resident_pages)
        if self.ram_budget_pages is not None:
            self.ram_capacity = min(self.ram_capacity, self.ram_budget_pages)
        self._peak_rss_bytes = 0

        self.vram_lru = LRUCache(capacity=self.vram_capacity)
        self.ram_lru = LRUCache(capacity=self.ram_capacity)

        # RAM pinned buffers: {expert_id: {layer_idx: tensor_dict}}
        self._ram_buffers: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._ram_lock = threading.Lock()
        self._load_lock = threading.RLock()
        self._io_stats = {"ssd_page_reads": 0, "expert_evictions": 0}
        self._offloaded = False

        # M3: per-expert H2D events. Every promote_to_vram records an event
        # on _h2d_stream and stores it here; wait_for_experts synchronizes
        # only on the needed ones so the compute stream never blocks on
        # unrelated transfers.
        self._pending_events: Dict[int, "torch.cuda.Event"] = {}
        self._events_lock = threading.Lock()

        # Storage backend: "safetensors" (clustered) or "pt" (legacy, one file per expert)
        self.storage_format = getattr(config, "storage_format", "safetensors")
        self._cluster_storage: Optional[ClusterStorage] = None
        if self.storage_format == "safetensors" and ClusterStorage.has_manifest(ssd_dir):
            # Manifest already exists on disk — load it up front so prefetch
            # can run cluster-aware from step 0.
            self._cluster_storage = ClusterStorage(ssd_dir)

    def _build_layer_group_map(self) -> Dict[int, int]:
        prelude = int(getattr(self.config, "prelude_layers", 0) or 0)
        groups = int(getattr(self.config, "recurrent_groups", self.num_layers) or self.num_layers)
        mapping = {}
        for li in range(self.num_layers):
            mapping[li] = li - prelude if prelude <= li < prelude + groups else -1
        return mapping

    def expert_key(self, layer_id: int, expert_id: int) -> tuple[int, int, int]:
        return (int(layer_id), int(self._layer_to_group.get(int(layer_id), -1)), int(expert_id))

    def _expand_keys(self, items) -> list:
        """Normalize legacy expert IDs and v2 page keys."""
        keys = []
        for item in items or []:
            if isinstance(item, (tuple, list)):
                if len(item) == 3:
                    keys.append((int(item[0]), int(item[1]), int(item[2])))
                elif len(item) == 2:
                    keys.append(self.expert_key(int(item[0]), int(item[1])))
                else:
                    raise ValueError(f"invalid expert page key: {item!r}")
            elif self.layer_aware:
                keys.extend(self.expert_key(li, int(item)) for li in range(self.num_layers))
            else:
                keys.append(int(item))
        return list(dict.fromkeys(keys))

    @staticmethod
    def _page_parts(key) -> tuple[int, int]:
        if not isinstance(key, tuple):
            raise TypeError(f"expected a layer-aware expert key, got {key!r}")
        return int(key[0]), int(key[2])

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
        per_expert_ram = max(
            self._expert_size_bytes() if self.layer_aware else self._expert_size_bytes() * self.num_layers,
            1,
        )
        cap = min(self.ram_capacity, self.vram_capacity * 4, int(safe_bytes / per_expert_ram))
        if getattr(self, "ram_budget_pages", None) is not None:
            cap = min(cap, self.ram_budget_pages)
        return max(1, cap)

    def set_protected_experts(self, keys) -> None:
        """Pin predicted pages against eviction (expanded like prefetch keys)."""
        protected = frozenset(self._expand_keys(list(keys or ())))
        self.vram_lru.protected = protected
        self.ram_lru.protected = protected

    def _expert_size_bytes(self) -> int:
        if not self.moe_layers:
            return 0
        expert = self.moe_layers[0].experts[0]
        size = sum(p.numel() * p.element_size() for p in expert.parameters())
        if size:
            return size
        cached = getattr(self, "_expert_bytes", 0)
        if cached:
            return cached
        # Lazy placeholders deliberately have no parameters. Fall back to the
        # Chronos FeedForward topology so cache capacities stay meaningful
        # when experts are replaced before model.to(device).
        hidden = int(getattr(self.config, "hidden_size", 0) or 0)
        intermediate = int(getattr(self.config, "moe_intermediate_size", 0) or 0)
        if hidden and intermediate:
            dtype = self._expert_dtype()
            return 3 * hidden * intermediate * torch.empty((), dtype=dtype).element_size()
        return 0

    def _expert_dtype(self) -> torch.dtype:
        """Expert pages live in the trunk's compute dtype (one precision
        policy end to end): a BF16 trunk gets BF16 pages, an FP32 CPU trunk
        FP32 pages.  SSD shards of another dtype are converted once when they
        are read into RAM, never per forward."""
        from chronos.precision import torch_model_dtype
        try:
            return torch_model_dtype(self.model)
        except StopIteration:
            return torch.float32

    def _build_live_expert(self):
        from chronos.model.kernel import FeedForward
        # No random initialization or temporary full allocation on a cache miss.
        with torch.device("meta"):
            return FeedForward(self.config, intermediate_size=self.config.moe_intermediate_size)

    def _build_placeholder_expert(self):
        from chronos.model.moe_chronos import LazyExpertPlaceholder
        return LazyExpertPlaceholder(
            self.config.hidden_size,
            self.config.moe_intermediate_size,
            self._expert_dtype(),
        )

    def _replace_expert_module(self, layer_idx: int, expert_id: int, module: nn.Module):
        self.moe_layers[layer_idx].experts[expert_id] = module

    @staticmethod
    def _expert_state_from_checkpoint(
        state_dict: Dict[str, torch.Tensor],
        layer_idx: int,
        expert_id: int,
    ) -> Dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_id}."
        state = {}
        for key, val in state_dict.items():
            if key.startswith(prefix):
                state[key[len(prefix):]] = val.detach().cpu().contiguous()
        return state

    def offload_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        clusters: Optional[List[List[int]]] = None,
        replace_live: bool = True,
    ):
        """Write expert weights to SSD directly from a full checkpoint.

        This is the real lazy-load path for CPU/MPS inference: we avoid
        materializing MoE expert weights into the live model before setup.
        """
        if clusters is not None:
            # Paging by the training clusters (lambda_cluster_locality > 0):
            # one .ctsr per cluster, so one SSD read brings a stream's experts.
            self._write_clusters_from(
                lambda li, ei: self._expert_state_from_checkpoint(state_dict, li, ei), clusters)
            if replace_live:
                self.replace_live_experts_with_placeholders()
            return
        if self.storage_format == "safetensors":
            # A raw .pth checkpoint has no co-activation cluster layout.
            # Falling back to one .pt shard per (layer, expert) preserves
            # true lazy loading; writing a single all-expert .ctsr cluster
            # would make the first miss pull every expert into RAM.
            self.storage_format = "pt"
            self._cluster_storage = None

        for li in range(self.num_layers):
            for ei in range(self.num_experts):
                path = self._ssd_path(ei, li)
                state = self._expert_state_from_checkpoint(state_dict, li, ei)
                if state:
                    torch.save(state, path)

        if replace_live:
            self.replace_live_experts_with_placeholders()

    def offload_from_state_reader(
        self,
        reader,
        clusters: Optional[List[List[int]]] = None,
        replace_live: bool = True,
    ):
        """Write expert shards to SSD from a random-access export reader.

        This path is used for Chronos safetensors/GGUF exports. It avoids
        materializing the full expert state dict in DRAM: each expert tensor is
        pulled from the export artifact only while that expert shard is written.
        """
        keys = list(reader.keys())
        if clusters is not None:
            def from_reader(li, ei):
                prefix = f"model.layers.{li}.mlp.experts.{ei}."
                return {k[len(prefix):]: reader.get_tensor(k).detach().cpu().contiguous()
                        for k in keys if k.startswith(prefix)}
            self._write_clusters_from(from_reader, clusters)
            if replace_live:
                self.replace_live_experts_with_placeholders()
            return
        if self.storage_format == "safetensors":
            self.storage_format = "pt"
            self._cluster_storage = None

        for li in range(self.num_layers):
            for ei in range(self.num_experts):
                path = self._ssd_path(ei, li)
                prefix = f"model.layers.{li}.mlp.experts.{ei}."
                state = {}
                for key in keys:
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = (
                            reader.get_tensor(key)
                            .detach()
                            .cpu()
                            .contiguous()
                        )
                if state:
                    torch.save(state, path)

        if replace_live:
            self.replace_live_experts_with_placeholders()

    def _write_clusters_from(self, get_state, clusters: List[List[int]]):
        """Shard experts into one ``.ctsr`` per cluster from ``get_state(layer, expert)``."""
        dtype = self._expert_dtype_cached

        def cast(li, ei):
            return {k: (v.to(dtype=dtype) if v.is_floating_point() and v.dtype != dtype else v)
                    for k, v in get_state(li, ei).items()}

        ClusterStorage.write_clusters_from_states(
            cast, clusters=clusters, output_dir=self.ssd_dir, num_layers=self.num_layers,
            num_experts=self.num_experts, layer_to_group=self._layer_to_group if self.layer_aware else None,
        )
        self.storage_format = "safetensors"
        self._cluster_storage = ClusterStorage(self.ssd_dir)

    def replace_live_experts_with_placeholders(self):
        """Drop live expert modules from the active model after SSD offload."""
        for li in range(self.num_layers):
            for ei in range(self.num_experts):
                self._replace_expert_module(li, ei, self._build_placeholder_expert())
        self._offloaded = True

    def _ssd_path(self, expert_id: int, layer_idx: int) -> str:
        return os.path.join(self.ssd_dir, f"expert_l{layer_idx}_e{expert_id}.pt")

    # ── SSD operations ────────────────────────────────────────────

    def offload_all_to_ssd(self, clusters: Optional[List[List[int]]] = None):
        """
        Serialize all expert weights to SSD.

        - storage_format == "safetensors" (default): writes one .ctsr file per
          cluster + cluster_manifest.json. If `clusters` is None, falls back
          to a single all-experts cluster (still mmap-friendly, just no
          co-occurrence reordering benefit).
        - storage_format == "pt" (legacy): one .pt pickle per (expert, layer).
        """
        if self.storage_format == "safetensors":
            self._offload_safetensors(clusters)
        else:
            self._offload_legacy_pt()

    def _offload_legacy_pt(self):
        for li, moe in enumerate(self.moe_layers):
            for ei, expert in enumerate(moe.experts):
                path = self._ssd_path(ei, li)
                state = {k: v.detach().cpu().contiguous() for k, v in expert.state_dict().items()}
                torch.save(state, path)
        self.replace_live_experts_with_placeholders()

    def _offload_safetensors(self, clusters: Optional[List[List[int]]]):
        if clusters is None:
            # Without calibrated co-activation data, do not couple every expert.
            clusters = [[eid] for eid in range(self.num_experts)]
        ClusterStorage.write_clusters(
            moe_layers=self.moe_layers,
            clusters=clusters,
            output_dir=self.ssd_dir,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            dtype=self._expert_dtype_cached,
            layer_to_group=self._layer_to_group if self.layer_aware else None,
        )
        self._cluster_storage = ClusterStorage(self.ssd_dir)
        self.replace_live_experts_with_placeholders()

    def attach_cluster_manifest(self, manifest_dir: Optional[str] = None):
        """Load (or reload) a cluster_manifest.json. Called by CacheManager
        after an offline repack, or to point ExpertStore at a different layout."""
        target = manifest_dir or self.ssd_dir
        if ClusterStorage.has_manifest(target):
            self._cluster_storage = ClusterStorage(target)
            self.storage_format = "safetensors"
            issues = self._cluster_storage.manifest.validate(
                num_layers=self.num_layers, num_experts=self.num_experts
            )
            if issues:
                import warnings
                warnings.warn(
                    "cluster manifest topology mismatch: " + "; ".join(issues),
                    RuntimeWarning,
                    stacklevel=2,
                )
            return not issues
        return False

    def _load_from_ssd(self, expert_id: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Load expert weights for all layers from SSD into CPU tensors.

        Legacy .pt path. The clustered safetensors path uses
        `_load_cluster_into_ram` instead.
        """
        layer_states: Dict[int, Dict[str, torch.Tensor]] = {}
        for li in range(self.num_layers):
            path = self._ssd_path(expert_id, li)
            if os.path.exists(path):
                layer_states[li] = torch.load(path, map_location="cpu")
        return layer_states

    # ── RAM operations ────────────────────────────────────────────

    def prefetch_to_ram(self, expert_ids):
        """
        Synchronously load expert weights from SSD into pinned RAM.
        Call this from a background thread for async prefetch.

        Cluster-aware path: when a safetensors manifest is attached, each
        request expands to its full Louvain cluster so co-occurring experts
        arrive in one sequential mmap — this is where we actually convert
        random reads into sequential reads. Requests that land in the same
        cluster are deduplicated.
        """
        expert_ids = self._expand_keys(expert_ids)
        if not expert_ids:
            return

        # Serialize foreground demand loads and the background prefetcher.
        # The RAM lock stays short so promotions can take a stable reference.
        with self._load_lock:
            if self._cluster_storage is not None:
                self._prefetch_clustered(expert_ids)
            else:
                self._prefetch_legacy(expert_ids)

    def _cache_ram_entry(self, key, states):
        cap = self._safe_ram_capacity()
        with self._ram_lock:
            self.ram_lru.capacity = cap
            self.ram_lru.remove(key)
            while len(self.ram_lru) >= cap:
                victim = self.ram_lru.victim()
                self.ram_lru.remove(victim)
                self._ram_buffers.pop(victim, None)
            self.ram_lru.put(key)
            self._ram_buffers[key] = states

    def _prefetch_clustered(self, expert_ids):
        cs = self._cluster_storage
        # Keep each requested page last in its cluster batch, so neighboring
        # pages can never evict the page the caller is about to promote.
        for request in expert_ids:
            if self.ram_lru.contains(request):
                self.ram_lru.touch(request)
                continue
            li, eid = self._page_parts(request) if self.layer_aware else (None, int(request))
            if eid not in cs.manifest.expert_to_cluster:
                continue
            cid = cs.cluster_for_expert(eid)
            cap = self._safe_ram_capacity()
            neighbors = [e for e in cs.experts_in_cluster(cid) if e != eid]
            candidates = neighbors[:max(0, cap - 1)] + [eid]
            keys = [self.expert_key(li, e) if self.layer_aware else e for e in candidates]
            keys = [key for key in keys if not self.ram_lru.contains(key)]
            pages = set()
            for key in keys:
                if self.layer_aware:
                    pages.add(self._page_parts(key))
                else:
                    pages.update((layer, key) for layer in range(self.num_layers))
            _maybe_sleep_ssd()
            nested = cs.load_cluster(cid, pages=pages)
            self._io_stats["ssd_page_reads"] += len(pages)
            for key in keys:
                if self.layer_aware:
                    layer, expert = self._page_parts(key)
                    state = nested.get(expert, {}).get(layer)
                    entry = {layer: state} if state is not None else {}
                else:
                    entry = nested.get(key, {})
                if entry:
                    self._cache_ram_entry(key, self._pin_layer_states(entry))

    def _prefetch_legacy(self, expert_ids: List[int]):
        for request in expert_ids:
            if self.ram_lru.contains(request):
                self.ram_lru.touch(request)
                continue
            # Simulate SSD read latency
            _maybe_sleep_ssd()
            dynamic_cap = self._safe_ram_capacity()
            self.ram_lru.capacity = dynamic_cap

            if self.layer_aware:
                li, eid = self._page_parts(request)
                path = self._ssd_path(eid, li)
                layer_states = {li: torch.load(path, map_location="cpu")} if os.path.exists(path) else {}
            else:
                eid = int(request)
                layer_states = self._load_from_ssd(eid)
            if not layer_states:
                continue
            self._io_stats["ssd_page_reads"] += len(layer_states)
            self._cache_ram_entry(request, self._pin_layer_states(layer_states))

    def _pin_layer_states(self, layer_states):
        pinned = {}
        for li, state in layer_states.items():
            pinned[li] = {}
            for k, v in state.items():
                value = v.to(dtype=self._expert_dtype_cached, device="cpu")
                # CPU/MPS have no CUDA DMA benefit. Some torch/macOS builds
                # segfault here instead of raising an exception.
                if self._h2d_stream is not None and not value.is_pinned():
                    value = value.pin_memory()
                pinned[li][k] = value
        return pinned

    # ── VRAM operations ───────────────────────────────────────────

    def promote_to_vram(self, expert_id, blocking: bool = False) -> bool:
        """
        Move expert from RAM → VRAM. Returns True if the H2D copy was issued.

        blocking=False (default, M3 path): issues H2D on _h2d_stream, records
        an event, and returns immediately. Caller must wait on the event (via
        ``wait_for_experts``) before the expert is used by compute.

        blocking=True (legacy path): synchronizes _h2d_stream before return,
        so the expert is guaranteed visible to the compute stream.
        """
        requests = self._expand_keys([expert_id])
        if len(requests) > 1:
            changed = False
            for request in requests:
                changed = self.promote_to_vram(request, blocking=False) or changed
            if blocking:
                self.sync_h2d()
            return changed
        expert_id = requests[0]
        if self.vram_lru.contains(expert_id):
            self.touch_expert(expert_id)
            if blocking:
                self.wait_for_experts([expert_id])
            return False
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
                eid = self._page_parts(expert_id)[1] if self.layer_aware else int(expert_id)
                expert = self.moe_layers[li].experts[eid]
                if not isinstance(expert, nn.Module) or len(list(expert.parameters())) == 0:
                    expert = self._build_live_expert()
                    self._replace_expert_module(li, eid, expert)
                expert.load_state_dict(
                    {k: v.to(device=self.device, dtype=self._expert_dtype_cached, non_blocking=True)
                     for k, v in state.items()}, strict=True, assign=True,
                )
                expert.requires_grad_(False)
                expert.eval()
            # Record an event on the H2D stream so consumers can wait on
            # *just this expert* instead of the whole stream.
            if self._h2d_stream is not None:
                evt = torch.cuda.Event(blocking=False)
                evt.record(self._h2d_stream)
                with self._events_lock:
                    self._pending_events[expert_id] = evt

        if blocking:
            self.sync_h2d()
        _metric("vram_experts", len(self.vram_lru))
        _metric("ram_experts", len(self.ram_lru))
        return True

    def wait_for_experts(self, expert_ids: List[int]) -> None:
        """Block the current (compute) stream until the named experts' H2D
        copies are done. No-op on CPU. Only syncs the relevant events; other
        in-flight transfers keep running."""
        if self._h2d_stream is None:
            return
        current = torch.cuda.current_stream(self.device)
        expert_ids = self._expand_keys(expert_ids)
        with self._events_lock:
            events = [self._pending_events.pop(eid, None) for eid in expert_ids]
        for evt in events:
            if evt is not None:
                # Make the compute stream wait for this event rather than
                # calling evt.synchronize() on the host, which would block
                # the CPU too.
                current.wait_event(evt)

    def touch_expert(self, expert_id) -> None:
        """Mark an expert as recently useful in every resident tier."""
        for key in self._expand_keys([expert_id]):
            self.vram_lru.touch(key)
            self.ram_lru.touch(key)

    def reprioritize_resident_experts(self, scores: Dict[int, float]) -> None:
        """Use recent router scores to make low-probability experts evict first."""
        self.vram_lru.reorder_by_score(scores)
        self.ram_lru.reorder_by_score(scores)

    def sync_h2d(self):
        """Wait for *all* pending H2D weight transfers to complete (legacy)."""
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()
        with self._events_lock:
            self._pending_events.clear()

    def _evict_from_vram(self, expert_id):
        # Inference weights are immutable and already persisted. Copying them
        # back on eviction wastes bandwidth and can displace the demanded RAM
        # page; on CPU it also creates a duplicate of shared storage.
        self.wait_for_experts([expert_id])
        if self.layer_aware:
            li, eid = self._page_parts(expert_id)
            candidates = [(li, eid)]
        else:
            candidates = [(li, int(expert_id)) for li in range(self.num_layers)]
        for li, eid in candidates:
            expert = self.moe_layers[li].experts[eid]
            if self._h2d_stream is not None:
                current = torch.cuda.current_stream(self.device)
                for param in expert.parameters():
                    param.record_stream(current)
            self._replace_expert_module(li, eid, self._build_placeholder_expert())
        self.vram_lru.remove(expert_id)
        self._io_stats["expert_evictions"] += 1

    # ── Mask generation ───────────────────────────────────────────

    def vram_availability_mask(self) -> torch.Tensor:
        """Returns [num_experts] bool tensor: True if expert is in VRAM."""
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        for eid in self.vram_lru.keys():
            if isinstance(eid, tuple):
                eid = eid[2]
            if eid < self.num_experts:
                mask[eid] = True
        return mask

    def vram_availability_masks(self) -> List[torch.Tensor]:
        """Return real per-physical-layer masks for the v2 cache."""
        if not self.layer_aware:
            mask = self.vram_availability_mask()
            return [mask.clone() for _ in range(self.num_layers)]
        masks = [torch.zeros(self.num_experts, dtype=torch.bool) for _ in range(self.num_layers)]
        for key in self.vram_lru.keys():
            li, eid = self._page_parts(key)
            if 0 <= li < self.num_layers and 0 <= eid < self.num_experts:
                masks[li][eid] = True
        return masks

    def stats(self) -> dict:
        available_ram_gb = self._physical_ram_bytes() / (1024 ** 3)
        with self._ram_lock:
            tensors = [v for layers in self._ram_buffers.values()
                       for state in layers.values() for v in state.values()]
        live = [p for moe in self.moe_layers for expert in moe.experts for p in expert.parameters()]
        ram_bytes = sum(v.numel() * v.element_size() for v in tensors)
        live_bytes = sum(v.numel() * v.element_size() for v in live)
        pinned_bytes = sum(v.numel() * v.element_size() for v in tensors
                           if self._h2d_stream is not None and v.is_pinned())
        # CPU modules share RAM pages. Count storage once across the two tiers.
        unique = {}
        for tensor in tensors + live:
            storage = tensor.untyped_storage()
            unique[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        rss = _process_rss_bytes()
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        return {
            "device": str(self.device),
            "rss_mb": round(rss / 2 ** 20, 1),
            "peak_rss_mb": round(self._peak_rss_bytes / 2 ** 20, 1),
            "max_resident_pages": self.max_resident_pages,
            "ram_budget_pages": self.ram_budget_pages,
            "protected_pages": len(self.vram_lru.protected),
            "vram_experts": len(self.vram_lru),
            "vram_capacity": self.vram_capacity,
            "ram_experts": len(self.ram_lru),
            "ram_capacity_dynamic": self._safe_ram_capacity(),
            "expert_size_kb": self._expert_size_bytes() // 1024,
            "cache_page_bytes": self._expert_size_bytes() * (1 if self.layer_aware else self.num_layers),
            "resident_expert_bytes": live_bytes,
            "ram_buffer_bytes": ram_bytes,
            "unique_expert_bytes": sum(unique.values()),
            "pinned_ram_used_gb": round(pinned_bytes / (1024 ** 3), 6),
            "available_ram_gb": round(available_ram_gb, 1),
            "pinned_ram_fraction": round(pinned_bytes / max(available_ram_gb * 1024 ** 3, 1), 4),
            "h2d_stream": "dedicated" if self._h2d_stream is not None else "default",
            "storage_format": self.storage_format,
            "cluster_aware": self._cluster_storage is not None,
            "layer_aware": self.layer_aware,
            "num_clusters": len(self._cluster_storage.manifest.clusters) if self._cluster_storage else 0,
            **self._io_stats,
        }


@contextlib.contextmanager
def _null_context():
    yield
