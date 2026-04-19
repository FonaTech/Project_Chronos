"""
chronos/runtime/cache_manager.py

Unified VRAM/RAM cache manager for Project Chronos inference.

Wraps ExpertStore + AsyncPrefetcher into a single interface used by
the inference engine. Provides:
- availability_mask() for soft gating
- step() to advance prefetch schedule after each decode step
- stats() for monitoring
"""
from typing import List, Optional
import torch

from chronos.io.expert_store import ExpertStore
from chronos.io.async_prefetcher import AsyncPrefetcher, PrefetchScheduler


class CacheManager:
    """
    Single entry point for all expert caching operations during inference.

    Lifecycle:
        mgr = CacheManager(model, config, ssd_dir="./expert_cache")
        mgr.start()
        for each decode step:
            mask = mgr.availability_mask()          # pass to model.forward
            outputs, lp = model(x, available_expert_masks=[mask]*L)
            mgr.step(lp, current_expert_ids)        # schedule next prefetch
        mgr.stop()
    """

    def __init__(self, model, config, ssd_dir: str = "./expert_cache"):
        self.config = config
        self.expert_store = ExpertStore(model, config, ssd_dir=ssd_dir)
        self.prefetcher = AsyncPrefetcher(self.expert_store)
        self.scheduler = PrefetchScheduler(self.prefetcher, self.expert_store)
        self._num_layers = len(self.expert_store.moe_layers)

    def start(self):
        """Start background prefetch thread."""
        self.prefetcher.start()

    def stop(self):
        """Stop background prefetch thread."""
        self.prefetcher.stop()

    def warm_up(self, initial_expert_ids: Optional[List[int]] = None):
        """
        Pre-load a set of experts into VRAM before generation starts.
        Defaults to experts 0..vram_capacity-1.
        """
        if initial_expert_ids is None:
            initial_expert_ids = list(range(
                min(self.config.num_experts, self.expert_store.vram_capacity)
            ))
        self.expert_store.prefetch_to_ram(initial_expert_ids)
        for eid in initial_expert_ids:
            self.expert_store.promote_to_vram(eid)
        # Ensure all H2D copies are complete before model sees the weights
        self.expert_store.sync_h2d()

    def availability_mask(self) -> torch.Tensor:
        """[num_experts] bool — True if expert is currently in VRAM."""
        return self.expert_store.vram_availability_mask()

    def availability_masks_all_layers(self) -> List[torch.Tensor]:
        """Returns the same mask replicated for each MoE layer."""
        mask = self.availability_mask()
        return [mask] * self._num_layers

    def step(self, lookahead_probs, current_expert_ids: List[int]):
        """
        Advance the prefetch schedule after one decode step.

        Args:
            lookahead_probs: [B, S, K+1, E] from LookaheadRouter (or None)
            current_expert_ids: expert IDs used in this step (for VRAM promotion)
        """
        self.scheduler.step(lookahead_probs, current_expert_ids)
        # Synchronize dedicated H2D stream so any just-promoted weights are
        # visible to the compute stream before the next token's forward pass.
        self.expert_store.sync_h2d()

    def stats(self) -> dict:
        return {
            **self.expert_store.stats(),
            **self.prefetcher.stats,
        }
