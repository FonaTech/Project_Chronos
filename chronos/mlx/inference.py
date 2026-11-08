"""
chronos/mlx/inference.py — MLX inference engine for Project Chronos.

Mirrors ChronosInferenceEngine (PyTorch) but uses:
- ChronosMLXModel for forward passes
- MLXExpertStore (unified memory, no H2D stream)
- Lazy MLX graphs evaluated per decode step (decode shapes change every
  token, so the step is not traced with mx.compile; RoPE uses a shapeless
  compiled kernel)
- Async prefetch via Python threading (SSD → unified memory)

Usage:
    engine = ChronosMLXInferenceEngine(model, config)
    for token_id in engine.generate("Explain how Chronos moves expert IO into prefill.", max_new_tokens=128):
        print(tokenizer.decode([token_id]), end="", flush=True)
"""
import threading
import queue
import time
from typing import Iterator, List, Optional

import mlx.core as mx
import mlx.nn as nn

from chronos.mlx.expert_store import MLXExpertStore
from chronos.mlx.moe import ChronosMLXMOE


class ChronosMLXInferenceEngine:
    """
    Streaming token-by-token inference with async expert prefetch.

    On Apple Silicon the prefetch thread reads .npy files from SSD and
    calls mx.array() to bring weights into unified memory.  The main
    thread calls mx.eval() (promote_to_vram) before the next forward
    pass — Metal overlaps the IO with prior computation automatically.
    """

    def __init__(self, model, config, ssd_dir: str = "./expert_cache_mlx"):
        # Inference only: training mode would randomly mask experts
        # (fallback_mask_prob) and use the training ACT schedule.
        model.eval()
        self.model = model
        self.config = config
        self.store = MLXExpertStore(model, config, ssd_dir=ssd_dir)
        self._prefetch_q: queue.Queue = queue.Queue(maxsize=config.prefetch_depth * 2)
        self._stop = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True, name="mlx-prefetch"
        )
        self._prefetch_thread.start()
        self.last_stats: dict = {}
        self._last_predicted: set[int] = set()
        self._stats_lock = threading.Lock()
        self._runtime_stats = self._new_runtime_stats()

    @staticmethod
    def _new_runtime_stats(all_resident_fast_path: bool = False) -> dict:
        return {
            "resident_hits": 0,
            "resident_misses": 0,
            "resident_vram_hits": 0,
            "resident_ram_hits": 0,
            "selection_hits": 0,
            "selection_misses": 0,
            "prediction_hits": 0,
            "prediction_total": 0,
            "prefetch_queue_drops": 0,
            "prefetch_wait_time_s": 0.0,
            "sync_ssd_loads": 0,
            "on_demand_load_time_s": 0.0,
            "cold_prediction_hits": 0,
            "cold_prediction_total": 0,
            "all_resident_fast_path": bool(all_resident_fast_path),
        }

    def setup(self, warm_expert_ids: Optional[List[int]] = None):
        self.store.offload_all_to_ssd()
        self.store.replace_live_experts_with_placeholders()
        self.store.warm_up(warm_expert_ids)

    def setup_with_clusters(self, clusters: List[List[int]], warm_expert_ids: Optional[List[int]] = None):
        self.store.offload_all_to_ssd(clusters=clusters)
        self.store.replace_live_experts_with_placeholders()
        self.store.warm_up(warm_expert_ids)

    def _install_runtime_hooks(self):
        def loader(eid: int) -> bool:
            eid = int(eid)
            was_hot = eid in self.store.hot_expert_ids()
            with self.store._lock:
                was_warm = (
                    eid in self.store._warm
                    and self.store._layer_states_complete(self.store._warm.get(eid))
                )
            with self._stats_lock:
                if was_hot or was_warm:
                    self._runtime_stats["selection_hits"] += 1
                else:
                    self._runtime_stats["selection_misses"] += 1
            t0 = time.monotonic()
            if not was_hot and not was_warm:
                with self._stats_lock:
                    self._runtime_stats["sync_ssd_loads"] += 1
                    self._runtime_stats["cold_prediction_total"] += 1
                    if eid in self._last_predicted:
                        self._runtime_stats["cold_prediction_hits"] += 1
            ok = self.store.promote_to_vram(eid)
            elapsed = time.monotonic() - t0
            with self._stats_lock:
                self._runtime_stats["on_demand_load_time_s"] += elapsed
                if ok:
                    self._runtime_stats["resident_hits" if (was_hot or was_warm) else "resident_misses"] += 1
                    if was_hot:
                        self._runtime_stats["resident_vram_hits"] += 1
                    elif was_warm:
                        self._runtime_stats["resident_ram_hits"] += 1
                else:
                    self._runtime_stats["resident_misses"] += 1
                self._runtime_stats["prediction_total"] += 1
                if (was_hot or was_warm) and eid in self._last_predicted:
                    self._runtime_stats["prediction_hits"] += 1
            return ok

        def touch(eid: int) -> None:
            eid = int(eid)
            with self.store._lock:
                if eid in self.store._hot_lru:
                    if self.store._expert_live_all_layers(eid):
                        self.store._hot_lru.move_to_end(eid)
                        with self._stats_lock:
                            self._runtime_stats["selection_hits"] += 1
                            self._runtime_stats["resident_vram_hits"] += 1
                            self._runtime_stats["prediction_total"] += 1
                            if eid in self._last_predicted:
                                self._runtime_stats["prediction_hits"] += 1
                    else:
                        self.store._hot_lru.pop(eid, None)

        for layer in self.model.layers:
            moe = getattr(layer, "mlp", None)
            if isinstance(moe, ChronosMLXMOE):
                moe.runtime_miss_policy = "sync_on_demand"
                moe.runtime_on_demand_loader = loader
                moe.runtime_touch_expert = touch

    def _attach_planner(self, planner) -> None:
        """Report every lazy-expert run to ``planner`` and let it rank evictions.

        Wraps the hooks of :meth:`_install_runtime_hooks`: before an expert
        of dispatch ``(layer, loop iteration)`` is loaded or run, the planner
        records it; the dispatch's not-yet-run experts (an MoE runs its
        selection in id order) are kept from eviction.  The store then evicts
        the page needed latest instead of the least recent one.
        """
        layer_of = {}
        for li, layer in enumerate(self.model.layers):
            moe = getattr(layer, "mlp", None)
            if not isinstance(moe, ChronosMLXMOE):
                continue
            layer_of[id(moe)] = li

            def record(eid, moe=moe, li=li):
                eid = int(eid)
                selected = getattr(moe, "_selected_host", None) or ()
                pending = [j for j in range(eid + 1, len(selected)) if selected[j]]
                planner.access(li, len(moe.iteration_router_probs) - 1, eid, pending)

            base_load, base_touch = moe.runtime_on_demand_loader, moe.runtime_touch_expert

            def load(eid, record=record, base=base_load):
                record(eid)
                return bool(base(eid)) if base is not None else False

            def touch(eid, record=record, base=base_touch):
                record(eid)
                if base is not None:
                    base(eid)

            moe.runtime_on_demand_loader = load
            moe.runtime_touch_expert = touch
        # Ranking replaces the protected window: a protected set the size of
        # the budget would turn every eviction into a forced fallback.
        self.store.set_protected_experts([])
        self.store.set_eviction_ranker(planner.victim)

    def _planned_prefetch(self, planner, timeout_s: float) -> int:
        """Queue (and briefly await) the pages the next forward needs early; returns the count."""
        hot_cap = max(int(getattr(self.store, "_capacity", 1) or 1), 1)
        warm_cap = max(int(getattr(self.store, "_warm_capacity", hot_cap) or hot_cap), 1)
        with self.store._lock:
            warm = set(self.store._warm.keys())
        hot = self.store.hot_expert_ids()
        fetch = planner.prefetch_plan(warm | hot, warm_cap)
        promote = planner.prefetch_plan(hot, hot_cap)
        if fetch or promote:
            self._prefetch_and_promote_window(list(dict.fromkeys(promote + fetch)), promote_ids=promote,
                                              timeout_s=timeout_s)
        return len(promote)

    def _clear_runtime_hooks(self):
        for layer in self.model.layers:
            moe = getattr(layer, "mlp", None)
            if isinstance(moe, ChronosMLXMOE):
                moe.runtime_miss_policy = "sync_on_demand"
                moe.runtime_on_demand_loader = None
                moe.runtime_touch_expert = None

    # ── Prefetch background thread ────────────────────────────────

    def _prefetch_loop(self):
        while not self._stop.is_set():
            try:
                expert_ids: List[int] = self._prefetch_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self.store.prefetch_to_ram(expert_ids)
            self._prefetch_q.task_done()

    def _schedule_prefetch(self, expert_ids: List[int]):
        if not expert_ids:
            return
        try:
            self._prefetch_q.put_nowait(expert_ids)
        except queue.Full:
            with self._stats_lock:
                self._runtime_stats["prefetch_queue_drops"] += 1

    def _prefetch_and_promote_window(
        self,
        prefetch_ids: List[int],
        promote_ids: Optional[List[int]] = None,
        timeout_s: float = 0.012,
    ) -> None:
        if not prefetch_ids or self.store.storage_format == "full_dram":
            return
        promote_ids = list(dict.fromkeys(int(eid) for eid in (promote_ids or prefetch_ids)))
        pending = []
        for eid in dict.fromkeys(int(eid) for eid in prefetch_ids):
            with self.store._lock:
                if eid in self.store._hot_lru:
                    continue
                if eid in self.store._warm and self.store._layer_states_complete(self.store._warm.get(eid)):
                    continue
            pending.append(eid)
        if pending:
            self._schedule_prefetch(pending)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            self._promote_ready(promote_ids)
            with self.store._lock:
                ready = all(
                    eid in self.store._hot_lru
                    or (eid in self.store._warm and self.store._layer_states_complete(self.store._warm.get(eid)))
                    for eid in promote_ids
                )
            if ready:
                break
            time.sleep(0.001)
        self._promote_ready(promote_ids)
        with self._stats_lock:
            self._runtime_stats["prefetch_wait_time_s"] += max(0.0, time.monotonic() - (deadline - max(0.0, float(timeout_s))))

    def _all_resident_fast_path_enabled(self) -> bool:
        if self.store.storage_format == "full_dram":
            return True
        num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        if num_experts <= 0:
            return False
        if int(getattr(self.store, "_capacity", 0) or 0) < num_experts:
            return False
        if int(getattr(self.store, "_warm_capacity", 0) or 0) < num_experts:
            return False
        return len(self.store.hot_expert_ids()) >= num_experts

    def _current_virtual_ids(self) -> List[int]:
        """Last-token top-k ids over every MoE dispatch of the last forward.

        Recurrent groups dispatch once per loop iteration (virtual layer)
        and may pick different experts each time; residency follows the
        union, so the next step's identical iterations stay hot.
        """
        values = self._virtual_route_weights(last_only=True)
        return [i for i, _v in sorted(((i, v) for i, v in enumerate(values) if v > 0), key=lambda t: (-t[1], t[0]))]

    def _virtual_route_weights(self, last_only: bool = True) -> List[float]:
        """Per-expert top-k routing weight summed over every dispatch of the last forward.

        ``last_only`` scores the last position (one decode or draft token);
        otherwise every position counts and the sum is divided by the number
        of positions, so a verify or prefill forward weighs like one token.
        """
        from chronos.mlx.moe import ChronosMLXMOE
        e = int(self.config.num_experts)
        k = max(1, min(int(self.config.num_experts_per_tok), e))
        rows, positions = [], 1
        for layer in self.model.layers:
            moe = getattr(layer, "mlp", None)
            if isinstance(moe, ChronosMLXMOE):
                for p in (moe.iteration_router_probs or ()):
                    if not last_only:
                        positions = max(positions, int(p.shape[1]))
                    rows.append((p[:, -1, :] if last_only else p).reshape(-1, e))
        if not rows:
            return [0.0] * e
        probs = mx.concatenate(rows, axis=0).astype(mx.float32)
        ids = mx.argpartition(-probs, kth=k - 1, axis=-1)[:, :k]
        weights = mx.take_along_axis(probs, ids, axis=-1)
        scores = ((ids[..., None] == mx.arange(e)).astype(mx.float32) * weights[..., None]).sum(axis=(0, 1))
        values = [float(v) / positions for v in scores.tolist()]
        with self._stats_lock:
            self._runtime_stats["virtual_dispatches"] = self._runtime_stats.get("virtual_dispatches", 0) + len(rows)
        return values

    def _protect_prediction_window(self, expert_ids: List[int]) -> None:
        capacity = max(
            1,
            int(getattr(self.store, "_warm_capacity", 1) or 1),
            int(getattr(self.store, "_capacity", 1) or 1),
        )
        hot_ids = sorted(self.store.hot_expert_ids())
        current = list(getattr(self, "_last_current", ()) or ())
        protected = list(dict.fromkeys([int(eid) for eid in expert_ids] + current + hot_ids))[:capacity]
        self.store.set_protected_experts(protected)

    def stop(self):
        self._stop.set()
        try:
            self._prefetch_q.put_nowait([])
        except queue.Full:
            pass
        if self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

    # ── Generation loop ───────────────────────────────────────────

    def generate(
        self,
        input_ids: mx.array,           # [1, S] (already tokenised)
        max_new_tokens: int = 128,
        temperature: float = 0.85,
        top_p: float = 0.9,
        scheduler=None,                # Optional[PrefillScheduler] — lazy import
        *,
        speculative: bool = False,     # recurrent-depth self-speculative decoding
        draft_depth: int = 1,
        gamma: int = 4,
        gamma_max: int = 8,
        adaptive_gamma: bool = True,
        top_k: int = 0,
        seed: Optional[int] = None,
        eos_token_id=None,
        residency: str = "window",     # lazy experts: "window" (LRU + protected window) | "planned"
        residency_decay: float = 0.7,
        decode_wait_s: float = 0.012,  # planned: wait for prefetched pages before a decode step
        verify_wait_s: float = 0.02,   # planned + speculative: wait before the verify forward
        repetition_penalty: float = 1.0,
    ) -> Iterator[int]:
        """
        Yields token IDs one at a time.

        If scheduler is provided (PrefillScheduler), runs prefill-time batch
        loading before the generation loop (dual-layer mode).  Otherwise falls
        back to per-token LookaheadRouter-driven prefetch.

        ``speculative=True`` drafts ``gamma`` tokens at ``draft_depth`` and
        verifies them in one full-depth forward (``chronos.mlx.speculative``);
        greedy output equals plain greedy decoding.  ``eos_token_id`` (int or
        list) stops generation after that token is yielded.

        ``residency="planned"`` replaces LRU eviction of lazy expert pages
        with next-use planning (``chronos.residency_planner``): the previous
        forward of the same kind (and, when speculative, the draft queue)
        predicts the order in which pages are needed, the page needed latest
        is evicted, and pages needed early are prefetched.  Output is
        unchanged; residency only decides what is loaded when.
        ``decode_wait_s`` / ``verify_wait_s`` bound how long a step waits for
        its prefetched pages before running (a page still missing is loaded
        on demand, so the wait only trades latency for fewer stalls).

        ``repetition_penalty`` (plain decoding only) applies the CTRL rule to
        every token already in the prompt or output; speculative decoding
        needs 1.0 because the penalty changes the target distribution.
        """
        if residency not in ("window", "planned"):
            raise ValueError("residency must be 'window' or 'planned'")
        if not 0.0 <= float(residency_decay) < 1.0:
            raise ValueError("residency_decay must be in [0, 1)")
        if float(decode_wait_s) < 0 or float(verify_wait_s) < 0:
            raise ValueError("decode_wait_s and verify_wait_s must be >= 0")
        if float(repetition_penalty) <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if speculative and float(repetition_penalty) != 1.0:
            raise ValueError("speculative decoding needs repetition_penalty=1.0")
        self.store.set_eviction_ranker(None)
        if speculative:
            yield from self._generate_speculative(
                input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                draft_depth=draft_depth, gamma=gamma, gamma_max=gamma_max, adaptive_gamma=adaptive_gamma,
                seed=seed, eos_token_id=eos_token_id, residency=residency, residency_decay=residency_decay,
                verify_wait_s=verify_wait_s)
            return
        penalty = float(repetition_penalty)
        seen = None
        if penalty != 1.0:
            import numpy as np
            seen = np.zeros(int(self.config.vocab_size), dtype=bool)
            seen[np.array(input_ids.tolist(), dtype=np.int64).reshape(-1)] = True

        def penalised(row):
            if seen is None:
                return row
            row = row.astype(mx.float32)
            return mx.where(mx.array(seen), mx.where(row < 0, row * penalty, row / penalty), row)
        t0 = time.monotonic()
        setup_mem = self._memory_snapshot()
        self._last_predicted = set()
        self._last_current = []
        with self._stats_lock:
            all_resident_fast_path = self._all_resident_fast_path_enabled()
            self._runtime_stats = self._new_runtime_stats(all_resident_fast_path)
        if all_resident_fast_path:
            self._clear_runtime_hooks()
            self.store.set_protected_experts([])
        else:
            self._install_runtime_hooks()
        planner = None
        if not all_resident_fast_path and scheduler is None and residency == "planned":
            from chronos.residency_planner import NextUsePlanner
            planner = NextUsePlanner(int(self.config.num_experts), decay=float(residency_decay))
            self._attach_planner(planner)
        hot_cap = int(getattr(self.store, "_capacity", 1) or 1)
        # ── Prefill phase: front-load expert IO ───────────────────────────
        if scheduler is not None and not all_resident_fast_path:
            import torch, numpy as np
            ids_np = np.array(input_ids.tolist(), dtype=np.int64)
            ids_pt = torch.from_numpy(ids_np)
            scheduler.prepare(ids_pt, device="cpu")
            scheduler.wait()

        # Prefill forward
        prefill_t0 = time.monotonic()
        if all_resident_fast_path:
            prefill_masks = None
        else:
            prefill_masks = self._build_avail_masks(None)
        if planner is not None:
            planner.begin_forward("multi")
        logits, lookahead_probs, cache = self.model(input_ids, available_expert_masks=prefill_masks)
        mx.eval(logits)
        if planner is not None:
            planner.end_forward()
        self._last_current = [] if all_resident_fast_path else self._current_virtual_ids()
        prefill_t1 = time.monotonic()
        prefill_mem = self._memory_snapshot()

        # Sample from prefill
        next_token = self._sample(penalised(logits[:, -1, :]), temperature, top_p)
        activated_ids: List[int] = []
        tokens = 1
        if planner is not None:
            if lookahead_probs is not None:
                planner.hint(self._predict_future_experts(lookahead_probs, capacity=hot_cap, max_steps=1))
        elif not all_resident_fast_path and scheduler is None and lookahead_probs is not None:
            future_ids = self._predict_future_experts(
                lookahead_probs,
                capacity=int(getattr(self.store, "_warm_capacity", self.config.num_experts) or self.config.num_experts),
            )
            immediate_ids = self._predict_future_experts(
                lookahead_probs,
                capacity=int(getattr(self.store, "_capacity", 1) or 1),
                max_steps=1,
            )
            self._last_predicted = set(int(eid) for eid in future_ids)
            self._protect_prediction_window(future_ids)
            self._prefetch_and_promote_window(future_ids, promote_ids=immediate_ids, timeout_s=0.025)
        eos = set([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
        first_id = int(next_token.item())
        if seen is not None:
            seen[first_id] = True
        yield first_id

        for _ in range(0 if first_id in eos else max_new_tokens - 1):
            if all_resident_fast_path:
                avail_masks = None
            elif scheduler is not None:
                avail_masks = self._build_avail_masks(next_token)
            elif planner is not None:
                planner.begin_forward("single")
                self._planned_prefetch(planner, float(decode_wait_s))
                self._last_predicted = set(planner.upcoming())
                avail_masks = self._build_avail_masks(next_token)
            else:
                # LookaheadRouter-driven prefetch
                if lookahead_probs is not None:
                    future_ids = self._predict_future_experts(
                        lookahead_probs,
                        capacity=int(getattr(self.store, "_warm_capacity", self.config.num_experts) or self.config.num_experts),
                    )
                    immediate_ids = self._predict_future_experts(
                        lookahead_probs,
                        capacity=int(getattr(self.store, "_capacity", 1) or 1),
                        max_steps=1,
                    )
                    self._last_predicted = set(int(eid) for eid in future_ids)
                    self._protect_prediction_window(future_ids)
                    self._prefetch_and_promote_window(future_ids, promote_ids=immediate_ids)
                avail_masks = self._build_avail_masks(next_token)

            token_in = next_token.reshape(1, 1)
            logits, lookahead_probs, cache = self.model(
                token_in,
                cache=cache,
                available_expert_masks=avail_masks,
            )
            mx.eval(logits)
            if planner is not None:
                planner.end_forward()
                if lookahead_probs is not None:
                    planner.hint(self._predict_future_experts(lookahead_probs, capacity=hot_cap, max_steps=1))
            if not all_resident_fast_path:
                self._last_current = self._current_virtual_ids()

            next_token = self._sample(penalised(logits[:, -1, :]), temperature, top_p)
            mx.eval(next_token)
            tok_id = int(next_token.item())
            if seen is not None:
                seen[tok_id] = True
            activated_ids.append(tok_id)
            tokens += 1
            yield tok_id
            if tok_id in eos:
                break

        # Update frequency heuristic after generation
        if scheduler is not None:
            scheduler.record_activation(list(set(activated_ids)))
        end = time.monotonic()
        end_mem = self._memory_snapshot()
        self.last_stats = {
            "tokens_generated": tokens,
            "elapsed_s": round(end - t0, 3),
            "tokens_per_sec": round(tokens / max(end - t0, 1e-6), 2),
            "prefill_time_s": round(prefill_t1 - prefill_t0, 3),
            "decode_time_s": round(max(0.0, end - prefill_t1), 3),
            "residency": {"mode": "full" if all_resident_fast_path else (
                              "planned" if planner is not None else "window"),
                          **({"planner": dict(planner.stats)} if planner is not None else {})},
            **self._memory_fields(setup_mem, "after_setup"),
            **self._memory_fields(prefill_mem, "after_prefill"),
            **self._memory_fields(end_mem, "after_decode"),
            **self._runtime_stat_fields(),
            **self.store.stats(),
        }
        self.store.set_eviction_ranker(None)

    def _generate_speculative(self, input_ids, *, max_new_tokens, temperature, top_p, top_k, draft_depth, gamma,
                              gamma_max, adaptive_gamma, seed, eos_token_id, residency="window",
                              residency_decay=0.7, verify_wait_s=0.02) -> Iterator[int]:
        """Self-speculative decoding with the same residency work as plain decoding.

        Every forward (draft or verify) runs with the current availability
        masks.  After it, the experts its last position routed to (all virtual
        layers) are unioned with the LookaheadRouter prediction and prefetched,
        so experts a draft touches are being loaded before the verify forward
        that re-reads them.

        ``residency="planned"`` hands eviction to a next-use planner
        (``chronos.residency_planner.NextUsePlanner``).  Each draft token ran
        the prelude and the first ``draft_depth`` iterations, and verify
        re-runs exactly those dispatches for the same tokens plus the deeper
        ones, so the draft queue is the verify forward's page-access
        reference: pages the queue needs early are prefetched while the
        drafts run and awaited right before verify (``verify_wait_s``), and
        the page needed latest (or not at all) is evicted first; the
        previous forward of the same kind covers the deeper iterations.
        Output is unchanged: residency only decides what is loaded when."""
        from chronos.mlx.speculative import SpecStats, speculative_tokens
        t0 = time.monotonic()
        setup_mem = self._memory_snapshot()
        self._last_predicted, self._last_current = set(), []
        with self._stats_lock:
            fast = self._all_resident_fast_path_enabled()
            self._runtime_stats = self._new_runtime_stats(fast)
        if fast:
            self._clear_runtime_hooks()
            self.store.set_protected_experts([])
        else:
            self._install_runtime_hooks()
        warm_cap = int(getattr(self.store, "_warm_capacity", self.config.num_experts) or self.config.num_experts)
        hot_cap = int(getattr(self.store, "_capacity", 1) or 1)
        timing = {"first": None}

        planner = None
        if not fast and residency == "planned":
            from chronos.residency_planner import NextUsePlanner
            planner = NextUsePlanner(int(self.config.num_experts), decay=float(residency_decay))
            self._attach_planner(planner)
        queue_stats = {"verify_plans": 0, "verify_planned": 0, "verify_ready": 0}

        def forward_fn(ids, cache, depth):
            verify = depth is None and cache is not None
            if planner is not None:
                kind = "draft" if depth is not None else ("multi" if int(ids.shape[1]) > 1 else "single")
                planner.begin_forward(kind)
                # Drafts only queue SSD reads (the next draft is not delayed);
                # verify waits briefly for the draft queue's pages.
                self._planned_prefetch(planner, 0.0 if depth is not None else float(verify_wait_s))
                if verify:
                    hot = self.store.hot_expert_ids()
                    upcoming = planner.upcoming(hot_cap)
                    queue_stats["verify_plans"] += 1
                    queue_stats["verify_planned"] += len(upcoming)
                    queue_stats["verify_ready"] += sum(1 for eid in upcoming if eid in hot)
            masks = None if fast else self._build_avail_masks(None)
            logits, lookahead, new_cache = self.model(ids, cache=cache, available_expert_masks=masks,
                                                      recurrent_depth=depth)
            mx.eval(logits)
            if planner is not None:
                planner.end_forward(verify=verify)
                if lookahead is not None:
                    planner.hint(self._predict_future_experts(lookahead, capacity=hot_cap, max_steps=1))
                self._last_current = self._current_virtual_ids()
                self._last_predicted = set(planner.upcoming())
            elif not fast:
                self._last_current = self._current_virtual_ids()
                predicted = (self._predict_future_experts(lookahead, capacity=warm_cap)
                             if lookahead is not None else [])
                window = list(dict.fromkeys(list(self._last_current) + predicted))[:max(warm_cap, 1)]
                self._last_predicted = set(window)
                self._protect_prediction_window(window)
                self._prefetch_and_promote_window(window, promote_ids=window[:max(hot_cap, 1)])
            return logits, lookahead, new_cache

        stats = SpecStats()
        from chronos.config_schema import resolve_recurrent_attention_mode
        cfg = self.model.config
        full_depth = int(getattr(cfg, "max_inference_depth", None) or cfg.recurrent_depth)
        recurrent = bool(getattr(cfg, "recurrent_enabled", False)) and int(cfg.recurrent_depth) > 1
        eff_draft = max(1, min(int(draft_depth), full_depth))
        if not recurrent or resolve_recurrent_attention_mode(cfg) == "per_iteration":
            eff_draft = full_depth                     # mirrors speculative_tokens' fallback
        do_sample = float(temperature) > 0
        try:
            for tok in speculative_tokens(
                    self.model, input_ids, stats, max_new_tokens=int(max_new_tokens), draft_depth=draft_depth,
                    gamma=gamma, adaptive_gamma=adaptive_gamma, gamma_max=gamma_max, do_sample=do_sample,
                    temperature=max(float(temperature), 1e-6), top_k=int(top_k or 0),
                    top_p=float(top_p) if do_sample else 1.0, eos_token_id=eos_token_id, seed=seed,
                    forward_fn=forward_fn):
                if timing["first"] is None:
                    timing["first"] = time.monotonic()
                yield int(tok)
        finally:
            end = time.monotonic()
            first = timing["first"] or end
            end_mem = self._memory_snapshot()
            self.last_stats = {
                "tokens_generated": stats.generated,
                "elapsed_s": round(end - t0, 3),
                "tokens_per_sec": round(stats.generated / max(end - t0, 1e-6), 2),
                "prefill_time_s": round(first - t0, 3),
                "decode_time_s": round(max(0.0, end - first), 3),
                "speculative": stats.as_dict() | {
                    "draft_depth": eff_draft, "full_depth": full_depth, "enabled": eff_draft < full_depth,
                    "gamma_final": (stats.gamma_history or [int(gamma)])[-1]},
                "residency": {"mode": "full" if fast else residency, **queue_stats,
                              "verify_ready_rate": round(queue_stats["verify_ready"]
                                                         / max(queue_stats["verify_planned"], 1), 4),
                              **({"planner": dict(planner.stats)} if planner is not None else {})},
                **self._memory_fields(setup_mem, "after_setup"),
                **self._memory_fields(end_mem, "after_decode"),
                **self._runtime_stat_fields(),
                **self.store.stats(),
            }
            self.store.set_eviction_ranker(None)

    def _predict_future_experts(
        self,
        lookahead_probs: mx.array,
        capacity: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> List[int]:
        """Extract bounded top-k expert IDs for future steps."""
        e = int(self.config.num_experts)
        budget = e if capacity is None else max(1, min(e, int(capacity)))
        values = self._future_expert_scores(lookahead_probs, max_steps=max_steps)
        scored = [(idx, value) for idx, value in enumerate(values) if value > 0.0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [int(idx) for idx, _value in scored[:budget]]

    def _future_expert_scores(self, lookahead_probs: mx.array, max_steps: Optional[int] = None) -> List[float]:
        """Per-expert LookaheadRouter score: top-k weight of each future step, weighted 1/(offset+1)."""
        # This store pages an expert ID across physical layers. Preserve
        # each prediction head until admission, then rank the union of IDs.
        e = int(self.config.num_experts)
        future = lookahead_probs[:, -1, 1:, ...]
        if max_steps is not None:
            future = future[:, :max(0, int(max_steps))]
        if future.shape[1] <= 0:
            return [0.0] * e
        top_k = max(1, min(int(self.config.num_experts_per_tok), e))
        scores = mx.zeros((e,), dtype=mx.float32)
        for offset in range(future.shape[1]):
            step = future[:, offset, ...].reshape(-1, e).astype(mx.float32)
            ids = mx.argpartition(-step, kth=top_k-1, axis=-1)[..., :top_k]
            vals = mx.take_along_axis(step, ids, axis=-1) / float(offset + 1)
            one_hot = (ids[..., None] == mx.arange(e)).astype(mx.float32)
            scores = scores + (one_hot*vals[..., None]).sum(axis=(0, 1))
        mx.eval(scores)
        return [float(v) for v in scores.tolist()]

    def _build_avail_masks(self, _token) -> List[set[int]] | None:
        """Promote prefetched experts and return per-layer availability masks."""
        n_layers = len(self.model.layers)
        if self.store.storage_format == "full_dram":
            return None
        mask = self.store.hot_expert_ids()
        return [mask] * n_layers

    def _promote_ready(self, expert_ids: List[int]) -> None:
        for eid in expert_ids:
            eid = int(eid)
            with self.store._lock:
                ready = (
                    eid in self.store._warm
                    and self.store._layer_states_complete(self.store._warm.get(eid))
                ) or eid in self.store._hot_lru
            if ready:
                self.store.promote_to_vram(eid)

    def _runtime_stat_fields(self) -> dict:
        with self._stats_lock:
            stats = dict(self._runtime_stats)
        total = int(stats.get("selection_hits", 0)) + int(stats.get("selection_misses", 0))
        pred_total = int(stats.get("prediction_total", 0))
        cold_pred_total = int(stats.get("cold_prediction_total", 0))
        all_resident_fast_path = bool(stats.get("all_resident_fast_path", False))
        resident_hit_rate = 1.0 if all_resident_fast_path else (
            float(stats.get("selection_hits", 0)) / max(total, 1)
        )
        return {
            "resident_hit_rate": round(resident_hit_rate, 4),
            "cache_hit_rate": round(resident_hit_rate, 4),
            "cache_hits": int(stats.get("selection_hits", 0)),
            "cache_misses": int(stats.get("selection_misses", 0)),
            "expert_selection_hits": int(stats.get("selection_hits", 0)),
            "expert_selection_misses": int(stats.get("selection_misses", 0)),
            "prediction_hit_rate": round(float(stats.get("prediction_hits", 0)) / max(pred_total, 1), 4),
            "prediction_hits": int(stats.get("prediction_hits", 0)),
            "prediction_total": pred_total,
            "cold_miss_predict_hit_rate": round(float(stats.get("cold_prediction_hits", 0)) / max(cold_pred_total, 1), 4),
            "cold_prediction_hits": int(stats.get("cold_prediction_hits", 0)),
            "cold_prediction_total": cold_pred_total,
            "all_resident_fast_path": all_resident_fast_path,
            "resident_vram_hits": int(stats.get("resident_vram_hits", 0)),
            "resident_ram_hits": int(stats.get("resident_ram_hits", 0)),
            "prefetch_queue_drops": int(stats.get("prefetch_queue_drops", 0)),
            "prefetch_wait_time_s": round(float(stats.get("prefetch_wait_time_s", 0.0)), 4),
            "sync_ssd_loads": int(stats.get("sync_ssd_loads", 0)),
            "on_demand_loads": int(stats.get("resident_misses", 0)),
            "on_demand_load_time_s": round(float(stats.get("on_demand_load_time_s", 0.0)), 4),
            "fallback_weight_rate": 0.0,
            "virtual_dispatches": int(stats.get("virtual_dispatches", 0)),
        }

    @staticmethod
    def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        scaled = logits / temperature
        probs = mx.softmax(scaled, axis=-1)
        # top-p nucleus sampling
        sorted_idx = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        # mask tokens beyond top-p threshold
        cutoff = mx.array(top_p)
        nucleus = (cumsum - sorted_probs) < cutoff
        filtered = mx.where(nucleus, sorted_probs, mx.zeros_like(sorted_probs))
        filtered = filtered / (filtered.sum(axis=-1, keepdims=True) + 1e-9)
        sample_idx = mx.random.categorical(mx.log(filtered + 1e-9))
        return mx.take(sorted_idx[0], sample_idx)

    @staticmethod
    def _memory_snapshot() -> dict:
        try:
            from chronos.backend.mac_diagnostics import mlx_memory_snapshot, rss_snapshot

            out = dict(mlx_memory_snapshot())
            out.update(rss_snapshot())
            return out
        except Exception:
            return {}

    @staticmethod
    def _memory_fields(snapshot: dict, suffix: str) -> dict:
        out = {}
        for key in ("rss_gb", "mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb"):
            if key in snapshot:
                prefix = key[:-3] if key.endswith("_gb") else key
                out[f"{prefix}_{suffix}_gb"] = snapshot[key]
        return out
