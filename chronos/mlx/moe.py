"""
chronos/mlx/moe.py — MLX-native ChronosMOEFeedForward.

Soft gating is expressed as weighted sum (compile-safe):
    out = avail[i] * expert_out + (1 - avail[i]) * shared_out

No Python branches on mask values, so a compiled training step
(chronos/mlx/training/compiled.py) keeps one graph per signature.
"""
import mlx.core as mx
import mlx.nn as nn

from chronos.mlx.checkpointing import draw, recomputing, recorded, side_output_sink
from chronos.mlx.moe_kernels import (
    gather_mm_supported, grouped_swiglu, sparse_dispatch_ok, stack_expert_weights, training_availability,
)


def _sparse_ok(dtype) -> bool:
    """Sorted gather_mm dispatch is used only where it passed the one-time
    self-check for this device/dtype (CPU: FP32 only).  Otherwise the exact
    masked-dense path runs: slower, never wrong."""
    return sparse_dispatch_ok(dtype)


class FeedForwardMLX(nn.Module):
    """SwiGLU FFN matching ``chronos.model.kernel.FeedForward``."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LazyFeedForwardMLX(nn.Module):
    """Placeholder for an expert whose weights are not resident in MLX memory."""

    def __call__(self, _x: mx.array) -> mx.array:
        raise RuntimeError("MLX lazy expert placeholder executed before materialization")


class ChronosMLXMOE(nn.Module):
    """
    MLX-native Mixture-of-Experts with:
    - LookaheadRouter output stored as last_router_probs
    - Compile-safe soft gating (avail mask → float multiply)
    - Shared expert fallback at zero I/O cost (unified memory)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        H = config.hidden_size
        I = config.moe_intermediate_size

        self.gate = nn.Linear(H, config.num_experts, bias=False)
        self.experts = [FeedForwardMLX(H, I) for _ in range(config.num_experts)]
        self.shared_experts = [
            FeedForwardMLX(H, I) for _ in range(config.num_shared_experts)
        ]
        object.__setattr__(self, "_last_router_probs", None)  # [B, S, E]
        # One entry per call: a recurrent block dispatches once per iteration.
        object.__setattr__(self, "_iteration_router_probs", [])
        # router_scope="shared": later iterations reuse iteration zero's ids.
        object.__setattr__(self, "router_scope", str(getattr(config, "router_scope", "per_iteration") or "per_iteration"))
        object.__setattr__(self, "_shared_topk_idx", None)
        # Per-forward cache of the stacked [E, out, in] expert weights: a
        # recurrent block runs this MoE once per iteration, but its weights
        # only change between optimizer updates.  Opened/closed by the model
        # forward; outside a scope every call stacks fresh (always correct).
        object.__setattr__(self, "_pack_scope", None)
        self.runtime_miss_policy = "sync_on_demand"
        self.fallback_mask_prob = float(getattr(config, "fallback_mask_prob", 0.0) or 0.0)
        # Sorted gather_mm dispatch (requires mlx >= 0.32.4 on M5-class GPUs).
        self.sparse_dispatch = bool(getattr(config, "moe_sparse_dispatch", True)) and gather_mm_supported()
        self.chunk_tokens = int(getattr(config, "moe_chunk_tokens", 0) or 0)
        self.runtime_on_demand_loader = None
        self.runtime_touch_expert = None

    @property
    def last_router_probs(self):
        return self._last_router_probs

    @property
    def iteration_router_probs(self):
        return self._iteration_router_probs

    def accept_side_output(self, kind, value):
        """Record one dispatch's router probabilities ``[B, S, E]``."""
        if kind != "probs":
            raise ValueError(f"unknown MoE side output {kind!r}")
        object.__setattr__(self, "_last_router_probs", value)
        self._iteration_router_probs.append(value)

    def reset_iteration_routing(self):
        object.__setattr__(self, "_iteration_router_probs", [])
        object.__setattr__(self, "_shared_topk_idx", None)

    def open_pack_scope(self):
        object.__setattr__(self, "_pack_scope", {})

    def close_pack_scope(self):
        object.__setattr__(self, "_pack_scope", None)

    def _expert_stacks(self):
        scope = self._pack_scope
        if scope is None or recomputing():
            # A recomputation stacks its own primals, so the expert-weight
            # gradients reach the region's inputs.
            return stack_expert_weights(self.experts)
        # Keyed on the live weight arrays, so an expert swapped in mid-forward
        # (lazy promotion) can never be served from a stale stack.
        key = tuple(id(getattr(e, p).weight) for e in self.experts for p in ("gate_proj", "up_proj", "down_proj"))
        hit = scope.get("w")
        if hit is None or hit[0] != key:
            hit = (key, stack_expert_weights(self.experts))
            scope["w"] = hit
        return hit[1]

    @last_router_probs.setter
    def last_router_probs(self, value):
        object.__setattr__(self, "_last_router_probs", value)

    def _grouped_chunked(self, x_flat, topk_idx, topk_weight, fallback):
        """Sparse dispatch over the routed experts, in token chunks when
        ``config.moe_chunk_tokens`` is set and the call is a training forward.

        Each chunk runs under ``mx.checkpoint``: its ``[chunk*K, I]`` gate/up
        activations are recomputed in the backward pass instead of kept for the
        whole row.  Routing, weights and the fallback mask are inputs, so every
        token sees the same experts and the result equals one dispatch (up to
        summation order inside gather_mm)."""
        stacks = self._expert_stacks()
        n = x_flat.shape[0]
        chunk = self.chunk_tokens
        if not self.training or chunk <= 0 or n <= chunk:
            return grouped_swiglu(x_flat, topk_idx, topk_weight, *stacks, fallback=fallback)
        avail, shared = (None, None) if fallback is None else fallback
        outs = []
        prev = None
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            xs = x_flat[start:end]
            if prev is not None:
                xs = mx.depends(xs, prev)   # one chunk's activations live at a time
            if fallback is None:
                def part(xc, ic, wc, g, u, d):
                    return grouped_swiglu(xc, ic, wc, g, u, d)
                prev = mx.checkpoint(part)(xs, topk_idx[start:end], topk_weight[start:end], *stacks)
            else:
                def part(xc, ic, wc, sc, g, u, d):
                    return grouped_swiglu(xc, ic, wc, g, u, d, fallback=(avail, sc))
                prev = mx.checkpoint(part)(xs, topk_idx[start:end], topk_weight[start:end], shared[start:end], *stacks)
            outs.append(prev)
        return mx.concatenate(outs, axis=0)

    def _shared_out(self, x: mx.array) -> mx.array:
        return sum(e(x) for e in self.shared_experts) / len(self.shared_experts)

    def _resolve_live(self, i: int, python_avail: set) -> bool:
        """Materialize selected expert ``i`` under the miss policy; True = real weights.

        Same decisions as the per-expert loop: a miss under ``sync_on_demand``
        loads synchronously (or raises); other policies fall back to the shared expert."""
        policy = getattr(self, "runtime_miss_policy", "")
        loader = getattr(self, "runtime_on_demand_loader", None)
        sync = loader is not None and policy == "sync_on_demand"
        live = i in python_avail or (sync and bool(loader(int(i))))
        if live and isinstance(self.experts[i], LazyFeedForwardMLX):
            if sync:
                loader(int(i))
            if isinstance(self.experts[i], LazyFeedForwardMLX):
                if policy == "sync_on_demand":
                    raise RuntimeError(f"MLX lazy expert {i} was selected but not materialized")
                return False
        if not live:
            if policy == "sync_on_demand":
                raise RuntimeError(f"MLX lazy expert {i} could not be loaded on demand")
            return False
        touch = getattr(self, "runtime_touch_expert", None)
        if touch is not None:
            touch(int(i))
        return True

    def _lazy_grouped(self, x_flat, topk_idx, topk_weight, selected_host, python_avail, shared_out):
        """Lazy/offload MoE through ``grouped_swiglu`` over the live experts only.

        The full-memory path runs the same kernel, so a lazy decode reproduces
        the full decode instead of re-rounding through the per-expert loop
        (bf16 on the GPU: a different summation order flips near-tied greedy
        tokens).  Routed ids are remapped into a compact stack of the live
        experts; a miss maps to a spare slot whose output the shared expert
        replaces (the loop path's fallback)."""
        selected = [i for i, s in enumerate(selected_host) if s]
        live, experts = [], []
        for i in selected:
            if self._resolve_live(i, python_avail):
                # Take the module now: loading the next selected expert may
                # evict this one from the hot set (its weights stay valid).
                live.append(i)
                experts.append(self.experts[i])
        if not live:
            # Every routed slot missed: each is served by the shared output.
            return shared_out + shared_out * topk_weight.sum(axis=-1, keepdims=True).astype(shared_out.dtype)
        remap = [len(live)] * self.num_experts
        for j, i in enumerate(live):
            remap[i] = j
        fallback = None
        if len(live) < len(selected):
            experts.append(experts[0])                     # spare slot; its output is replaced
            fallback = (mx.arange(len(experts)) < len(live), shared_out)
        local = mx.array(remap, dtype=mx.int32)[topk_idx]
        routed = grouped_swiglu(x_flat, local, topk_weight, *stack_expert_weights(experts), fallback=fallback)
        return routed + shared_out

    def __call__(
        self,
        x: mx.array,
        available_expert_mask: mx.array = None,
    ) -> mx.array:
        """
        x: [B, S, H]
        available_expert_mask: [num_experts] float (1.0 = in GPU, 0.0 = miss)
                                or None for training (all available).
        """
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)                        # [N, H]

        scores = mx.softmax(self.gate(x_flat).astype(mx.float32), axis=-1)  # [N, E]
        probs = scores.reshape(B, S, self.num_experts)
        # Inside a recomputed block (``chronos.mlx.checkpointing``) side
        # outputs leave the region as region outputs; the loss then reads the
        # forward-pass arrays and never a recomputed copy.
        sink = side_output_sink()

        # top-k routing (router_scope="shared": later iterations reuse the
        # first dispatch's ids).  Indices are discrete: gradients flow through
        # topk_weight (gathered scores) only.  ``recorded`` makes a
        # recomputation follow the forward pass's routing exactly.
        shared_idx = recorded(lambda: self._shared_topk_idx if self.router_scope == "shared" else None)
        if shared_idx is not None and shared_idx.shape[0] == scores.shape[0]:
            topk_idx = shared_idx
        else:
            k = self.num_experts_per_tok
            topk_idx = recorded(lambda: mx.stop_gradient(mx.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]))
            if self.router_scope == "shared" and self._shared_topk_idx is None and not recomputing():
                object.__setattr__(self, "_shared_topk_idx", topk_idx)
        if sink is not None:
            sink.append((self, "probs", probs))
        else:
            self.accept_side_output("probs", probs)
        topk_weight = mx.take_along_axis(scores, topk_idx, axis=-1)   # [N, K]
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(axis=-1, keepdims=True) + 1e-20)

        y = mx.zeros_like(x_flat)

        if available_expert_mask is None and self.sparse_dispatch and _sparse_ok(x_flat.dtype) and not any(
            isinstance(e, LazyFeedForwardMLX) for e in self.experts
        ):
            # Sparse dispatch: each token runs only its top-k experts.
            shared = self._shared_out(x_flat) if len(self.shared_experts) > 0 else None
            fallback = None
            if self.training and self.fallback_mask_prob > 0.0 and shared is not None:
                # Cache-miss simulation (PyTorch parity): an unavailable routed
                # expert is served by the shared expert, as at offload inference.
                fallback = (draw(lambda: training_availability(self.num_experts, self.fallback_mask_prob)), shared)
            y = self._grouped_chunked(x_flat, topk_idx, topk_weight, fallback)
            if shared is not None:
                y = y + shared
        elif available_expert_mask is None:
            # Masked-dense reference path (old mlx or explicit opt-out).
            for i, expert in enumerate(self.experts):
                # mask over tokens where expert i is selected
                sel = (topk_idx == i)            # [N, K] bool
                tok_mask = sel.any(axis=-1)       # [N] bool
                w = mx.where(sel, topk_weight, 0.0).sum(axis=-1, keepdims=True)  # [N, 1]
                contrib = expert(x_flat) * w * tok_mask[:, None]
                y = y + contrib
            if len(self.shared_experts) > 0:
                y = y + self._shared_out(x_flat)
        else:
            # Inference: compile-safe soft gating
            python_avail = None
            if isinstance(available_expert_mask, (set, list, tuple)):
                python_avail = set(int(v) for v in available_expert_mask)
                avail = None
            else:
                avail = available_expert_mask.astype(x_flat.dtype)  # [E] float
            shared_out = self._shared_out(x_flat)               # [N, H]
            selected_host = None
            if python_avail is not None:
                # One host read for every expert's selection (was one
                # ``.item()`` sync per expert per MoE call during decode).
                hit = (topk_idx.reshape(-1)[:, None] == mx.arange(self.num_experts)[None, :]).any(axis=0)
                selected_host = hit.tolist()
                # Read by residency planners from the runtime hooks below.
                object.__setattr__(self, "_selected_host", selected_host)
                if self.sparse_dispatch and len(self.shared_experts) > 0 and _sparse_ok(x_flat.dtype):
                    y = self._lazy_grouped(x_flat, topk_idx, topk_weight, selected_host, python_avail, shared_out)
                    return y.reshape(B, S, H)

            for i, expert in enumerate(self.experts):
                sel = (topk_idx == i)
                tok_mask = sel.any(axis=-1)
                w = mx.where(sel, topk_weight, 0.0).sum(axis=-1, keepdims=True)

                if python_avail is not None:
                    selected = bool(selected_host[i])
                    if not selected:
                        continue
                    is_avail = i in python_avail
                    if selected and not is_avail:
                        loader = getattr(self, "runtime_on_demand_loader", None)
                        if loader is not None and getattr(self, "runtime_miss_policy", "") == "sync_on_demand":
                            is_avail = bool(loader(int(i)))
                    if is_avail:
                        if isinstance(self.experts[i], LazyFeedForwardMLX):
                            loader = getattr(self, "runtime_on_demand_loader", None)
                            if selected and loader is not None and getattr(self, "runtime_miss_policy", "") == "sync_on_demand":
                                is_avail = bool(loader(int(i)))
                        if isinstance(self.experts[i], LazyFeedForwardMLX):
                            if selected and getattr(self, "runtime_miss_policy", "") == "sync_on_demand":
                                raise RuntimeError(
                                    f"MLX lazy expert {i} was selected but not materialized"
                                )
                            blended = shared_out
                        else:
                            touch = getattr(self, "runtime_touch_expert", None)
                            if touch is not None:
                                touch(int(i))
                            expert_out = self.experts[i](x_flat)
                            blended = expert_out
                    else:
                        if selected and getattr(self, "runtime_miss_policy", "") == "sync_on_demand":
                            raise RuntimeError(
                                f"MLX lazy expert {i} could not be loaded on demand"
                            )
                        blended = shared_out
                else:
                    if isinstance(expert, LazyFeedForwardMLX):
                        raise RuntimeError(
                            "MLX lazy expert placeholder received a tensor mask. "
                            "Use Python set masks for lazy/offload inference so "
                            "selected experts can be materialized before execution."
                        )
                    expert_out  = expert(x_flat)
                    # avail[i]==1 → expert; avail[i]==0 → shared fallback
                    blended = avail[i] * expert_out + (1.0 - avail[i]) * shared_out
                contrib = blended * w * tok_mask[:, None]
                y = y + contrib
            if len(self.shared_experts) > 0:
                y = y + shared_out

        return y.reshape(B, S, H)
