"""
chronos/mlx/model.py — Full MLX Chronos model.

Architecture mirrors the PyTorch version (ChronosForCausalLM) but uses:
- mlx.nn.Module throughout
- MLAAttentionMLX (even layers) / SlidingWindowAttentionMLX (odd layers)
- ChronosMLXMOE for feed-forward
- LookaheadRouter as a plain Linear projection
- Training micro steps are compiled by chronos/mlx/training/compiled.py
  (one graph per depth/shape/config signature; Python decisions such as
  the executed depth are resolved outside the trace)

On Apple Silicon, mx.eval() materialises arrays into Metal shared memory.
Prefetch = eagerly eval expert weights during the prefetch window.
"""
import math
import mlx.core as mx
import mlx.nn as nn

from chronos.mlx.attention import MLAAttentionMLX, SlidingWindowAttentionMLX
from chronos.mlx.checkpointing import call_block
from chronos.mlx.moe import ChronosMLXMOE, FeedForwardMLX
from chronos.mlx.recurrent import RecurrentControllerMLX, auxiliary_state
from chronos.config_schema import resolve_recurrent_attention_mode, recurrent_cache_layout


class RMSNormMLX(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class LookaheadRouterMLX(nn.Module):
    """Predicts expert probabilities for t, t+1, ..., t+K steps."""

    def __init__(self, config):
        super().__init__()
        K = getattr(config, 'lookahead_steps', 2)
        self.lookahead_steps = K
        self.layer_aware = bool(getattr(config, "layer_aware_lookahead", False))
        self.num_groups = int(getattr(config, "recurrent_groups", 1) or 1)
        hidden = max(1, config.hidden_size // 4)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(
                hidden,
                config.num_experts * (K + 1) * (self.num_groups if self.layer_aware else 1),
                bias=False,
            ),
        )
        self.num_experts = config.num_experts

    def __call__(self, x: mx.array) -> mx.array:
        B, S, H = x.shape
        out = self.proj(x).astype(mx.float32)                          # [B, S, E*(K+1)]
        shape = (B, S, self.lookahead_steps + 1)
        if self.layer_aware:
            shape += (self.num_groups,)
        expected = (self.lookahead_steps + 1)*self.num_experts*(self.num_groups if self.layer_aware else 1)
        if out.shape[-1] != expected:
            raise ValueError(f"MLX lookahead projection width {out.shape[-1]} != config width {expected}; load the checkpoint config with its weights")
        return mx.softmax(out.reshape(*shape, self.num_experts), axis=-1)



def _recompute_block(cfg, index: int, block) -> bool:
    """Whether physical block ``index`` is an ``mx.checkpoint`` region under
    ``config.recompute_policy`` ("all" | "mla" | "swa" | "alternate")."""
    policy = getattr(cfg, "recompute_policy", "all")
    if policy == "all":
        return True
    if policy == "alternate":
        return index % 2 == 0
    is_mla = isinstance(block.self_attn, MLAAttentionMLX)
    return is_mla if policy == "mla" else not is_mla


class ChronosMLXBlock(nn.Module):
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.input_norm  = RMSNormMLX(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm   = RMSNormMLX(config.hidden_size, eps=config.rms_norm_eps)

        use_hybrid = getattr(config, 'use_hybrid_attention', True)
        if use_hybrid:
            if layer_id % 2 == 0:
                self.self_attn = MLAAttentionMLX(config)
            else:
                self.self_attn = SlidingWindowAttentionMLX(config)
        else:
            self.self_attn = SlidingWindowAttentionMLX(config)
            self.self_attn.window_size = None     # dense causal GQA (a 2**63 window overflowed int32)

        object.__setattr__(self, "_cfg", config)
        if getattr(config, 'use_moe', True):
            self.mlp = ChronosMLXMOE(config)
        else:
            self.mlp = FeedForwardMLX(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,
        available_expert_mask: mx.array = None,
        latent_only: bool = False,
        kv_override=None,
        return_shared_kv: bool = False,
        top_k: int | None = None,
    ):
        new_cache, shared = cache, None
        if not latent_only:
            out = self.self_attn(self.input_norm(x), mask=mask, cache=cache,
                                 kv_override=kv_override, return_shared_kv=return_shared_kv)
            if return_shared_kv:
                h, new_cache, shared = out
            else:
                h, new_cache = out
            x = x + h
        mlp_in = self.post_norm(x)
        # recompute_scope="mlp": only the FFN/MoE is a recomputed region.
        sub = (self.training and cache is None and getattr(self._cfg, "recompute_scope", "block") == "mlp"
            and bool(getattr(self._cfg, "activation_checkpointing", False)))
        if isinstance(self.mlp, ChronosMLXMOE):
            if sub:
                x = x + call_block(self.mlp, mlp_in, available_expert_mask=available_expert_mask, top_k=top_k)
            else:
                x = x + self.mlp(mlp_in, available_expert_mask=available_expert_mask, top_k=top_k)
        else:
            x = x + call_block(self.mlp, mlp_in) if sub else x + self.mlp(mlp_in)
        if return_shared_kv:
            return x, new_cache, shared
        return x, new_cache


class ChronosMLXModel(nn.Module):
    """
    Full Chronos model in MLX.

    Returns logits [B,S,V] and lookahead [B,S,K+1,G,E] (legacy: no G axis).
    Cache: list of per-layer attention caches (passed back in, None at start).
    """

    def __init__(self, config):
        super().__init__()
        # Kernel-level bound on MLX's free-buffer cache (min(RAM/8, 16 GiB)),
        # applied once per process before any weight is allocated; a caller
        # that set a lower limit first (recipe/env) keeps it.
        from chronos.mlx.memory import apply_cache_limit
        apply_cache_limit()
        self.config = config
        # Image/audio encoders + gated cross-attention (Torch parity), only
        # for the modalities the config enables.
        from chronos.mlx.multimodal import MultimodalFusionMLX
        self.multimodal_fusion = MultimodalFusionMLX(config) if config.modalities else None
        self.recurrent_controller = RecurrentControllerMLX(config)
        object.__setattr__(self, "last_recurrent_state", None)
        object.__setattr__(self, "last_modality_states", {})
        self.dropout = nn.Dropout(config.dropout)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [ChronosMLXBlock(i, config)
                       for i in range(config.num_hidden_layers)]
        self.norm = RMSNormMLX(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = None if config.tie_word_embeddings else nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # LookaheadRouter inserted after block 0
        self.lookahead_router = LookaheadRouterMLX(config)

    def __call__(
        self,
        input_ids: mx.array,               # [B, S]
        cache=None,                         # list of per-layer caches, or None
        available_expert_masks=None,        # list of [E] float masks, or None
        logits_dtype=None,                  # e.g. mx.float32 for losses
        recurrent_depth=None,               # draft depth override (shared_first/first_only)
        modality_inputs=None,               # {"image"|"audio"|"*_embeds"|"*_mask": array}
        modality_mask=None,                 # [B] or [B, 2] (image, audio) presence
        return_hidden: bool = False,       # skip the LM head (chunked losses)
        segments=None,        # [B, S] document ids: attention stays in a document
    ):
        """``segments`` ``[B, S]`` (non-decreasing document ids of a packed row,
        as ``chronos.expert_clusters.segment_ids`` gives) keeps every attention
        layer inside the query's own document (document masking for varlen
        packing).  Rows longer than one block take the blockwise path
        (``attention_block_size``); shorter ones get an explicit mask.

        ``return_hidden=True`` returns ``(hidden [B,S,H], lookahead, cache)``
        without the LM head, so a loss never materializes [B,S,V] logits
        (``chronos.mlx.chunked_ce``).

        ``logits_dtype=mx.float32`` makes the logits FP32 at birth: the
        final hidden [B,S,H] and the head [V,H] are upcast (H/V the size of
        the logits) instead of rounding logits to the compute dtype and
        casting the large [B,S,V] tensor afterwards."""
        self._set_segments(segments)
        moes = [layer.mlp for layer in self.layers if isinstance(layer.mlp, ChronosMLXMOE)]
        for moe in moes:
            moe.reset_iteration_routing()
            moe.open_pack_scope()
        try:
            return self._forward(input_ids, cache, available_expert_masks, logits_dtype, recurrent_depth,
                                 modality_inputs, modality_mask, return_hidden)
        finally:
            for moe in moes:
                moe.close_pack_scope()
            self._set_segments(None)

    def _forward(self, input_ids, cache, available_expert_masks, logits_dtype, depth_override=None,
                 modality_inputs=None, modality_mask=None, return_hidden=False):
        B, S = input_ids.shape
        x = self.dropout(self.embed_tokens(input_ids))   # [B, S, H]
        object.__setattr__(self, "last_modality_states", {})
        if self.multimodal_fusion is not None:
            x, states = self.multimodal_fusion(x, modality_inputs, modality_mask)
            object.__setattr__(self, "last_modality_states", states)
        elif modality_inputs:
            raise ValueError("modality_inputs were provided, but this checkpoint has no enabled modalities; "
                             "set config.modalities before constructing the model")

        # Each attention layer builds its own boolean causal / window mask
        # (fused SDPA); no additive mask is materialized in the compute dtype.
        mask = None

        new_caches = []
        lookahead_probs = None

        if cache is None:
            cache = [None] * len(self.layers)

        cfg = self.config
        object.__setattr__(self, "last_recurrent_state", None)
        # Activation recomputation (Torch parity: training, no incoming cache).
        recompute = self.training and bool(getattr(cfg, "activation_checkpointing", False))

        def run(i, hidden, slot=None, **kw):
            avail = available_expert_masks[i] if available_expert_masks is not None else None
            c = cache[i if slot is None else slot] if "kv_override" not in kw else None
            if (recompute and c is None and getattr(cfg, "recompute_scope", "block") == "block"
                and _recompute_block(cfg, i, self.layers[i])):
                return call_block(self.layers[i], hidden, mask=mask, cache=None, available_expert_mask=avail, **kw)
            return self.layers[i](hidden, mask=mask, cache=c, available_expert_mask=avail, **kw)

        if cfg.recurrent_enabled and cfg.recurrent_depth > 1:
            controller = self.recurrent_controller
            depth = controller.execution_depth(cfg, self.training)
            adaptive = controller.adaptive_active(cfg, self.training)
            mode = resolve_recurrent_attention_mode(cfg)
            if depth_override is not None:
                # Shallow (draft) execution, Torch parity: only depth-free
                # cache layouts can share a KV stream across depths.
                if mode == "per_iteration":
                    raise ValueError("depth override needs a depth-independent cache (shared_first/first_only)")
                depth = max(1, min(int(depth_override), int(cfg.recurrent_depth)))
            layout = recurrent_cache_layout(cfg, depth)
            if len(cache) != layout.num_slots:
                if all(c is None for c in cache):
                    cache = [None] * layout.num_slots
                else:
                    raise ValueError(f"MLX recurrent cache has {len(cache)} slots; {mode} depth={depth} expects {layout.num_slots}")
            new_caches = [None]*layout.num_slots
            for i in range(cfg.prelude_layers):
                x, new_caches[i] = run(i, x)
            # Lookahead observes the post-prelude state (Torch parity).
            lookahead_probs = self.lookahead_router(
                mx.stop_gradient(x) if getattr(cfg, "lookahead_input_detach", False) else x)
            # MLX always returns a cache, so per_iteration (one slot per
            # iteration) never exits early; shared_first/first_only may.
            allow_exit = mode != "per_iteration"
            diagnostics = []
            order = str(getattr(cfg, "recurrent_order", "layer") or "layer")
            # Graded top-k: loop iterations after the first route to fewer experts (0 = same as the first).
            later_k = int(getattr(cfg, "recurrent_later_topk", 0) or 0)
            # Prefix blocks (block order): iterations after the first run only the first later_g layers of the block.
            later_g = int(getattr(cfg, "recurrent_later_groups", 0) or 0)
            target = min(1.0, max(1e-6, cfg.halting_threshold))
            
            def run_layer(group, step, x, shared):
                """One loop iteration of one physical layer: condition, attend/route, contract."""
                i = cfg.prelude_layers + group
                tk = {"top_k": later_k} if later_k and step > 0 else {}
                conditioned = controller.condition(x, step, group)
                slot = layout.slot(i, step)
                if mode == "per_iteration":
                    refined, new_caches[slot] = run(i, conditioned, slot, **tk)
                elif mode == "first_only":
                    refined, candidate = run(i, conditioned, slot, latent_only=step > 0, **tk)
                    if step == 0:
                        new_caches[slot] = candidate
                elif step == 0:
                    refined, new_caches[slot], shared = run(i, conditioned, slot, return_shared_kv=True)
                else:
                    refined, _ = run(i, conditioned, kv_override=shared, **tk)
                return controller.contract(x, refined, group, cfg), shared
            
            def track(x, step, halted, cumulative):
                prob = controller.halt_probability(x)
                # Exact per-token ACT halting (Torch parity, see model_chronos).
                if step + 1 >= cfg.min_recurrent_depth:
                    clipped = mx.clip(prob.astype(mx.float32), 0.0, 1.0)
                    if halted is None:
                        cumulative = mx.zeros_like(clipped)
                        halted = mx.zeros(clipped.shape, dtype=mx.bool_)
                    newly = mx.logical_and(mx.logical_not(halted), cumulative + clipped >= target)
                    cumulative = mx.where(mx.logical_or(halted, newly), cumulative, mx.minimum(cumulative + clipped, 1.0))
                    halted = mx.logical_or(halted, newly)
                return prob, halted, cumulative
            
            def can_exit(halted, step):
                if not self.training and adaptive and allow_exit and halted is not None and step + 1 < depth:
                    return bool(mx.all(halted).item())
                return False
            
            def finish(states, halts, exited, x):
                mixed, state = auxiliary_state(states, halts, cfg.min_recurrent_depth, cfg.halting_threshold)
                if adaptive:
                    x = mixed
                else:
                    x = states[-1]
                    state["ponder_loss"] = mx.zeros(())
                    state["depth_used"] = mx.full(halts[0].shape, len(states), dtype=mx.float32)
                state["exited"] = exited
                diagnostics.append(state)
                return x
            
            if order == "block":
                # Block-interleaved recurrence: (layer0 layer1 ... layerG-1) x depth.
                # One state trajectory and one halting head per block iteration; each
                # physical layer still keeps its own dt, loop/group embeddings and KV.
                shared = [None] * cfg.recurrent_groups
                states, halts = [], []
                halted = cumulative = None
                exited = False
                for step in range(depth):
                    for group in range(cfg.recurrent_groups if step == 0 or not later_g else min(cfg.recurrent_groups, later_g)):
                        x, shared[group] = run_layer(group, step, x, shared[group])
                    states.append(x)
                    prob, halted, cumulative = track(x, step, halted, cumulative)
                    halts.append(prob)
                    if can_exit(halted, step):
                        exited = True
                        break
                x = finish(states, halts, exited, x)
            else:
                # Layer-consecutive recurrence (default): layer0 x depth, layer1 x depth, ...
                for group in range(cfg.recurrent_groups):
                    states, halts = [], []
                    exited, shared = False, None
                    halted = cumulative = None
                    for step in range(depth):
                        x, shared = run_layer(group, step, x, shared)
                        states.append(x)
                        prob, halted, cumulative = track(x, step, halted, cumulative)
                        halts.append(prob)
                        if can_exit(halted, step):
                            exited = True
                            break
                    x = finish(states, halts, exited, x)
            for i in range(cfg.prelude_layers+cfg.recurrent_groups, len(self.layers)):
                slot = layout.slot(i, 0)
                x, new_caches[slot] = run(i, x, slot)
            max_depth = max(s["halting_probs"].shape[-1] for s in diagnostics)
            halting = [mx.pad(s["halting_probs"], [(0,0), (0,0), (0,max_depth-s["halting_probs"].shape[-1])], constant_values=1) for s in diagnostics]
            combined = {key: mx.stack([s[key] for s in diagnostics]).mean() for key in ("ponder_loss", "distill_loss", "stability_loss", "contraction")}
            combined.update(halting_probs=mx.stack(halting, axis=-2), depth_used=mx.stack([s["depth_used"] for s in diagnostics], axis=-1),
                            early_exit_ratio=mx.array(sum(int(s["exited"]) for s in diagnostics)/len(diagnostics)))
            object.__setattr__(self, "last_recurrent_state", combined)
        else:
            for i in range(len(self.layers)):
                x, candidate = run(i, x)
                new_caches.append(candidate)
                if i == 0:
                    lookahead_probs = self.lookahead_router(
                        mx.stop_gradient(x) if getattr(cfg, "lookahead_input_detach", False) else x)

        x = self.norm(x)
        if return_hidden:
            return x, lookahead_probs, new_caches
        head = self.head_weight()
        if logits_dtype is not None and logits_dtype != head.dtype:
            logits = x.astype(logits_dtype) @ head.astype(logits_dtype).T
        else:
            logits = self.embed_tokens.as_linear(x) if self.lm_head is None else self.lm_head(x)
        return logits, lookahead_probs, new_caches

    def _set_segments(self, segments) -> None:
        """Give every attention layer this forward's document ids (or clear them)."""
        seg = None if segments is None else mx.stop_gradient(mx.array(segments).astype(mx.int32))
        for layer in self.layers:
            object.__setattr__(layer.self_attn, "_segments", seg)

    def head_weight(self) -> mx.array:
        """The LM head matrix ``[V, H]`` (the embedding when tied)."""
        return self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    @staticmethod
    def from_chronos_pytorch(pt_model, config, include_experts: bool = True, dtype: str | None = None):
        """
        Convert a trained PyTorch ChronosForCausalLM to ChronosMLXModel.

        ``dtype`` is the MLX compute dtype (``chronos.precision.resolve_mlx``);
        each tensor is converted once, straight from the checkpoint dtype
        (BF16 crosses bit-exactly).  ``None`` keeps the legacy FP32 load.
        """
        from chronos.precision import torch_to_mx
        mlx_model = ChronosMLXModel(config)
        target = dtype or "float32"

        def _to_mx(pt_tensor):
            return torch_to_mx(pt_tensor, target)

        def _assign_linear(linear, sd, key, missing):
            if key in sd:
                if tuple(linear.weight.shape) != tuple(sd[key].shape):
                    raise ValueError(f"MLX checkpoint shape mismatch for {key}: {tuple(sd[key].shape)} != {tuple(linear.weight.shape)}")
                linear.weight = _to_mx(sd[key])
            else:
                missing.append(key)

        def _assign_norm(norm, sd, key, missing):
            if key in sd:
                norm.weight = _to_mx(sd[key])
            else:
                missing.append(key)

        # Embedding + head
        sd = pt_model.state_dict()
        missing = []
        for key, obj in [
            ("model.embed_tokens.weight", mlx_model.embed_tokens),
            ("lm_head.weight", mlx_model.lm_head),
        ]:
            if obj is None:
                continue
            if key in sd:
                obj.weight = _to_mx(sd[key])
            else:
                missing.append(key)
        _assign_norm(mlx_model.norm, sd, "model.norm.weight", missing)

        for i in range(config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            blk = mlx_model.layers[i]
            _assign_norm(blk.input_norm, sd, f"{prefix}.input_layernorm.weight", missing)
            _assign_norm(blk.post_norm, sd, f"{prefix}.post_attention_layernorm.weight", missing)

            attn = blk.self_attn
            attn_linears = [
                "q_nope_proj", "q_rope_proj", "kv_down_proj",
                "k_nope_proj", "k_rope_proj", "v_proj", "o_proj",
                "q_proj", "k_proj",
            ]
            for name in attn_linears:
                if hasattr(attn, name):
                    _assign_linear(getattr(attn, name), sd, f"{prefix}.self_attn.{name}.weight", missing)
            for name in ["q_nope_norm", "q_rope_norm", "kv_down_norm", "q_norm", "k_norm"]:
                if hasattr(attn, name):
                    _assign_norm(getattr(attn, name), sd, f"{prefix}.self_attn.{name}.weight", missing)

            moe = blk.mlp
            if isinstance(moe, ChronosMLXMOE):
                _assign_linear(moe.gate, sd, f"{prefix}.mlp.gate.weight", missing)
                if include_experts:
                    for ei, expert in enumerate(moe.experts):
                        for proj in ("gate_proj", "up_proj", "down_proj"):
                            _assign_linear(
                                getattr(expert, proj),
                                sd,
                                f"{prefix}.mlp.experts.{ei}.{proj}.weight",
                                missing,
                            )
                for si, expert in enumerate(moe.shared_experts):
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        _assign_linear(
                            getattr(expert, proj),
                            sd,
                            f"{prefix}.mlp.shared_experts.{si}.{proj}.weight",
                            missing,
                        )

            else:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    _assign_linear(getattr(moe, proj), sd, f"{prefix}.mlp.{proj}.weight", missing)

        controller = mlx_model.recurrent_controller
        for key in ("loop_embeddings", "group_embeddings", "step_sizes"):
            source = f"model.recurrent_controller.{key}"
            if source in sd:
                setattr(controller, key, _to_mx(sd[source]))
            else:
                missing.append(source)
        _assign_norm(controller.halting_norm, sd, "model.recurrent_controller.halting_norm.weight", missing)
        _assign_linear(controller.halting_head, sd, "model.recurrent_controller.halting_head.weight", missing)
        controller.halting_head.bias = _to_mx(sd["model.recurrent_controller.halting_head.bias"])
        proj_layers = mlx_model.lookahead_router.proj.layers
        _assign_linear(proj_layers[0], sd, "model.lookahead_router.proj.0.weight", missing)
        _assign_linear(proj_layers[2], sd, "model.lookahead_router.proj.2.weight", missing)
        if mlx_model.multimodal_fusion is not None:
            from chronos.mlx.multimodal import load_torch_fusion_state
            load_torch_fusion_state(mlx_model.multimodal_fusion, sd, _to_mx, missing)

        if missing:
            raise RuntimeError(
                "MLX conversion is missing Chronos checkpoint tensors: "
                + ", ".join(missing[:40])
            )
        if dtype is not None:
            mlx_model.set_dtype(getattr(mx, target))   # params not in the checkpoint (fresh init)
        mlx_model.train(pt_model.training)
        mx.eval(mlx_model.parameters())
        return mlx_model
