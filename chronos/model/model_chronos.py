import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin

from .kernel import FeedForward, rope_tables
from .config import ChronosConfig
from .precision_ops import RMSNorm, lm_head_logits
from .lookahead_router import LookaheadRouter
from .moe_chronos import ChronosMOEFeedForward
from .hybrid_attention import MLAAttention, make_attention
from .recurrent import RecurrentController, recurrent_auxiliary_state
from .outputs import ChronosCausalLMOutput
from .multimodal import MultimodalFusion
from chronos.config_schema import resolve_recurrent_attention_mode, recurrent_cache_layout


def cache_sequence_length(past_key_values) -> int:
    """Absolute number of tokens already in a Chronos KV cache.

    MLA entries are ``(c_kv,)``; sliding-window entries carry their absolute
    position as the third element because their K/V tensors are evicted.
    """
    for entry in past_key_values or ():
        if entry is not None:
            return int(entry[2]) if len(entry) > 2 else int(entry[0].shape[1])
    return 0


def draft_exit_kl(draft_logits: torch.Tensor, full_logits: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    """Mean token KL(full.detach() || draft) in FP32 over supervised positions.

    Positions follow the CE shift (logit t predicts label t+1); without
    labels every position counts.  Teacher gradients are stopped so the
    term only pulls the shallow exit toward the full-depth distribution.
    """
    teacher = F.log_softmax(full_logits.detach().float(), dim=-1)
    student = F.log_softmax(draft_logits.float(), dim=-1)
    kl = (teacher.exp() * (teacher - student)).sum(-1)          # [B, S]
    if labels is None:
        return kl.mean()
    keep = torch.zeros_like(kl, dtype=torch.bool)
    keep[:, :-1] = labels[:, 1:] != -100
    return (kl * keep).sum() / keep.sum().clamp_min(1)


class ChronosBlock(nn.Module):
    """
    Transformer block with hybrid attention:
      even layers → MLAAttention  (compressed KV cache, global context)
      odd  layers → SlidingWindowAttention (bounded KV cache, local context)
    """

    def __init__(self, layer_id: int, config: ChronosConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = make_attention(layer_id, config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ChronosMOEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
        available_expert_mask=None,
        latent_only: bool = False,
        kv_override=None,
        return_shared_kv: bool = False,
    ):
        """One physical block.

        ``latent_only`` skips attention (v2 ``first_only`` recurrence).
        ``return_shared_kv`` / ``kv_override`` implement the v3
        ``shared_first`` contract: iteration zero returns its full K/V view,
        later iterations attend with fresh queries against that view.
        """
        residual = hidden_states
        shared_kv = None
        if latent_only:
            # Middle recurrent iterations refine the latent state without
            # rereading the prefix or appending another KV entry.
            present_key_value = past_key_value
        else:
            extra = {}
            if kv_override is not None or return_shared_kv:
                extra = {"kv_override": kv_override, "return_shared_kv": return_shared_kv}
            attn_out = self.self_attn(
                self.input_layernorm(hidden_states),
                position_embeddings,
                past_key_value,
                use_cache,
                attention_mask,
                **extra,
            )
            if return_shared_kv:
                hidden_states, present_key_value, shared_kv = attn_out
            else:
                hidden_states, present_key_value = attn_out
            hidden_states = hidden_states + residual
        mlp_in = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, ChronosMOEFeedForward):
            hidden_states = hidden_states + self.mlp(mlp_in, available_expert_mask)
        else:
            hidden_states = hidden_states + self.mlp(mlp_in)
        if return_shared_kv:
            return hidden_states, present_key_value, shared_kv
        return hidden_states, present_key_value


def apply_repetition_penalty(logits, input_ids, penalty: float):
    """CTRL-style penalty: shrink seen-token logits toward zero.

    Dividing a negative logit by ``penalty > 1`` would *raise* its
    probability; multiply negatives and divide positives instead.
    """
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, input_ids, True)
    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen, penalized, logits)


class ChronosModel(nn.Module):
    """
    Backbone: pre-norm decoder (native kernel, ``chronos.model.kernel``) that:
    - Uses ChronosBlock (with ChronosMOEFeedForward)
    - Inserts LookaheadRouter after block 0
    - Exposes router_probs_seq for temporal loss computation
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            ChronosBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lookahead_router = LookaheadRouter(config)
        self.recurrent_controller = RecurrentController(config)
        # Pure-text models carry no image/audio encoder parameters or compute.
        self.multimodal_fusion = MultimodalFusion(config) if config.modalities else None
        # Plain FP32 attributes (not buffers): ``model.to(bf16)`` must not
        # round them.  The table is the shared one every SWA/global layer
        # uses (``kernel.rope_tables``), so it costs no extra memory.
        self.freqs_cos, self.freqs_sin = rope_tables(
            config.head_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling, "cpu")
        self.last_recurrent_state = None

    def _recompute_block(self, layer) -> bool:
        """Whether ``layer`` is recomputed under ``config.recompute_policy``
        ("all" | "mla" | "swa" | "alternate"); MLX parity."""
        policy = getattr(self.config, "recompute_policy", "all")
        if policy == "all":
            return True
        if policy == "alternate":
            return next(i for i, other in enumerate(self.layers) if other is layer) % 2 == 0
        is_mla = isinstance(layer.self_attn, MLAAttention)
        return is_mla if policy == "mla" else not is_mla

    def _run_block(
        self, layer, hidden_states, position_embeddings, *, past_key_value=None,
        use_cache=False, attention_mask=None, available_expert_mask=None,
        latent_only: bool = False,
        kv_override=None,
        return_shared_kv: bool = False,
    ):
        """Run one physical block with optional selective recomputation."""
        if (
            bool(getattr(self.config, "activation_checkpointing", False))
            and self.training and not use_cache and not return_shared_kv
            and self._recompute_block(layer)
        ):
            from torch.utils.checkpoint import checkpoint

            def transition(states):
                return layer(
                    states, position_embeddings, past_key_value=None,
                    use_cache=False, attention_mask=attention_mask,
                    available_expert_mask=available_expert_mask,
                    latent_only=latent_only, kv_override=kv_override,
                )[0]

            return checkpoint(transition, hidden_states, use_reentrant=False), None
        return layer(
            hidden_states, position_embeddings,
            past_key_value=past_key_value, use_cache=use_cache,
            attention_mask=attention_mask,
            available_expert_mask=available_expert_mask,
            latent_only=latent_only,
            kv_override=kv_override,
            return_shared_kv=return_shared_kv,
        )

    def _forward_recurrent(
        self,
        hidden_states,
        position_embeddings,
        past_key_values,
        use_cache,
        attention_mask,
        available_expert_masks,
        depth_override=None,
    ):
        """Execute Prelude → shared groups × depth → Coda.

        ``recurrent_attention_mode`` selects how loop iterations read context:

        * ``shared_first`` (v3 default): every iteration attends.  Iteration
          zero writes the group's single KV entry; later iterations recompute
          queries from the refined state and attend to that same K/V view.
          Training, prefill and decode therefore see identical keys, the KV
          footprint equals the physical layer count, and the ACT exit never
          leaves a cache hole.
        * ``per_iteration``: every iteration owns a KV slot (virtual-layer
          cache, ``prelude + groups·depth + coda`` entries).  Exact but D×
          the KV memory; kept as an ablation.  Decode depth is fixed.
        * ``first_only``: v2 semantics; iterations after zero skip attention.
        """
        cfg = self.config
        p = int(cfg.prelude_layers)
        g = int(cfg.recurrent_groups)
        c = int(cfg.coda_layers)
        mode = resolve_recurrent_attention_mode(cfg)
        depth = self.recurrent_controller.execution_depth(cfg, self.training)
        if depth_override is not None:
            # Self-speculative drafting runs fewer loop iterations.  Only
            # depth-independent cache layouts can share a KV stream.
            if mode == "per_iteration":
                raise ValueError("depth override needs a depth-independent cache (shared_first/first_only)")
            depth = max(1, min(int(depth_override), int(cfg.recurrent_depth)))
        layout = recurrent_cache_layout(cfg, depth)
        if past_key_values is None:
            past_key_values = [None] * layout.num_slots
        elif len(past_key_values) != layout.num_slots:
            raise ValueError(
                f"recurrent cache has {len(past_key_values)} slots; mode={mode} depth={depth} "
                f"expects {layout.num_slots} (decode must reuse the prefill depth)"
            )
        presents = [None] * layout.num_slots

        def mask_for(li):
            return available_expert_masks[li] if available_expert_masks else None

        # Prelude remains a normal one-pass stack.
        for li in range(p):
            hidden_states, presents[li] = self._run_block(
                self.layers[li], hidden_states, position_embeddings,
                past_key_value=past_key_values[li], use_cache=use_cache,
                attention_mask=attention_mask, available_expert_mask=mask_for(li),
            )
        # Lookahead observes the post-prelude state: earliest stable routing
        # evidence, identical for every recurrent depth (draft or verify).
        lookahead_source = hidden_states

        recurrent_states = []
        # Physical early exit never changes the KV a later token sees in
        # shared_first/first_only (only iteration zero writes KV), so it is
        # allowed during prefill and decode.  per_iteration owns one slot per
        # iteration: exiting would leave cache holes, so only no-cache
        # forwards may exit there.
        allow_adaptive_exit = mode != "per_iteration" or not use_cache
        adaptive_active = self.recurrent_controller.adaptive_active(cfg, self.training)
        threshold = float(cfg.halting_threshold)
        min_depth = int(cfg.min_recurrent_depth)
        for group in range(g):
            li = p + group
            layer = self.layers[li]
            group_states, group_halts = [], []
            exited_early = False
            shared_kv = None
            halted = cumulative = None
            target = min(1.0, max(1e-6, threshold))
            for step in range(depth):
                conditioned = self.recurrent_controller.condition(hidden_states, step, group)
                slot = layout.slot(li, step)
                block_kwargs = dict(
                    use_cache=use_cache, attention_mask=attention_mask,
                    available_expert_mask=mask_for(li),
                )
                if mode == "per_iteration":
                    refined, presents[slot] = self._run_block(
                        layer, conditioned, position_embeddings,
                        past_key_value=past_key_values[slot], **block_kwargs)
                elif mode == "first_only":
                    refined, present = self._run_block(
                        layer, conditioned, position_embeddings,
                        past_key_value=past_key_values[slot], latent_only=step > 0, **block_kwargs)
                    if step == 0:
                        presents[slot] = present
                elif step == 0:  # shared_first, writer iteration
                    refined, presents[slot], shared_kv = self._run_block(
                        layer, conditioned, position_embeddings,
                        past_key_value=past_key_values[slot], return_shared_kv=True, **block_kwargs)
                else:  # shared_first, reader iteration
                    refined, _ = self._run_block(
                        layer, conditioned, position_embeddings,
                        past_key_value=None, kv_override=shared_kv,
                        use_cache=False, attention_mask=attention_mask,
                        available_expert_mask=mask_for(li))
                hidden_states = self.recurrent_controller.contract(hidden_states, refined, group)
                group_states.append(hidden_states)
                halt_probability = self.recurrent_controller.halt_probability(hidden_states)
                group_halts.append(halt_probability)
                # Exact ACT bookkeeping (same rule as recurrent_auxiliary_state):
                # a token halts once its accumulated mass reaches the threshold
                # after min_depth.  Later iterations get zero mixture weight, so
                # once *every* token has halted the remaining iterations cannot
                # change the output; exiting is then exact, and each token's
                # result is independent of what else is in the batch (required
                # for speculative verification to equal token-by-token decode).
                if step + 1 >= min_depth:
                    clipped = halt_probability.float().clamp(0.0, 1.0)
                    if halted is None:
                        cumulative = torch.zeros_like(clipped)
                        halted = torch.zeros_like(clipped, dtype=torch.bool)
                    newly = (~halted) & ((cumulative + clipped) >= target)
                    cumulative = torch.where(halted | newly, cumulative, (cumulative + clipped).clamp(max=1.0))
                    halted = halted | newly
                if (
                    not self.training and adaptive_active and allow_adaptive_exit
                    and halted is not None and step + 1 < depth
                ):
                    if bool(halted.all()):  # one fused device→host read per step
                        exited_early = True
                        break
            # ACT is state interpolation rather than a ragged Python loop.
            mixed, state = recurrent_auxiliary_state(
                group_states, group_halts, min_depth=min_depth, threshold=threshold,
            )
            if adaptive_active:
                hidden_states = mixed
            else:
                hidden_states = group_states[-1]
                state.ponder_cost = hidden_states.new_zeros(())
                state.depth_used = torch.full_like(state.depth_used, float(len(group_states)))
            state.exited_early = exited_early
            recurrent_states.append(state)

        # Coda uses the remaining physical layers.
        for ci in range(c):
            li = p + g + ci
            slot = layout.slot(li, 0)
            hidden_states, presents[slot] = self._run_block(
                self.layers[li], hidden_states, position_embeddings,
                past_key_value=past_key_values[slot], use_cache=use_cache,
                attention_mask=attention_mask, available_expert_mask=mask_for(li),
            )

        if recurrent_states:
            max_depth = max(int(s.halting_probs.shape[-1]) for s in recurrent_states)
            padded_halts = [
                F.pad(s.halting_probs, (0, max_depth - int(s.halting_probs.shape[-1])), value=1.0)
                for s in recurrent_states
            ]
            self.last_recurrent_state = {
                "halting_probs": torch.stack(padded_halts, dim=-2),
                "ponder_loss": torch.stack([s.ponder_cost for s in recurrent_states]).mean(),
                "stability_loss": torch.stack([s.stability_loss for s in recurrent_states]).mean(),
                "distill_loss": torch.stack([s.distill_loss for s in recurrent_states]).mean(),
                "contraction": torch.stack([s.contraction for s in recurrent_states]).mean(),
                # Keep the group axis so layer/group-aware schedulers can
                # correlate ponder depth with their cache keys.
                "depth_used": torch.stack([s.depth_used for s in recurrent_states], dim=-1),
                "early_exit_ratio": hidden_states.new_tensor(
                    sum(int(s.exited_early) for s in recurrent_states) / max(len(recurrent_states), 1)
                ),
                "attention_mode": mode,
                "executed_depth": depth,
            }
        else:
            self.last_recurrent_state = None
        return hidden_states, presents, lookahead_source

    def _ensure_rope_cache(self, needed_len: int, device: torch.device) -> None:
        """Grow RoPE buffers when generation exceeds the checkpoint context."""
        current_len = int(self.freqs_cos.shape[0])
        if needed_len <= current_len and self.freqs_cos.device == device:
            return
        target_len = max(needed_len, current_len * 2, int(self.config.max_position_embeddings))
        self.freqs_cos, self.freqs_sin = rope_tables(
            self.config.head_dim, target_len, self.config.rope_theta, self.config.rope_scaling, device)
        self.config.max_position_embeddings = max(
            int(self.config.max_position_embeddings),
            target_len,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        available_expert_masks=None,  # list[Tensor|None] per layer
        modality_inputs=None,
        modality_mask=None,
        recurrent_depth=None,
        **kwargs,
    ):
        B, S = input_ids.shape
        for layer in self.layers:
            if isinstance(layer.mlp, ChronosMOEFeedForward):
                layer.mlp.reset_iteration_routing()
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        # MLA caches only c_kv; SlidingWindow adds an absolute position counter.
        start_pos = cache_sequence_length(past_key_values)
        self._ensure_rope_cache(start_pos + S, input_ids.device)

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        modality_states = {}
        if self.multimodal_fusion is not None:
            hidden_states, modality_states = self.multimodal_fusion(
                hidden_states, modality_inputs, modality_mask
            )
        elif modality_inputs:
            raise ValueError(
                "modality_inputs were provided, but this checkpoint has no enabled modalities; "
                "set config.modalities before constructing the model"
            )
        # Pass full freqs so hybrid attention layers can slice as needed
        position_embeddings = (
            self.freqs_cos,
            self.freqs_sin,
        )

        presents = []
        lookahead_probs = None  # [B, S, K, E] from block 0

        # A one-loop model is routed through the proven v1 graph.  The loop
        # controller is an exact no-op in this migration mode, including the
        # original lookahead observation point after physical block zero.
        recurrent_active = (
            bool(getattr(self.config, "recurrent_enabled", False))
            and int(getattr(self.config, "recurrent_depth", 1)) > 1
        )
        if recurrent_active:
            hidden_states, presents, lookahead_source = self._forward_recurrent(
                hidden_states, position_embeddings, past_key_values, use_cache,
                attention_mask, available_expert_masks, depth_override=recurrent_depth,
            )
            # Post-prelude state: earliest stable routing observation.
            lookahead_probs = self.lookahead_router(lookahead_source)
        else:
            self.last_recurrent_state = None
            past_key_values = past_key_values or [None] * len(self.layers)
            for i, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
                mask = available_expert_masks[i] if available_expert_masks else None
                hidden_states, present = self._run_block(
                    layer,
                    hidden_states, position_embeddings,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    available_expert_mask=mask,
                )
                presents.append(present)
                if i == 0:
                    lookahead_probs = self.lookahead_router(hidden_states)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers
             if isinstance(l.mlp, ChronosMOEFeedForward) and l.mlp.aux_loss is not None],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss, lookahead_probs, modality_states


class RecurrentChronosModel(ChronosModel):
    """Explicit v2 backbone name for runtime/checkpoint dispatch.

    ``ChronosModel`` remains importable as the v1-compatible executor.  The
    subclass intentionally shares parameter names so v1→depth-1 migration is
    identity-preserving and does not require tensor-key rewrites.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__(config)
        if int(getattr(config, "architecture_version", 2)) < 2:
            raise ValueError("RecurrentChronosModel requires architecture_version >= 2")


class ChronosForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ChronosConfig
    # Map of tied-weight keys (lm_head.weight ↔ model.embed_tokens.weight).
    # Newer Transformers expects a dict; older versions accept a list. We
    # use a dict to satisfy the newest API.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @property
    def all_tied_weights_keys(self) -> dict:
        """Compat shim: newer Transformers expects this property to exist
        on every PreTrainedModel; older base classes don't define it."""
        v = self._tied_weights_keys
        if isinstance(v, dict):
            return v
        if isinstance(v, (list, tuple, set)):
            return {k: None for k in v}
        return {}

    def __init__(self, config: ChronosConfig = None):
        self.config = config or ChronosConfig()
        super().__init__(self.config)
        model_cls = (
            RecurrentChronosModel
            if int(getattr(self.config, "architecture_version", 2)) >= 2
            else ChronosModel
        )
        self.model = model_cls(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.last_draft_exit_kl = None
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        available_expert_masks=None,
        modality_inputs=None,
        modality_mask=None,
        legacy_output=False,
        recurrent_depth=None,
        logits_dtype=None,
        **kwargs,
    ):
        """``logits_dtype``: dtype the logits are born in.  Defaults to FP32
        on loss paths (``labels`` given, or training mode: DPO/ORPO/GRPO/KD
        take log-softmax of the logits) and to the compute dtype for
        inference, where sampling upcasts only the kept [B, V] rows.

        With ``draft_exit_kl_coef > 0`` a training forward with labels also
        runs the model at ``draft_exit_depth`` first and stores
        KL(full.detach() || draft) on ``self.last_draft_exit_kl`` (consumed by
        ``chronos_loss_term``); the full-depth pass runs last so routing and
        recurrent diagnostics describe the full model."""
        self.last_draft_exit_kl = None
        draft_logits = None
        if self._draft_exit_active(labels, past_key_values, use_cache, recurrent_depth):
            draft_hidden = self.model(
                input_ids, attention_mask, None, False,
                available_expert_masks=available_expert_masks,
                modality_inputs=modality_inputs, modality_mask=modality_mask,
                recurrent_depth=int(self.config.draft_exit_depth),
            )[0]
            draft_logits = lm_head_logits(draft_hidden, self.lm_head.weight, torch.float32)
        hidden_states, past_key_values, aux_loss, lookahead_probs, modality_states = self.model(
            input_ids, attention_mask, past_key_values, use_cache,
            available_expert_masks=available_expert_masks,
            modality_inputs=modality_inputs,
            modality_mask=modality_mask,
            recurrent_depth=recurrent_depth,
        )
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        if logits_dtype is None and (labels is not None or self.training):
            logits_dtype = torch.float32
        logits = lm_head_logits(hidden_states[:, slice_indices, :], self.lm_head.weight, logits_dtype)

        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        if draft_logits is not None:
            draft_logits = draft_logits[:, slice_indices, :]
            kl_labels = labels[:, -logits.shape[1]:] if labels.shape[1] != logits.shape[1] else labels
            self.last_draft_exit_kl = draft_exit_kl(draft_logits, logits, kl_labels)

        rs = self.model.last_recurrent_state or {}
        result = ChronosCausalLMOutput(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            lookahead_probs=lookahead_probs,
            halting_probs=rs.get("halting_probs"),
            ponder_loss=rs.get("ponder_loss"),
            recurrent_stability_loss=rs.get("stability_loss"),
            recurrent_distill_loss=rs.get("distill_loss"),
            recurrent_contraction=rs.get("contraction"),
            recurrent_depth_used=rs.get("depth_used"),
            cache_hints={
                "layer_aware": bool(getattr(self.config, "layer_aware_cache", False)),
                "early_exit_ratio": rs.get("early_exit_ratio"),
            },
            modality_states=modality_states,
        )
        return result.legacy_tuple() if legacy_output else result

    def _draft_exit_active(self, labels, past_key_values, use_cache, recurrent_depth) -> bool:
        cfg = self.config
        if not (self.training and labels is not None and past_key_values is None and not use_cache):
            return False
        if float(getattr(cfg, "draft_exit_kl_coef", 0.0) or 0.0) <= 0.0 or recurrent_depth is not None:
            return False
        if not getattr(cfg, "recurrent_enabled", True) or int(cfg.recurrent_depth) <= 1:
            return False
        return 1 <= int(getattr(cfg, "draft_exit_depth", 1)) < int(cfg.recurrent_depth)

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=512,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        do_sample=True,
        repetition_penalty=1.0,
        available_expert_masks=None,
        modality_inputs=None,
        modality_mask=None,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs)
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = cache_sequence_length(past_key_values) if past_key_values else 0
            result = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values,
                use_cache=use_cache,
                available_expert_masks=available_expert_masks,
                modality_inputs=modality_inputs,
                modality_mask=modality_mask,
            )
            outputs, _ = result.legacy_tuple()
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
                )
            logits = outputs.logits[:, -1, :].float() / temperature   # [B, V] sampling row only
            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
            if top_k > 0:
                k = min(int(top_k), int(logits.shape[-1]))
                logits[logits < torch.topk(logits, k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()
        return input_ids
