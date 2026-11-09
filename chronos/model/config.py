from chronos.model.kernel import ChronosBaseConfig


class ChronosConfig(ChronosBaseConfig):
    """
    Extends ChronosBaseConfig with Project Chronos-specific fields:
    - lookahead_steps: how many future token steps to predict expert routing for
    - num_shared_experts: experts always resident in VRAM as fallback
    - lambda_balance / lambda_temporal: loss coefficients
    - vram_budget_gb: VRAM cap for expert cache
    - prefetch_depth: async prefetch queue depth
    """
    model_type = "chronos"

    def __init__(self, **kwargs):
        # Chronos v2 recurrent-depth kernel.  ``num_hidden_layers`` remains
        # the number of *physical* parameter blocks so old checkpoints stay
        # shape-compatible; recurrent_depth controls how often the recurrent
        # portion is applied.  A depth of one is exactly the v1 execution
        # graph and is used when loading legacy checkpoints.
        self.architecture_version: int = kwargs.pop("architecture_version", 3)
        self.recurrent_enabled: bool = kwargs.pop("recurrent_enabled", True)
        self.recurrent_depth: int = kwargs.pop("recurrent_depth", 1)
        self.min_recurrent_depth: int = kwargs.pop("min_recurrent_depth", 1)
        self.prelude_layers: int = kwargs.pop("prelude_layers", 0)
        self.coda_layers: int = kwargs.pop("coda_layers", 0)
        self.recurrent_groups = kwargs.pop("recurrent_groups", None)
        self.adaptive_halting: bool = kwargs.pop("adaptive_halting", False)
        self.halting_threshold: float = kwargs.pop("halting_threshold", 0.99)
        self.halting_epsilon: float = kwargs.pop("halting_epsilon", 0.01)
        # ACT needs a visible incentive to stop once the task loss is already
        # low; 1e-3 was too weak and long runs learned to keep all virtual
        # steps.  1e-2 preserves quality in the convergence benchmark while
        # making adaptive depth a stable operating point.
        self.ponder_loss_coef: float = kwargs.pop("ponder_loss_coef", 1e-2)
        self.recurrent_distill_coef: float = kwargs.pop("recurrent_distill_coef", 0.05)
        self.recurrent_stability_coef: float = kwargs.pop("recurrent_stability_coef", 1e-3)
        self.recurrent_contraction_coef: float = kwargs.pop("recurrent_contraction_coef", 0.0)
        self.recurrent_contraction_budget: float = kwargs.pop("recurrent_contraction_budget", 1.0)
        # Curvature is normalized by hidden-state RMS in the recurrent
        # executor, so 0.1 is a portable relative stability budget.  The old
        # absolute 1e-3 default almost never allowed an adaptive exit.
        self.recurrent_stability_threshold: float = kwargs.pop("recurrent_stability_threshold", 0.1)
        # A full Euler step is numerically valid but is too aggressive when a
        # freshly initialized block is reused 2–8 times.  Starting at 0.25
        # keeps the identity-preserving migration path stable while allowing
        # the learned per-group step size to grow toward dt_max.
        self.recurrent_dt_init: float = kwargs.pop("recurrent_dt_init", 0.25)
        self.recurrent_dt_min: float = kwargs.pop("recurrent_dt_min", 0.05)
        self.recurrent_dt_max: float = kwargs.pop("recurrent_dt_max", 1.0)
        # v3 recurrent attention contract (see config_schema.resolve_*):
        # shared_first | per_iteration | first_only.  An explicit legacy
        # ``recurrent_latent_refinement`` without a mode keeps v2 meaning.
        explicit_latent = "recurrent_latent_refinement" in kwargs
        self.recurrent_latent_refinement: bool = kwargs.pop("recurrent_latent_refinement", True)
        mode = kwargs.pop("recurrent_attention_mode", None)
        if not mode:
            if self.architecture_version >= 3 and not explicit_latent:
                mode = "shared_first"
            else:
                mode = "first_only" if self.recurrent_latent_refinement else "per_iteration"
        mode = str(mode).lower()
        if mode not in ("shared_first", "per_iteration", "first_only"):
            raise ValueError(f"unknown recurrent_attention_mode {mode!r}")
        self.recurrent_attention_mode: str = mode
        # Keep the legacy flag coherent for estimator/export consumers.
        self.recurrent_latent_refinement = mode == "first_only"
        # Execution/training policy for shared recurrent blocks.  ``fixed``
        # preserves the historical graph; ``scheduled`` samples a depth
        # bucket during training and ``adaptive`` enables ACT mixing once the
        # warmup has elapsed.  ``parallel_refine`` is an inference hint used
        # by runtimes that can batch latent iterations.
        self.recurrent_strategy: str = kwargs.pop("recurrent_strategy", "scheduled")
        self.recurrent_depth_jitter: float = kwargs.pop("recurrent_depth_jitter", 0.0)
        self.recurrent_parallel_refine: bool = kwargs.pop("recurrent_parallel_refine", False)
        self.recurrent_gradient_checkpoint_groups: bool = kwargs.pop(
            "recurrent_gradient_checkpoint_groups", False
        )
        self.max_inference_depth = kwargs.pop("max_inference_depth", None)
        # 5-D [B,S,K,L,E] prediction is enabled by v2 presets.  Keeping the
        # switch explicit lets depth=1 legacy checkpoints retain their public
        # 4-D output contract until migrated.
        self.layer_aware_lookahead: bool = kwargs.pop("layer_aware_lookahead", False)
        self.layer_aware_cache: bool = kwargs.pop("layer_aware_cache", False)
        self.depth_schedule: str = kwargs.pop("depth_schedule", "fixed_then_adaptive")
        self.depth_schedule_warmup_fraction: float = kwargs.pop(
            "depth_schedule_warmup_fraction", 0.25
        )
        self.training_paging: bool = kwargs.pop("training_paging", False)
        self.optimizer_paging: bool = kwargs.pop("optimizer_paging", False)
        self.optimizer_page_dir: str = kwargs.pop("optimizer_page_dir", "./optimizer_pages")
        self.page_size_mb: int = kwargs.pop("page_size_mb", 64)
        self.activation_checkpointing: bool = kwargs.pop("activation_checkpointing", False)
        # Long-context training (0.3.2a1).  ce_chunk_size > 0 computes the LM
        # loss in row chunks from the final hidden state, never building the
        # [B, S, V] logits (0 = full logits, the historical path).
        self.ce_chunk_size: int = int(kwargs.pop("ce_chunk_size", 0) or 0)
        # attention_block_size > 0 runs MLX attention over long multi-row
        # queries in blocks that read only their visible keys (no [S, T] mask;
        # SWA costs O(S * window)).  Exact; 0 = one fused call (historical).
        self.attention_block_size: int = int(kwargs.pop("attention_block_size", 0) or 0)
        # Packed rows: when True, the temporal / router-locality / lookahead
        # regularizers drop token pairs that straddle a document separator,
        # so no routing penalty links two unrelated documents.
        self.reset_router_at_documents: bool = bool(kwargs.pop("reset_router_at_documents", False))
        # Packed rows: attention stays inside each document (separator-delimited).
        self.document_masking: bool = bool(kwargs.pop("document_masking", False))
        # Which blocks are recomputed when activation_checkpointing is on:
        # "all", "mla" (global-attention blocks), "swa", "alternate" (every second
        # block).  Selective policies trade some memory for less recompute time.
        self.recompute_policy: str = str(kwargs.pop("recompute_policy", "all") or "all").lower()
        if self.recompute_policy not in ("all", "mla", "swa", "alternate"):
            raise ValueError(f"unknown recompute_policy {self.recompute_policy!r}")
        # What activation_checkpointing recomputes: "block" (a whole block) or "mlp" (only the
        # MoE/FFN of each block: most saved activations, none of the quadratic attention).
        # MLX only; recompute_policy applies to scope=block.
        self.recompute_scope: str = str(kwargs.pop("recompute_scope", "block") or "block").lower()
        if self.recompute_scope not in ("block", "mlp"):
            raise ValueError(f"unknown recompute_scope {self.recompute_scope!r}")
        # MLA training with attention_block_size: keep only the latent c_kv and rebuild K and V
        # per attention block, in both passes (no [B, T, nKV, d] K/V tensors).  Exact.
        self.mla_upproj_recompute: bool = bool(kwargs.pop("mla_upproj_recompute", False))
        # MoE sparse dispatch in chunks of this many tokens, each recomputed in the
        # backward pass, so [N*K, I] expert activations never exist for a whole row
        # (0 = one dispatch, the historical path).
        self.moe_chunk_tokens: int = int(kwargs.pop("moe_chunk_tokens", 0) or 0)
        # Lookahead supervision's own document rule: None follows
        # reset_router_at_documents; True/False decouples it from the temporal and
        # router-locality terms.
        _lrd = kwargs.pop("reset_lookahead_at_documents", None)
        if isinstance(_lrd, str):      # web UI tri-state: "" follows the router switch
            _lrd = None if _lrd.strip() == "" else _lrd.strip().lower() in ("1", "true", "yes", "on")
        self.reset_lookahead_at_documents = None if _lrd is None else bool(_lrd)
        # Two-stage lookahead decoupling.  True: the LookaheadRouter reads a stop-gradient copy of
        # the hidden state, so its losses train only the router and add no gradient path
        # through the backbone (fine-tune the router on a frozen trunk).
        self.lookahead_input_detach: bool = bool(kwargs.pop("lookahead_input_detach", False))
        # > 0: the temporal / locality terms keep gradient only for the newer token of a pair
        # every this-many positions (blocks of that length, the older side is detached), so
        # no gradient chain crosses a whole row.  0 = full chain (historical).
        self.temporal_grad_span: int = int(kwargs.pop("temporal_grad_span", 0) or 0)
        # Long-context RoPE for training: {"type": "yarn"|"linear"|"ntk", "factor": f,
        # "original_max_position_embeddings": n, ...}; applied after the base init.
        _rts = kwargs.pop("rope_train_scaling", None)
        self.rope_train_scaling = dict(_rts) if _rts else None

        # Multimodal encoders are native kernel inputs, but are instantiated
        # only for modalities discovered/enabled by the data pipeline.
        raw_modalities = kwargs.pop("modalities", [])
        if isinstance(raw_modalities, str):
            raw_modalities = [x.strip() for x in raw_modalities.split(",") if x.strip()]
        self.modalities = sorted(set(str(x).lower() for x in raw_modalities if str(x).lower() != "text"))
        self.auto_detect_modalities: bool = kwargs.pop("auto_detect_modalities", True)
        self.image_size: int = kwargs.pop("image_size", 224)
        self.image_patch_size: int = kwargs.pop("image_patch_size", 16)
        self.max_image_tokens: int = kwargs.pop("max_image_tokens", 196)
        self.audio_sample_rate: int = kwargs.pop("audio_sample_rate", 16000)
        self.max_audio_seconds: float = kwargs.pop("max_audio_seconds", 30.0)
        self.max_audio_tokens: int = kwargs.pop("max_audio_tokens", 128)
        self.modality_dropout: float = kwargs.pop("modality_dropout", 0.05)
        self.freeze_modality_encoders: bool = kwargs.pop("freeze_modality_encoders", False)
        self.image_encoder_type: str = kwargs.pop("image_encoder_type", "patch")
        self.audio_encoder_type: str = kwargs.pop("audio_encoder_type", "conv")
        self.modality_fusion_heads: int = kwargs.pop("modality_fusion_heads", 4)
        self.media_error_policy: str = kwargs.pop("media_error_policy", "warn")
        self.modality_adapters: dict = dict(kwargs.pop("modality_adapters", {}))
        self.modality_alignment_coef: float = kwargs.pop("modality_alignment_coef", 0.0)
        self.extra_metadata: dict = dict(kwargs.pop("extra_metadata", {}))
        self.lookahead_steps: int = kwargs.pop("lookahead_steps", 2)
        self.num_shared_experts: int = kwargs.pop("num_shared_experts", 1)
        self.lambda_balance: float = kwargs.pop("lambda_balance", 5e-4)
        self.lambda_temporal: float = kwargs.pop("lambda_temporal", 3e-3)
        self.vram_budget_gb: float = kwargs.pop("vram_budget_gb", 4.0)
        self.prefetch_depth: int = kwargs.pop("prefetch_depth", 2)
        # Hybrid attention params
        self.use_hybrid_attention: bool = kwargs.pop("use_hybrid_attention", True)
        self.kv_latent_dim: int = kwargs.pop("kv_latent_dim", 64)   # MLA compression dim
        self.rope_dim: int = kwargs.pop("rope_dim", 32)              # RoPE portion of head_dim
        # How the rotated slice of each head is built.  "per_head": rotate the
        # rope_dim channels with their own frequency ladder (a true rotation,
        # relative-position exact).  "legacy_sliced": slice the first rope_dim
        # columns of the head_dim table (what checkpoints before 2026-11-06 were
        # trained with; kept so they load bit-identically).
        self.rope_layout: str = kwargs.pop("rope_layout", "legacy_sliced")
        if self.rope_layout not in ("per_head", "legacy_sliced"):
            raise ValueError(f"rope_layout must be 'per_head' or 'legacy_sliced', got {self.rope_layout!r}")
        self.sliding_window_size: int = kwargs.pop("sliding_window_size", 2048)
        # Inference KV storage: "auto" keeps the compute dtype; "int8" stores
        # MLA latents (and cached rope keys) / SWA K,V as per-token int8 with
        # one FP32 scale per ``kv_quant_group_size`` channels.  Training and
        # no-cache forwards never quantize.
        self.kv_cache_dtype: str = str(kwargs.pop("kv_cache_dtype", "auto") or "auto").lower()
        if self.kv_cache_dtype in ("fp16", "bf16", "fp32", "compute", "none"):
            self.kv_cache_dtype = "auto"     # the cache follows the compute dtype
        self.kv_quant_group_size: int = int(kwargs.pop("kv_quant_group_size", 32))
        # MLA decode path: "absorbed" folds W_uk into the query and W_uv into
        # the output (scores against the latent cache, no per-step K/V
        # up-projection of the whole prefix); "expanded" is the reference
        # path; "auto" absorbs for cached single-chunk decode (S <= 8) only.
        # Default "expanded": like every new option, absorption is opt-in.
        # "auto" mixes paths (expanded prefill, absorbed decode), which in
        # bf16 rounds differently from a full forward and can flip a near-tied
        # top-k route; numerics then differ from the pre-v3 kernel.
        self.mla_decode: str = str(kwargs.pop("mla_decode", "expanded") or "expanded").lower()
        # Also cache post-RoPE rope keys [B, T, nKV, rope_dim]: memory for
        # throughput (skips re-projecting + rotating the prefix each step).
        self.mla_cache_rope: bool = bool(kwargs.pop("mla_cache_rope", False))
        # Expert selection across recurrent iterations of one group:
        # "per_iteration" routes every iteration; "shared" reuses iteration
        # zero's top-k ids (weights still re-scored), so the active expert
        # union per group is at most top-k per token (lazy-loading friendly).
        self.router_scope: str = str(kwargs.pop("router_scope", "per_iteration") or "per_iteration").lower()
        # Self-speculative training hook: KL(full-depth || draft-depth) on the
        # LM head, teacher detached.  0 disables (no extra compute).
        self.draft_exit_kl_coef: float = float(kwargs.pop("draft_exit_kl_coef", kwargs.pop("draft_exit_kl", 0.0)) or 0.0)
        self.draft_exit_depth: int = int(kwargs.pop("draft_exit_depth", 1))
        # Hard expert-memory budget for lazy inference (None = heuristic).
        # max_resident_experts caps fast-tier (VRAM / Metal-hot) pages;
        # ram_budget_mb caps the RAM tier in MiB of expert pages.
        self.max_resident_experts = kwargs.pop("max_resident_experts", None)
        self.ram_budget_mb = kwargs.pop("ram_budget_mb", None)
        for name, value, allowed in (
            ("mla_decode", self.mla_decode, ("auto", "absorbed", "expanded")),
            ("router_scope", self.router_scope, ("per_iteration", "shared")),
            ("kv_cache_dtype", self.kv_cache_dtype, ("auto", "int8")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        # Pinned memory safety: max RAM for expert buffers (fraction of physical RAM)
        self.pinned_memory_max_fraction: float = kwargs.pop("pinned_memory_max_fraction", 0.25)
        # Storage backend for expert weights on SSD.
        # "safetensors" packs each Louvain cluster into one .ctsr file +
        # cluster_manifest.json (mmap-friendly, sequential reads).
        # "pt" keeps the legacy one-pickle-per-expert layout.
        self.storage_format: str = kwargs.pop("storage_format", "safetensors")
        # Optional path to a precomputed cluster_manifest.json. When set,
        # CacheManager will attach this layout instead of the one in ssd_dir.
        self.cluster_manifest_path = kwargs.pop("cluster_manifest_path", None)
        # M2: weight on the lookahead supervision loss
        self.lambda_lookahead: float = kwargs.pop("lambda_lookahead", 0.1)
        # Auxiliary hit-rate loss for the lookahead head. The soft CE above
        # learns distribution shape; this term directly rewards future top-k
        # expert-set recall, which is what offload prefetch needs.
        self.lambda_lookahead_topk: float = kwargs.pop("lambda_lookahead_topk", 0.15)
        self.lambda_lookahead_union: float = kwargs.pop("lambda_lookahead_union", 0.05)
        self.lambda_router_locality: float = kwargs.pop("lambda_router_locality", 0.02)
        # Cluster-locality constraint: within each ``cluster_locality_window``
        # tokens (0 = whole sequence) and each virtual layer (every loop
        # iteration), push routing mass into few expert clusters.  The
        # partition is ``chronos.expert_clusters.training_clusters`` and is the
        # one lazy inference pages by (``inference_clusters``: automatic when
        # λ > 0).  The class default stays 0 so a stored recipe or checkpoint
        # that predates the field keeps its training objective; every entry
        # point that creates a new training config (planner, presets, CLI,
        # task extensions) turns it on at the measured
        # ``chronos.expert_clusters.DEFAULT_LAMBDA``.
        self.lambda_cluster_locality: float = kwargs.pop("lambda_cluster_locality", 0.0)
        self.cluster_locality_clusters: int = kwargs.pop("cluster_locality_clusters", 0)
        self.cluster_locality_window: int = kwargs.pop("cluster_locality_window", 32)
        # Unit of the constraint: "window" (fixed windows above) or "segment"
        # (the batch's documents, read from the tokens' own separators; a
        # document longer than the window is cut into window-sized parts).
        # Neither consults labels or task tokens.
        self.cluster_locality_mode: str = str(kwargs.pop("cluster_locality_mode", "window") or "window")
        # M4: weight on the router KL anchor (alignment stages only).
        # Prevents SFT/DPO/ORPO/GRPO gradients from drifting routing away
        # from the pretrained distribution that the cluster layout was
        # optimized for. Typical values:
        #   pretrain: 0.0,  SFT: 0.01,  DPO/ORPO/GRPO: 0.1
        self.lambda_router_anchor: float = kwargs.pop("lambda_router_anchor", 0.0)
        # Optional training-time miss simulation. When >0, MoE layers randomly
        # route selected experts through the shared fallback while still adding
        # the shared residual, matching the SSD/DRAM offload inference formula.
        self.fallback_mask_prob: float = kwargs.pop("fallback_mask_prob", 0.08)
        # Runtime offload policy. "on_demand" is the predictive low-latency
        # path: cold SSD misses are queued for RAM and covered by shared
        # fallback this token. "sync_on_demand" is the diagnostic path that
        # synchronously loads the exact missed expert for Full-DRAM parity.
        # "fallback_diagnostic" keeps the pure shared-fallback stress path.
        self.offload_miss_policy_default: str = kwargs.pop("offload_miss_policy_default", "on_demand")
        self.recommended_resident_experts: int | None = kwargs.pop("recommended_resident_experts", None)
        self.offload_working_set_experts: int | None = kwargs.pop("offload_working_set_experts", None)
        # M7: regularization defaults. ChronosBaseConfig defaults dropout=0.0
        # and weight_decay isn't even a config field — both major causes
        # of overfitting on small corpora. Override here.
        kwargs.setdefault("dropout", 0.1)
        # Force MoE on
        kwargs.setdefault("use_moe", True)
        super().__init__(**kwargs)
        if self.rope_train_scaling is not None:
            self.rope_scaling = dict(self.rope_train_scaling)

        from chronos.config_schema import normalize_model_dimensions
        resolved = normalize_model_dimensions(vars(self))
        for key, value in resolved.items():
            setattr(self, key, value)

    @property
    def virtual_num_hidden_layers(self) -> int:
        """Executed attention depth, including repeated shared blocks."""
        if not self.recurrent_enabled:
            return int(self.num_hidden_layers)
        return (
            int(self.prelude_layers)
            + int(self.recurrent_groups) * int(self.recurrent_depth)
            + int(self.coda_layers)
        )
