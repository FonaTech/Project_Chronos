"""
ui/tabs/config_tab.py — ChronosConfig editor + live param estimator sidebar.

Merges the former Designer tab: the architecture sliders here are the
single source of truth; the right-hand "Estimator" panel reads the same
state and displays parameter counts, memory footprints and estimated
decode tokens/sec in real time.

Data-path is intentionally NOT in this tab — it lives in the Train tab
(and is per-stage in the Pipeline tab) so each workflow owns its own
dataset choice. Only save_dir is shared model-wide.
"""
from ui.gradio_compat import gr

from ui.i18n import t, register_translatable
from ui.estimator import (
    ArchConfig, total_params, active_params, memory_footprint,
    compute_params, active_param_bounds, parameter_breakdown, estimated_decode_tps, fmt_bytes, fmt_params,
)
from ui.planner_panel import build_planner_panel
from ui.presets import (
    CONFIG_INPUT_ORDER, DEFAULT_PRESET, BASE_MOE_DEFAULTS, preset_names, get_preset, values_in_input_order,
    save_config, load_config,
)


def _normalize_dtype_name(value: str | None) -> str:
    normalized = (value or "auto").strip().lower()
    return {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "full": "float32",
        "": "auto",
    }.get(normalized, normalized)


# Shared slider ranges (match Designer v1, which had the more permissive
# bounds). The old Config tab had stricter bounds that surprised users
# toggling between tabs and rejected values they typed manually.
RANGES = {
    "hidden_size":        (8, 16384, 8),
    "num_hidden_layers":  (1, 128, 1),
    "num_experts":        (1, 1024, 1),
    "experts_per_tok":    (1, 32, 1),
    "shared_experts":     (0, 32, 1),
    "lookahead_steps":    (0, 16, 1),
    "kv_latent_dim":      (4, 1024, 4),
    "rope_dim":           (2, 256, 2),
    "sliding_window":     (8, 65536, 8),
    "moe_intermediate":   (8, 65536, 8),
    "vocab_size":         (32, 200000, 32),
}


def _estimate(cfg_dict: dict):
    """Compute estimator outputs from a config dict.

    A moe_intermediate_size of 0 (or missing) is treated as "auto-derive
    from hidden_size", matching the ChronosBaseConfig default rule. This keeps
    the estimator consistent with what the trainer actually builds.
    """
    arch = ArchConfig.from_config(cfg_dict)
    seq_len = int(cfg_dict.get("max_seq_len", cfg_dict.get("max_position_embeddings", 1024)))
    mem = memory_footprint(arch, seq_len)
    low, high = active_param_bounds(arch)
    detail = {
        "parameters": parameter_breakdown(arch),
        "active_unique_min": low, "active_unique_max": high,
        "physical_blocks": arch.num_hidden_layers,
        "prelude_groups_coda": [arch.prelude_layers, arch.recurrent_groups, arch.coda_layers],
        "decode_depth_ceiling": arch.executed_depth,
        "weight_use_count": compute_params(arch),
        "inference_batch_size": 1, "context_tokens": seq_len,
        "dtype_assumption": arch.dtype if arch.dtype != "auto" else "float32 (auto upper-bound scenario)",
        "memory_bytes": mem,
        "training_peak_excludes": ["activations", "attention workspace", "rollback snapshots", "allocator pools"],
    }
    return (
        fmt_params(total_params(arch)),
        fmt_params(low) if low == high else f"{fmt_params(low)} – {fmt_params(high)}",
        fmt_params(compute_params(arch)),
        fmt_bytes(mem["vram_estimate_bytes"]), fmt_bytes(mem["ram_estimate_bytes"]),
        fmt_bytes(mem["ssd_estimate_bytes"]), fmt_bytes(mem["kv_cache_bytes"]),
        f"{estimated_decode_tps(arch):.1f}", detail,
    )


def merge_config_inputs(previous: dict, values: list) -> dict:
    """Apply editor fields by name while retaining loaded fields without widgets."""
    if len(values) != len(CONFIG_INPUT_ORDER):
        raise ValueError(f"Config editor expects {len(CONFIG_INPUT_ORDER)} values, received {len(values)}")
    cfg = dict(previous or BASE_MOE_DEFAULTS)
    for key, value in zip(CONFIG_INPUT_ORDER, values):
        if key == "modalities":
            value = value if isinstance(value, list) else str(value or "").split(",")
            value = sorted(set(m.strip().lower() for m in value if m.strip() and m.strip().lower() != "text"))
        elif key == "dtype":
            value = _normalize_dtype_name(value)
        else:
            default = BASE_MOE_DEFAULTS.get(key)
            if value is None:
                value = default
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
        cfg[key] = value
    # A loaded checkpoint/HF config may contain the canonical alias too.
    # Keep the visible sequence-length control authoritative when it is edited.
    if "max_position_embeddings" in cfg:
        cfg["max_position_embeddings"] = cfg["max_seq_len"]
    # The shared training widgets are the pretrain row of the per-stage table:
    # an edited widget must not be silently overridden by a stale plan value.
    stages = cfg.get("stage_overrides")
    if isinstance(stages, dict) and isinstance(stages.get("pretrain"), dict):
        pretrain = {k: (cfg[k] if k in CONFIG_INPUT_ORDER and k in cfg else v) for k, v in stages["pretrain"].items()}
        cfg["stage_overrides"] = {**stages, "pretrain": pretrain}
    return cfg


def build_config_tab():
    """Returns (config_state, list_of_input_components, save_dir).

    Note: the pre-M6 signature returned (state, inputs, data_path, save_dir).
    data_path has been moved to the Train tab, so the tuple now has 3
    elements. Callers must be updated.
    """
    initial_preset = DEFAULT_PRESET
    initial_cfg = get_preset(initial_preset)
    config_state = gr.State(dict(initial_cfg))

    with gr.Tab(t("tab.config")) as tab:
        register_translatable(tab, "tab.config")

        # ── Presets / Save / Load (top of tab) ─────────────────
        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=preset_names(),
                value=initial_preset,
                label=t("config.preset"),
                scale=2,
            )
            register_translatable(preset_dd, "config.preset")
            apply_preset_btn = gr.Button(t("config.load_preset"), variant="secondary", scale=1)
            register_translatable(apply_preset_btn, "config.load_preset")
            reset_btn = gr.Button(t("config.reset_minimind"), variant="secondary", scale=1)
            register_translatable(reset_btn, "config.reset_minimind")
        with gr.Row():
            cfg_path = gr.Textbox(
                value="./out/chronos_config.json",
                label=t("config.cfg_path"),
                scale=3,
            )
            register_translatable(cfg_path, "config.cfg_path")
            save_btn = gr.Button(t("config.save_cfg"), variant="primary", scale=1)
            register_translatable(save_btn, "config.save_cfg")
            load_btn = gr.Button(t("config.load_cfg"), variant="primary", scale=1)
            register_translatable(load_btn, "config.load_cfg")
        save_status = gr.Markdown("", visible=True)
        # Auto-plan accordion; filled in below once the editor outputs exist.
        planner_slot = gr.Column()

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"### {t('config.arch')}")

                D = initial_cfg

                with gr.Row():
                    hidden_size     = gr.Slider(*RANGES["hidden_size"],       value=D["hidden_size"],         label=t("config.hidden_size"))
                    num_layers      = gr.Slider(*RANGES["num_hidden_layers"], value=D["num_hidden_layers"],   label=t("config.num_layers"))
                    num_experts     = gr.Slider(*RANGES["num_experts"],       value=D["num_experts"],         label=t("config.num_experts"))
                    experts_per_tok = gr.Slider(*RANGES["experts_per_tok"],   value=D["num_experts_per_tok"], label=t("config.experts_per_tok"))
                    register_translatable(hidden_size,     "config.hidden_size")
                    register_translatable(num_layers,      "config.num_layers")
                    register_translatable(num_experts,     "config.num_experts")
                    register_translatable(experts_per_tok, "config.experts_per_tok")

                with gr.Row():
                    num_shared     = gr.Slider(*RANGES["shared_experts"],  value=D["num_shared_experts"],  label=t("config.shared_experts"))
                    lookahead      = gr.Slider(*RANGES["lookahead_steps"], value=D["lookahead_steps"],     label=t("config.lookahead"))
                    kv_latent_dim  = gr.Slider(*RANGES["kv_latent_dim"],   value=D["kv_latent_dim"],       label=t("config.kv_latent"))
                    sliding_window = gr.Slider(*RANGES["sliding_window"],  value=D["sliding_window_size"], label=t("config.sliding_window"))
                    register_translatable(num_shared,     "config.shared_experts")
                    register_translatable(lookahead,      "config.lookahead")
                    register_translatable(kv_latent_dim,  "config.kv_latent")
                    register_translatable(sliding_window, "config.sliding_window")

                with gr.Row():
                    num_heads        = gr.Slider(1, 256, value=D["num_attention_heads"], step=1, label=t("config.num_attn_heads"))
                    num_kv_heads     = gr.Slider(1, 256, value=D["num_key_value_heads"], step=1, label=t("config.num_kv_heads"))
                    rope_dim         = gr.Slider(*RANGES["rope_dim"],      value=D["rope_dim"], label=t("config.rope_dim"))
                    moe_intermediate = gr.Slider(
                        0, 65536, value=D["moe_intermediate_size"], step=8,
                        label=t("config.moe_inter"),
                    )
                    register_translatable(num_heads,        "config.num_attn_heads")
                    register_translatable(num_kv_heads,     "config.num_kv_heads")
                    register_translatable(rope_dim,         "config.rope_dim")
                    register_translatable(moe_intermediate, "config.moe_inter")

                with gr.Row():
                    vocab_size = gr.Number(value=D["vocab_size"], precision=0, label=t("config.vocab_size"))
                    dtype      = gr.Dropdown(
                        choices=["auto", "float32", "bfloat16", "float16"],
                        value=_normalize_dtype_name(D.get("dtype", "auto")), label=t("config.dtype"),
                    )
                    tie_word_embeddings = gr.Checkbox(value=D["tie_word_embeddings"], label=t("config.tie_emb"))
                    register_translatable(vocab_size,          "config.vocab_size")
                    register_translatable(dtype,               "config.dtype")
                    register_translatable(tie_word_embeddings, "config.tie_emb")

                gr.Markdown(f"### {t('config.loss')}")
                with gr.Row():
                    lambda_balance       = gr.Number(value=D["lambda_balance"],       label=t("config.lambda_balance"),  precision=6)
                    lambda_temporal      = gr.Number(value=D["lambda_temporal"],      label=t("config.lambda_temporal"), precision=6)
                    lambda_lookahead     = gr.Number(value=D["lambda_lookahead"],     label=t("config.lambda_la"),       precision=6)
                    lambda_lookahead_topk = gr.Number(value=D.get("lambda_lookahead_topk", 0.05),
                                                       label=t("config.lambda_la_topk"), precision=6)
                    lambda_router_anchor = gr.Number(value=D["lambda_router_anchor"], label=t("config.lambda_anchor"),   precision=6)
                    fallback_mask_prob   = gr.Slider(0.0, 1.0, value=D.get("fallback_mask_prob", 0.0),
                                                      step=0.01, label=t("config.fallback_mask"))
                    register_translatable(lambda_balance,        "config.lambda_balance")
                    register_translatable(lambda_temporal,       "config.lambda_temporal")
                    register_translatable(lambda_lookahead,      "config.lambda_la")
                    register_translatable(lambda_lookahead_topk, "config.lambda_la_topk")
                    register_translatable(lambda_router_anchor,  "config.lambda_anchor")
                    register_translatable(fallback_mask_prob,    "config.fallback_mask")
                with gr.Row():
                    lambda_cluster_locality = gr.Number(value=D.get("lambda_cluster_locality", 0.3), precision=4,
                                                        label=t("config.lambda_cluster"))
                    cluster_locality_clusters = gr.Number(value=D.get("cluster_locality_clusters", 0), precision=0,
                                                          label=t("config.cluster_count"))
                    cluster_locality_window = gr.Number(value=D.get("cluster_locality_window", 32), precision=0,
                                                        label=t("config.cluster_window"))
                    register_translatable(lambda_cluster_locality,   "config.lambda_cluster")
                    register_translatable(cluster_locality_clusters, "config.cluster_count")
                    register_translatable(cluster_locality_window,   "config.cluster_window")

                gr.Markdown(f"### {t('config.hw')}")
                with gr.Row():
                    vram_budget     = gr.Slider(0.01, 256.0, value=D["vram_budget_gb"],             step=0.01,  label=t("config.vram_budget"))
                    pinned_frac     = gr.Slider(0.01, 0.95,  value=D["pinned_memory_max_fraction"], step=0.01,  label=t("config.pinned_frac"))
                    use_hybrid_attn = gr.Checkbox(value=D["use_hybrid_attention"], label=t("config.hybrid_attn"))
                    storage_format  = gr.Dropdown(choices=["safetensors", "pt"], value=D["storage_format"],
                                                  label=t("config.storage_format"))
                    register_translatable(vram_budget,     "config.vram_budget")
                    register_translatable(pinned_frac,     "config.pinned_frac")
                    register_translatable(use_hybrid_attn, "config.hybrid_attn")
                    register_translatable(storage_format,  "config.storage_format")

                gr.Markdown(f"### {t('config.recurrent_section')}")
                with gr.Row():
                    recurrent_enabled = gr.Checkbox(value=bool(D.get("recurrent_enabled", True)), label=t("config.recurrent_enabled"))
                    recurrent_groups = gr.Slider(1, 128, value=int(D.get("recurrent_groups", 2)), step=1, label=t("config.recurrent_groups"))
                    recurrent_depth = gr.Slider(1, 16, value=int(D.get("recurrent_depth", 4)), step=1, label=t("config.recurrent_depth"))
                    min_recurrent_depth = gr.Slider(1, 16, value=int(D.get("min_recurrent_depth", 2)), step=1, label=t("config.min_recurrent_depth"))
                    adaptive_halting = gr.Checkbox(value=bool(D.get("adaptive_halting", True)), label=t("config.adaptive_halting"))
                with gr.Row():
                    halting_threshold = gr.Slider(0.5, 1.0, value=float(D.get("halting_threshold", .99)), step=.001, label=t("config.halting_threshold"))
                    max_inference_depth = gr.Slider(1, 32, value=int(D.get("max_inference_depth", D.get("recurrent_depth", 4))), step=1, label=t("config.max_inference_depth"))
                    ponder_loss_coef = gr.Number(value=float(D.get("ponder_loss_coef", .001)), label=t("config.ponder_loss_coef"), precision=6)
                    recurrent_distill_coef = gr.Number(value=float(D.get("recurrent_distill_coef", .05)), label=t("config.recurrent_distill_coef"), precision=6)
                    recurrent_stability_coef = gr.Number(value=float(D.get("recurrent_stability_coef", .001)), label=t("config.recurrent_stability_coef"), precision=6)
                    recurrent_latent_refinement = gr.Checkbox(
                        value=bool(D.get("recurrent_latent_refinement", True)),
                        label=t("config.recurrent_latent_refinement"),
                    )
                    depth_schedule = gr.Dropdown(
                        choices=["fixed_then_adaptive", "fixed", "adaptive"],
                        value=str(D.get("depth_schedule", "fixed_then_adaptive")), label=t("config.depth_schedule"),
                    )
                    recurrent_strategy = gr.Dropdown(
                        choices=["fixed", "scheduled", "adaptive"],
                        value=str(D.get("recurrent_strategy", "scheduled")),
                        label="recurrent strategy",
                    )
                    recurrent_depth_jitter = gr.Slider(
                        0.0, 1.0, value=float(D.get("recurrent_depth_jitter", 0.0)),
                        step=.05, label="depth jitter",
                    )
                with gr.Row():
                    depth_schedule_warmup_fraction = gr.Slider(
                        0.0, 1.0, value=float(D.get("depth_schedule_warmup_fraction", .25)), step=.01,
                        label=t("config.depth_schedule_warmup_fraction"),
                    )
                    layer_aware_cache = gr.Checkbox(value=bool(D.get("layer_aware_cache", True)), label=t("config.layer_aware_cache"))
                    prefetch_depth = gr.Slider(0, 64, value=int(D.get("prefetch_depth", 2)), step=1, label=t("config.prefetch_depth"))
                    training_paging = gr.Checkbox(value=bool(D.get("training_paging", False)), label=t("config.training_paging"))
                    optimizer_paging = gr.Checkbox(value=bool(D.get("optimizer_paging", False)), label=t("config.optimizer_paging"))
                    page_size_mb = gr.Number(value=int(D.get("page_size_mb", 64)), precision=0, label=t("config.page_size_mb"))
                    optimizer_page_dir = gr.Textbox(value=str(D.get("optimizer_page_dir", "./optimizer_pages")), label=t("config.optimizer_page_dir"))
                    modalities = gr.Textbox(value=",".join(D.get("modalities", [])), label=t("config.modalities"))

                gr.Markdown(f"### {t('config.train')}")
                with gr.Row():
                    learning_rate = gr.Number(value=D["learning_rate"], label=t("config.lr"), precision=6)
                    batch_size    = gr.Slider(1, 512,    value=D["batch_size"],         step=1,   label=t("config.batch_size"))
                    accum_steps   = gr.Slider(1, 256,    value=D["accumulation_steps"], step=1,   label=t("config.accum"))
                    max_seq_len   = gr.Slider(8, 131072, value=D["max_seq_len"],        step=8,   label=t("config.max_seq_len"))
                    register_translatable(learning_rate, "config.lr")
                    register_translatable(batch_size,    "config.batch_size")
                    register_translatable(accum_steps,   "config.accum")
                    register_translatable(max_seq_len,   "config.max_seq_len")

                with gr.Row():
                    epochs        = gr.Slider(1, 1000,    value=D["epochs"],         step=1,   label=t("config.epochs"))
                    max_steps     = gr.Number(value=int(D.get("max_steps", 0) or 0), precision=0, label=t("train.max_steps"))
                    save_interval = gr.Slider(10, 100000, value=D["save_interval"],  step=10,  label=t("config.save_interval"))
                    log_interval  = gr.Slider(1, 1000,    value=D.get("log_interval", 10), step=1, label="log_interval")
                    save_dir      = gr.Textbox(label=t("config.save_dir"),           value=D["save_dir"])
                    register_translatable(epochs,        "config.epochs")
                    register_translatable(max_steps,     "train.max_steps")
                    register_translatable(save_interval, "config.save_interval")
                    register_translatable(save_dir,      "config.save_dir")

                with gr.Row():
                    weight_decay = gr.Number(
                        value=float(D.get("weight_decay", 0.01)),
                        label=t("config.weight_decay"),
                        precision=6,
                    )
                    grad_clip = gr.Number(
                        value=float(D.get("grad_clip", 1.0)),
                        label="grad_clip",
                        precision=4,
                    )
                    cpu_threads = gr.Textbox(
                        value=str(D.get("cpu_threads", "auto")),
                        label=t("config.cpu_threads"),
                    )
                    cpu_budget_percent = gr.Slider(
                        1, 100,
                        value=float(D.get("cpu_budget_percent", 100)),
                        step=1,
                        label=t("config.cpu_budget_percent"),
                    )
                    num_workers = gr.Textbox(
                        value=str(D.get("num_workers", "auto")),
                        label=t("config.num_workers"),
                    )
                    register_translatable(weight_decay, "config.weight_decay")
                    register_translatable(cpu_threads, "config.cpu_threads")
                    register_translatable(cpu_budget_percent, "config.cpu_budget_percent")
                    register_translatable(num_workers, "config.num_workers")

                with gr.Accordion("Stage optimizer / rollout options", open=False):
                    with gr.Row():
                        reward_spec = gr.Textbox(
                            value=str(D.get("reward_spec", "toy")),
                            label="reward",
                            placeholder="toy or lm:/path/to/reward-model",
                        )
                        max_gen_len = gr.Slider(
                            1, 2048,
                            value=int(D.get("max_gen_len", 24)),
                            step=1,
                            label="max_gen_len",
                        )
                        num_generations = gr.Slider(
                            1, 64,
                            value=int(D.get("num_generations", 4)),
                            step=1,
                            label="num_generations",
                        )
                        temperature = gr.Number(
                            value=float(D.get("temperature", 1.0)),
                            label="temperature",
                            precision=4,
                        )
                    with gr.Row():
                        beta = gr.Number(value=float(D.get("beta", 0.1)), label="beta", precision=6)
                        alpha = gr.Number(value=float(D.get("alpha", 0.7)), label="alpha", precision=6)
                        lambda_or = gr.Number(value=float(D.get("lambda_or", 0.1)), label="lambda_or", precision=6)

                config_display = gr.JSON(label="Current Config (saved to config_state)", value=initial_cfg)

            with gr.Column(scale=1, min_width=260):
                gr.Markdown(f"### 🧬 {t('designer.title')}")
                total_box  = gr.Textbox(label=t("designer.total"),  interactive=False)
                active_box = gr.Textbox(label=t("designer.active"), interactive=False)
                compute_box = gr.Textbox(label=t("designer.compute"), interactive=False)
                vram_box   = gr.Textbox(label=t("designer.vram"),   interactive=False)
                ram_box    = gr.Textbox(label=t("designer.ram"),    interactive=False)
                ssd_box    = gr.Textbox(label=t("designer.ssd"),    interactive=False)
                kv_box     = gr.Textbox(label=t("designer.kv"),     interactive=False)
                tps_box    = gr.Textbox(label=t("designer.tps"),    interactive=False)
                gr.Markdown(t("designer.notes"))
                detail_box = gr.JSON(label=t("designer.details"))
                register_translatable(total_box,  "designer.total")
                register_translatable(active_box, "designer.active")
                register_translatable(compute_box, "designer.compute")
                register_translatable(vram_box,   "designer.vram")
                register_translatable(ram_box,    "designer.ram")
                register_translatable(ssd_box,    "designer.ssd")
                register_translatable(kv_box,     "designer.kv")
                register_translatable(tps_box,    "designer.tps")

        # IMPORTANT: this list also defines the order Autotune uses to push
        # best-params back via gr.update(). Don't reorder without updating
        # ui/tabs/autotune_tab.py.
        config_components = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "num_experts_per_tok": experts_per_tok,
            "num_shared_experts": num_shared,
            "lookahead_steps": lookahead,
            "kv_latent_dim": kv_latent_dim,
            "sliding_window_size": sliding_window,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "rope_dim": rope_dim,
            "moe_intermediate_size": moe_intermediate,
            "vocab_size": vocab_size,
            "dtype": dtype,
            "tie_word_embeddings": tie_word_embeddings,
            "lambda_balance": lambda_balance,
            "lambda_temporal": lambda_temporal,
            "lambda_lookahead": lambda_lookahead,
            "lambda_lookahead_topk": lambda_lookahead_topk,
            "lambda_router_anchor": lambda_router_anchor,
            "lambda_cluster_locality": lambda_cluster_locality,
            "cluster_locality_clusters": cluster_locality_clusters,
            "cluster_locality_window": cluster_locality_window,
            "fallback_mask_prob": fallback_mask_prob,
            "vram_budget_gb": vram_budget,
            "pinned_memory_max_fraction": pinned_frac,
            "use_hybrid_attention": use_hybrid_attn,
            "storage_format": storage_format,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "accumulation_steps": accum_steps,
            "max_seq_len": max_seq_len,
            "epochs": epochs,
            "max_steps": max_steps,
            "save_interval": save_interval,
            "log_interval": log_interval,
            "save_dir": save_dir,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "cpu_threads": cpu_threads,
            "cpu_budget_percent": cpu_budget_percent,
            "num_workers": num_workers,
            "reward_spec": reward_spec,
            "max_gen_len": max_gen_len,
            "num_generations": num_generations,
            "temperature": temperature,
            "beta": beta,
            "alpha": alpha,
            "lambda_or": lambda_or,
            "recurrent_enabled": recurrent_enabled,
            "recurrent_groups": recurrent_groups,
            "recurrent_depth": recurrent_depth,
            "min_recurrent_depth": min_recurrent_depth,
            "adaptive_halting": adaptive_halting,
            "halting_threshold": halting_threshold,
            "max_inference_depth": max_inference_depth,
            "ponder_loss_coef": ponder_loss_coef,
            "recurrent_distill_coef": recurrent_distill_coef,
            "recurrent_stability_coef": recurrent_stability_coef,
            "recurrent_latent_refinement": recurrent_latent_refinement,
            "depth_schedule": depth_schedule,
            "recurrent_strategy": recurrent_strategy,
            "recurrent_depth_jitter": recurrent_depth_jitter,
            "depth_schedule_warmup_fraction": depth_schedule_warmup_fraction,
            "layer_aware_cache": layer_aware_cache,
            "prefetch_depth": prefetch_depth,
            "training_paging": training_paging,
            "optimizer_paging": optimizer_paging,
            "page_size_mb": page_size_mb,
            "optimizer_page_dir": optimizer_page_dir,
            "modalities": modalities,
        }
        if set(config_components) != set(CONFIG_INPUT_ORDER):
            raise RuntimeError("Config component registry differs from CONFIG_INPUT_ORDER")
        all_inputs = [config_components[key] for key in CONFIG_INPUT_ORDER]
        estimator_outs = [total_box, active_box, compute_box, vram_box, ram_box, ssd_box, kv_box, tps_box, detail_box]

        def estimator_values(cfg):
            try:
                return list(_estimate(cfg))
            except (ValueError, TypeError, ZeroDivisionError) as exc:
                return ["—"] * 8 + [{"configuration_error": str(exc)}]

        def update_config(previous, *values):
            cfg = merge_config_inputs(previous, values)
            return [cfg, cfg] + estimator_values(cfg)

        # User input only: preset/load updates must not trigger 68 competing
        # change handlers that overwrite each other's state.
        for inp in all_inputs:
            inp.input(fn=update_config, inputs=[config_state] + all_inputs,
                      outputs=[config_state, config_display] + estimator_outs)

        # External producers (autotune) update the state as well as widgets.
        config_state.change(fn=lambda cfg: [cfg] + estimator_values(cfg),
                            inputs=[config_state], outputs=[config_display] + estimator_outs)

        config_outputs = [config_state, config_display] + all_inputs + estimator_outs

        def config_result(cfg):
            return [dict(cfg), dict(cfg)] + [gr.update(value=v) for v in values_in_input_order(cfg)] + estimator_values(cfg)

        def apply_preset(name):
            return config_result(get_preset(name))

        def reset_minimind():
            return config_result(get_preset("Chronos-v2 legacy (768H, 6400 vocab)"))

        def do_save(previous, path, *values):
            try:
                # Read the widgets at click time; a pending input event must
                # not cause Save to write a stale or default configuration.
                cfg = merge_config_inputs(previous, values)
                abs_path = save_config(cfg, path or "./out/chronos_config.json")
                return f"✅ Saved to `{abs_path}`"
            except Exception as exc:
                return f"❌ Save failed: {exc}"

        def do_load(path):
            try:
                cfg = load_config(path)
                return [f"✅ Loaded from `{path}`"] + config_result(cfg)
            except Exception as exc:
                return [f"❌ Load failed: {exc}"] + [gr.update() for _ in config_outputs]

        with planner_slot:
            build_planner_panel(config_state, config_result, config_outputs)

        apply_preset_btn.click(fn=apply_preset, inputs=[preset_dd], outputs=config_outputs)
        preset_dd.input(fn=apply_preset, inputs=[preset_dd], outputs=config_outputs)
        reset_btn.click(fn=reset_minimind, outputs=config_outputs)
        save_btn.click(fn=do_save, inputs=[config_state, cfg_path] + all_inputs, outputs=[save_status])
        load_btn.click(fn=do_load, inputs=[cfg_path], outputs=[save_status] + config_outputs)

        for box, value in zip(estimator_outs, estimator_values(initial_cfg)):
            box.value = value

    return config_state, all_inputs, save_dir
