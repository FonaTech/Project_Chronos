"""
ui/tabs/train_tab.py — legacy Gradio training tab (sessions: chronos.workbench.training)
"""
import os
import time

from ui.gradio_compat import gr

from ui.i18n import t, register_translatable, get_current_lang
from chronos.backend import training_available, resolve_training_device  # noqa: F401
import chronos.workbench.training as _training
from chronos.workbench.training import (  # noqa: F401  (re-exported for callers and tests)
    _available_train_backend_choices,
    _cleanup_gradio_temp_dataset,
    _cleanup_training_dataset_artifact,
    _close_dataset_handles,
    _coerce_max_steps,
    _cpu_budget_for_backend,
    _default_sample_prompts,
    _default_train_backend_value,
    _distill_teacher_placeholder,
    _effective_training_config,
    _extract_gradio_file_path,
    _format_effective_training_config,
    _gradio_temp_roots,
    _is_auto_value,
    _normalize_dtype_for_trainer,
    _normalize_stage_init_value,
    _path_in_gradio_temp,
    _path_in_training_upload_cache,
    _planned_epoch_iters,
    _planned_optimizer_updates,
    _planned_train_steps,
    _prepare_training_run_config,
    _project_root,
    _resolve_training_data_path,
    _sniff_checkpoint,
    _stage_gradio_upload_for_training,
    _stage_init_placeholder,
    _STAGE_OVERRIDE_KEYS,
    _stage_sample_prompt,
    _stage_text,
    _sync_metal_backend,
    _train_backend_dropdown_choices,
    _train_backend_label,
    _trainer_logger_sink,
    _training_upload_cache_root,
    _verify_checkpoint_warn,
    CHECKPOINT_TOPOLOGY_KEYS,
    DEFAULT_SAMPLE_PROMPT_VALUES,
    DEFAULT_STAGE_DATA_VALUES,
    DEFAULT_STAGE_INIT_VALUES,
    DEFAULT_STAGE_SAMPLE_PROMPTS,
    DISTILL_TEACHER_PLACEHOLDER,
    PRETRAIN_INIT_PLACEHOLDER,
    resolve_tokenizer_dir,
    STAGE_CHART_META,
    STAGE_CHECKPOINT_PREFIX,
    STAGE_DEFAULT_DATA,
    STAGE_DEFAULT_INIT,
    STAGE_HELP_TEXT,
    STAGE_LABELS,
    STAGE_METRIC_LABELS,
    STAGE_UI_ORDER,
    TRAIN_BACKEND_CHOICES,
    TRAIN_BACKEND_LABELS,
    TrainSession,
    UPSTREAM_INIT_PLACEHOLDER,
)

_training.set_language_provider(get_current_lang)
_session = TrainSession()


def _fmt_eta(seconds: float) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def _make_loss_chart(metrics: list[dict], mode: str = "pretrain", total_steps: int = 0, t_start: float = 0.0):
    """Build matplotlib figure: loss curves + throughput + progress/ETA."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        meta = STAGE_CHART_META.get(mode, STAGE_CHART_META["pretrain"])

        # Close any prior figures held by this thread to avoid the
        # "More than 20 figures opened" warning Gradio's Plot component triggers.
        plt.close("all")

        if not metrics:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="#888")
            ax.set_axis_off()
            plt.tight_layout()
            return fig

        steps = [m["step"] for m in metrics]
        cur_step = steps[-1]
        elapsed = max(0.0, time.monotonic() - t_start) if t_start else 0.0
        # ETA from average step rate so far (not last point — smoother).
        rate = cur_step / elapsed if elapsed > 0 else 0.0
        remaining = (total_steps - cur_step) / rate if rate > 0 and total_steps else float("nan")
        progress_pct = min(100.0, 100.0 * cur_step / total_steps) if total_steps else 0.0

        # M8b: split val markers from train metrics.
        train_metrics = [m for m in metrics if not m.get("val")]
        val_metrics   = [m for m in metrics if m.get("val")]
        train_steps = [m["step"] for m in train_metrics]
        val_steps   = [m["step"] for m in val_metrics]

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5),
                                 gridspec_kw={"width_ratios": [4, 3, 3]})
        fig.patch.set_facecolor("#1a1a2e")

        # ── Left: loss curves
        ax = axes[0]
        ax.set_facecolor("#16213e")
        if train_metrics:
            ax.plot(train_steps, [m["total"] for m in train_metrics],    color="#e94560", lw=1.5, label="train total")
            ax.plot(train_steps, [m.get("primary", m.get("ce", 0.0)) for m in train_metrics],
                    color="#0f3460", lw=1.2, label=meta["primary"])
            ax.plot(train_steps, [m.get("secondary", m.get("aux", 0.0)) for m in train_metrics],
                    color="#533483", lw=1.0, label=meta["secondary"])
            ax.plot(train_steps, [m.get("tertiary", m.get("temporal", 0.0)) for m in train_metrics],
                    color="#e2b714", lw=0.9, label=meta["tertiary"], ls="--")
        if val_metrics:
            ax.plot(val_steps, [m.get("primary", m.get("ce", 0.0)) for m in val_metrics], color="#39d98a",
                    lw=2.0, marker="o", markersize=4, label=meta["val"])
        ax.set_title(meta["title"], color="white", fontsize=11)
        ax.set_xlabel("Step", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=8, facecolor="#16213e", labelcolor="white")

        # ── Middle: stage-aware speed panel
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        tps_steps = [m["step"] for m in train_metrics if m.get("tps", 0) > 0]
        tps_vals  = [m["tps"]  for m in train_metrics if m.get("tps", 0) > 0]
        overlay_ax = None
        if mode == "grpo":
            reward_steps = [m["step"] for m in train_metrics if "tertiary" in m]
            reward_vals = [m.get("tertiary", 0.0) for m in train_metrics if "tertiary" in m]
            if tps_steps:
                ax2.plot(
                    tps_steps,
                    tps_vals,
                    color="#00b4d8",
                    lw=1.6,
                    label=meta.get("speed_label", "steps/s"),
                )
                ax2.fill_between(tps_steps, tps_vals, alpha=0.18, color="#00b4d8")
            if reward_steps:
                overlay_ax = ax2.twinx()
                overlay_ax.set_facecolor("none")
                overlay_ax.plot(
                    reward_steps,
                    reward_vals,
                    color="#e2b714",
                    lw=1.3,
                    ls="--",
                    label=meta.get("speed_overlay", meta["tertiary"]),
                )
                overlay_ax.tick_params(colors="#f1d36b")
                overlay_ax.set_ylabel(meta.get("speed_overlay", meta["tertiary"]), color="#f1d36b", fontsize=9)
                for spine in overlay_ax.spines.values():
                    spine.set_edgecolor("#333")
        elif tps_steps:
            ax2.plot(
                tps_steps,
                tps_vals,
                color="#00b4d8",
                lw=1.5,
                label=meta.get("speed_label", "steps/s"),
            )
            ax2.fill_between(tps_steps, tps_vals, alpha=0.2, color="#00b4d8")
        ax2.set_title(meta.get("speed_title", "Training Speed (steps/s)"), color="white", fontsize=11)
        ax2.set_xlabel("Step", color="#aaa")
        ax2.set_ylabel(meta.get("speed_label", "steps/s"), color="#8ae9ff", fontsize=9)
        ax2.tick_params(colors="#aaa")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")
        if tps_steps or overlay_ax is not None:
            lines, labels = ax2.get_legend_handles_labels()
            if overlay_ax is not None:
                extra_lines, extra_labels = overlay_ax.get_legend_handles_labels()
                lines += extra_lines
                labels += extra_labels
            if labels:
                ax2.legend(lines, labels, fontsize=8, facecolor="#16213e", labelcolor="white", loc="upper right")

        # ── Right: progress bar + numeric panel
        ax3 = axes[2]
        ax3.set_facecolor("#16213e")
        # Horizontal bar
        bar_y = 0.65
        ax3.barh([bar_y], [100], height=0.18, color="#2a2a44", edgecolor="#444")
        ax3.barh([bar_y], [progress_pct], height=0.18, color="#39d98a")
        ax3.set_xlim(0, 100); ax3.set_ylim(0, 1)
        ax3.set_xticks([0, 25, 50, 75, 100])
        ax3.set_yticks([])
        ax3.tick_params(colors="#aaa")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#333")

        # Big text overlay
        ax3.text(50, 0.92, f"Progress  {progress_pct:5.1f}%",
                 ha="center", va="center", color="white", fontsize=12, weight="bold")
        eta_str = _fmt_eta(remaining)
        elapsed_str = _fmt_eta(elapsed)
        ax3.text(50, 0.42, f"step {cur_step:,} / {total_steps or '?':,}",
                 ha="center", va="center", color="#cfd8e3", fontsize=11)
        ax3.text(50, 0.25, f"elapsed {elapsed_str}    ETA {eta_str}",
                 ha="center", va="center", color="#9aa6b2", fontsize=10)
        ax3.text(50, 0.10, f"~{rate:.2f} {meta.get('speed_label', 'steps/s')} avg",
                 ha="center", va="center", color="#9aa6b2", fontsize=9)
        ax3.set_title("Progress & ETA", color="white", fontsize=11)

        plt.tight_layout(pad=1.5)
        return fig

    except Exception:
        return None


def build_train_tab(config_state: gr.State, cfg_save_dir: gr.Textbox):
    with gr.Tab(t("tab.train")) as tab:
        register_translatable(tab, "tab.train")

        # ── Train owns its dataset choice (moved out of Config tab) ─
        stage_help = gr.Markdown(_stage_text(STAGE_HELP_TEXT, "pretrain"))
        with gr.Row():
            data_path = gr.Textbox(
                label=t("config.data_path"),
                value=STAGE_DEFAULT_DATA["pretrain"],
                placeholder=STAGE_DEFAULT_DATA["pretrain"],
                scale=3,
            )
            register_translatable(data_path, "config.data_path")
            dataset_upload = gr.File(
                label=t("train.dataset_upload"),
                file_types=[".jsonl"], type="filepath", scale=2,
            )
            register_translatable(dataset_upload, "train.dataset_upload")
        dataset_upload.change(
            fn=_stage_gradio_upload_for_training,
            inputs=dataset_upload,
            outputs=data_path,
        )

        # ── Mode + controls ───────────────────────────────────────
        with gr.Row():
            mode = gr.Radio(
                STAGE_UI_ORDER,
                value="pretrain", label=t("train.mode")
            )
            register_translatable(mode, "train.mode")
            train_backend = gr.Dropdown(
                choices=_train_backend_dropdown_choices(),
                value=_default_train_backend_value(),
                label=t("train.backend"),
                scale=1,
            )
            register_translatable(train_backend, "train.backend")
            status_box = gr.Textbox(
                value="idle", label=t("train.status"),
                interactive=False, scale=1,
            )
            register_translatable(status_box, "train.status")

        # ── Init-weight / stage-specific extras ────────────────────
        with gr.Row():
            init_weight = gr.Textbox(
                label=t("train.init_weight"),
                placeholder=_stage_init_placeholder("pretrain"),
                scale=3,
            )
            register_translatable(init_weight, "train.init_weight")
            teacher_path = gr.Textbox(
                label=t("pipeline.teacher_path"),
                placeholder=_distill_teacher_placeholder(),
                visible=False,
                scale=2,
            )
            register_translatable(teacher_path, "pipeline.teacher_path")

        with gr.Row():
            max_steps_in = gr.Textbox(
                value="from Config tab",
                label=t("train.max_steps"), scale=1,
                interactive=False,
            )
            val_ratio_in = gr.Slider(
                0.0, 0.5, value=0.05, step=0.01,
                label=t("train.val_ratio"), scale=2,
            )
            weight_decay_in = gr.Textbox(
                value="from Config tab",
                label=t("train.weight_decay"),
                interactive=False,
                scale=1,
            )
            register_translatable(max_steps_in,    "train.max_steps")
            register_translatable(val_ratio_in,    "train.val_ratio")
            register_translatable(weight_decay_in, "train.weight_decay")

        with gr.Row():
            sample_prompt_in = gr.Textbox(
                value=_stage_sample_prompt("pretrain"),
                label=t("train.sample_prompt"),
                placeholder=_stage_sample_prompt("pretrain"),
                scale=4,
            )
            register_translatable(sample_prompt_in, "train.sample_prompt")

        with gr.Row():
            start_btn = gr.Button(t("train.start"), variant="primary")
            stop_btn  = gr.Button(t("train.stop"),  variant="stop")
            clear_btn = gr.Button(t("train.clear_log"), variant="secondary")
            register_translatable(start_btn, "train.start")
            register_translatable(stop_btn,  "train.stop")
            register_translatable(clear_btn, "train.clear_log")

        # ── Metrics chart ─────────────────────────────────────────
        chart = gr.Plot(label=t("train.chart"), show_label=True)
        register_translatable(chart, "train.chart")

        # ── Scalar readouts ───────────────────────────────────────
        with gr.Row():
            step_box     = gr.Number(label=t("train.step"),     value=0,   interactive=False, precision=0)
            loss_box     = gr.Number(label=STAGE_METRIC_LABELS["pretrain"][0], value=0.0, interactive=False, precision=4)
            ce_box       = gr.Number(label=STAGE_METRIC_LABELS["pretrain"][1], value=0.0, interactive=False, precision=4)
            aux_box      = gr.Number(label=STAGE_METRIC_LABELS["pretrain"][2], value=0.0, interactive=False, precision=4)
            tps_box      = gr.Number(label=t("train.tps"),      value=0.0, interactive=False, precision=2)
            register_translatable(step_box, "train.step")
            register_translatable(tps_box,  "train.tps")

        # ── Scrollable log ────────────────────────────────────────
        log_box = gr.Textbox(
            label=t("train.log"), lines=18, max_lines=30,
            interactive=False, autoscroll=True,
        )
        register_translatable(log_box, "train.log")

        effective_config_box = gr.JSON(
            label="Effective Training Config",
            value={},
        )

        # ── Live generation sample (updated each save_interval) ───
        sample_box = gr.Textbox(
            label=t("train.sample_output"), lines=6,
            interactive=False, autoscroll=False,
        )
        register_translatable(sample_box, "train.sample_output")

        # ── Callbacks ─────────────────────────────────────────────

        def sync_stage_fields(train_mode, current_prompt, current_data_path, current_init_weight):
            current = (current_prompt or "").strip()
            default_upstream = STAGE_DEFAULT_INIT.get(train_mode, "")
            help_md = _stage_text(STAGE_HELP_TEXT, train_mode)
            data_default = STAGE_DEFAULT_DATA.get(train_mode, "")
            metric_labels = STAGE_METRIC_LABELS.get(train_mode, STAGE_METRIC_LABELS["pretrain"])
            init_placeholder = _stage_init_placeholder(train_mode)
            current_data = (current_data_path or "").strip()
            current_init = _normalize_stage_init_value(current_init_weight)
            data_value = current_data
            if (not current_data) or current_data in DEFAULT_STAGE_DATA_VALUES:
                data_value = data_default
            init_value = current_init
            teacher_visible = train_mode == "distill"
            if (not current) or current in DEFAULT_SAMPLE_PROMPT_VALUES:
                prompt_update = gr.update(
                    value=_stage_sample_prompt(train_mode),
                    placeholder=_stage_sample_prompt(train_mode),
                )
            else:
                prompt_update = gr.update(placeholder=_stage_sample_prompt(train_mode))
            return (
                prompt_update,
                gr.update(value=help_md),
                gr.update(value=data_value, placeholder=data_default),
                gr.update(value=init_value, placeholder=init_placeholder),
                gr.update(visible=teacher_visible, placeholder=_distill_teacher_placeholder()),
                gr.update(label=metric_labels[0]),
                gr.update(label=metric_labels[1]),
                gr.update(label=metric_labels[2]),
            )

        def start_training(cfg, train_mode, selected_backend, dpath, upload_file, iw, ms, vr, _wd_display, teacher, sp):
            if _session.is_running():
                return "already running", 0, 0.0, 0.0, 0.0, 0.0, "Already running.\n", None, "", _session.get_effective_config()
            cfg, initial_effective = _prepare_training_run_config(
                cfg,
                train_mode,
                selected_backend,
                dpath,
                upload_file,
                iw,
                ms,
                vr,
                teacher,
                sp,
            )
            _session.effective_config = initial_effective
            _session.start(cfg, train_mode)
            return "running", 0, 0.0, 0.0, 0.0, 0.0, "Starting...\n", None, "", initial_effective

        def stop_training():
            _session.stop()
            return "stopped"

        def clear_log():
            return ""

        def poll(current_log: str, current_mode: str):
            new_lines = _session.drain_log()
            # Append new lines to existing log; keep last 500 lines to avoid unbounded growth
            if new_lines:
                combined = (current_log + "\n" + new_lines).lstrip("\n")
            else:
                combined = current_log

            lines = combined.split("\n")
            if len(lines) > 500:
                combined = "\n".join(lines[-500:])

            metrics = _session.get_metrics()
            fig = _make_loss_chart(metrics, current_mode, _session.total_steps, _session.t_start)

            # Latest scalar values from last metric entry
            if metrics:
                last = metrics[-1]
                ce_v = last.get("primary", last.get("ce", 0.0))
                aux_v = last.get("secondary", last.get("aux", 0.0))
                tps_v = last.get("tps", 0.0)
            else:
                ce_v = aux_v = tps_v = 0.0

            return (
                _session.status,
                _session.step or 0,
                _session.loss or 0.0,
                ce_v,
                aux_v,
                tps_v,
                combined,
                fig,
                _session.last_sample or "",
                _session.get_effective_config(),
            )

        mode.change(
            fn=sync_stage_fields,
            inputs=[mode, sample_prompt_in, data_path, init_weight],
            outputs=[
                sample_prompt_in, stage_help, data_path, init_weight,
                teacher_path,
                loss_box, ce_box, aux_box,
            ],
        )

        start_btn.click(
            fn=start_training,
            inputs=[config_state, mode, train_backend, data_path, dataset_upload, init_weight,
                    max_steps_in, val_ratio_in, weight_decay_in,
                    teacher_path, sample_prompt_in],
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box,
                     log_box, chart, sample_box, effective_config_box],
        )
        stop_btn.click(fn=stop_training, outputs=[status_box])
        clear_btn.click(fn=clear_log, outputs=[log_box])

        timer = gr.Timer(value=2.0)
        timer.tick(
            fn=poll,
            inputs=[log_box, mode],
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box,
                     log_box, chart, sample_box, effective_config_box],
        )

    return tab
