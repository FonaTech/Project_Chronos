"""
ui/tabs/autotune_tab.py — ChronosAutoTuner λ search via Optuna TPE
"""
import time

import gradio as gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable
from chronos.tuning.chronos_auto_tuner import ChronosAutoTuner, ChronosSearchSpaceConfig

_tuner = ChronosAutoTuner()


def build_autotune_tab(config_state: gr.State):
    with gr.Tab(t("tab.autotune")) as tab:
        register_translatable(tab, "tab.autotune")

        gr.Markdown("### Optuna TPE — Auto-search λ1, λ2, lookahead_steps")

        with gr.Row():
            n_trials    = gr.Slider(5, 100, value=20, step=5,  label=t("autotune.n_trials"))
            probe_steps = gr.Slider(20, 500, value=80, step=20, label=t("autotune.probe_steps"))
            register_translatable(n_trials,    "autotune.n_trials")
            register_translatable(probe_steps, "autotune.probe_steps")

        with gr.Row():
            tune_lambda_balance  = gr.Checkbox(value=True,  label="Search λ1 (balance)")
            tune_lambda_temporal = gr.Checkbox(value=True,  label="Search λ2 (temporal)")
            tune_lookahead       = gr.Checkbox(value=False, label="Search lookahead_steps")
            tune_lr              = gr.Checkbox(value=True,  label="Search learning rate")

        with gr.Row():
            start_btn = gr.Button(t("autotune.start"), variant="primary")
            stop_btn  = gr.Button(t("autotune.stop"),  variant="stop")
            register_translatable(start_btn, "autotune.start")
            register_translatable(stop_btn,  "autotune.stop")

        status_box = gr.Textbox(value="idle", label="Status", interactive=False)
        log_box    = gr.Textbox(label=t("autotune.log"), lines=15, interactive=False, autoscroll=True)
        best_box   = gr.JSON(label=t("autotune.best"))
        register_translatable(log_box,  "autotune.log")
        register_translatable(best_box, "autotune.best")

        def start_tune(cfg, n_tr, p_steps, tlb, tlt, tla, tlr):
            if _tuner.is_running():
                return "already running", "", {}

            ss = ChronosSearchSpaceConfig(
                tune_lambda_balance=tlb,
                tune_lambda_temporal=tlt,
                tune_lookahead_steps=tla,
                tune_lr=tlr,
            )
            data_path = cfg.get("data_path", "")
            if not data_path:
                return "error: set data_path in Config tab first", "", {}

            import os
            if not os.path.exists(data_path):
                return f"error: data_path not found: {data_path}", "", {}

            _tuner.start(
                model_id=cfg.get("save_dir", "./out") + f"/chronos_{cfg.get('hidden_size',512)}_moe.pth",
                dataset_path=data_path,
                train_ratio=0.95,
                prompt_template="alpaca",
                think_mode="keep",
                search_space=ss,
                n_trials=int(n_tr),
                probe_steps=int(p_steps),
                output_dir="./auto_tune_cache",
                seed=42,
            )
            return "running", "Auto-tune started...\n", {}

        def stop_tune():
            _tuner.stop()
            return "stopped"

        def poll_tune():
            events = _tuner.poll()
            log_lines = []
            for ev in events:
                if ev.get("type") == "log":
                    log_lines.append(ev.get("line", ""))
            best = {}
            if _tuner.result and _tuner.result.best_params:
                best = _tuner.result.best_params
            return _tuner.status, "\n".join(log_lines), best

        start_btn.click(
            fn=start_tune,
            inputs=[config_state, n_trials, probe_steps,
                    tune_lambda_balance, tune_lambda_temporal,
                    tune_lookahead, tune_lr],
            outputs=[status_box, log_box, best_box],
        )
        stop_btn.click(fn=stop_tune, outputs=[status_box])

        timer = gr.Timer(value=2.0)
        timer.tick(fn=poll_tune, outputs=[status_box, log_box, best_box])

    return tab
