"""
ui/tabs/benchmark_tab.py — MiniMind vs Chronos comparison
"""
import json
import os

import gradio as gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable

RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../benchmark_results.json')


def build_benchmark_tab():
    with gr.Tab(t("tab.benchmark")) as tab:
        register_translatable(tab, "tab.benchmark")

        run_btn = gr.Button(t("bench.run"), variant="primary")
        register_translatable(run_btn, "bench.run")

        with gr.Row():
            results_box = gr.JSON(label=t("bench.results"))
            register_translatable(results_box, "bench.results")

        log_box = gr.Textbox(
            label=t("bench.log"), lines=20, interactive=False, autoscroll=True
        )
        register_translatable(log_box, "bench.log")

        # Load existing results on startup
        def load_existing():
            p = os.path.abspath(RESULTS_FILE)
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f), "Loaded previous results."
            return {}, "No results yet. Click Run."

        def run_benchmark():
            import subprocess, sys
            script = os.path.join(os.path.dirname(__file__), '../../benchmark_compare.py')
            log_lines = []
            yield {}, "Running benchmark... (this takes ~2 minutes)\n"
            try:
                proc = subprocess.Popen(
                    [sys.executable, script],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    log_lines.append(line.rstrip())
                    yield {}, "\n".join(log_lines[-50:])
                proc.wait()
                p = os.path.abspath(RESULTS_FILE)
                if os.path.exists(p):
                    with open(p) as f:
                        results = json.load(f)
                    yield results, "\n".join(log_lines)
                else:
                    yield {}, "\n".join(log_lines) + "\n[No results file found]"
            except Exception as e:
                yield {}, f"Error: {e}"

        run_btn.click(fn=run_benchmark, outputs=[results_box, log_box])

        # Load on tab render
        results_box.value, log_box.value = load_existing()

    return tab
