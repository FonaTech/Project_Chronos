"""ui/tabs/serve_tab.py — start/stop ``chronos serve`` from the console.

The server runs as a child process (argument list, no shell) so it keeps its
own memory accounting and survives UI reloads only while this process lives.
Security contract (same as the CLI): loopback by default; any other host
requires *allow remote* plus an API key.  The key is passed through the
``CHRONOS_API_KEY`` environment variable, never on the command line, so it
does not show up in ``ps``.
"""
from __future__ import annotations

import atexit
from ui.gradio_compat import gr
from ui.i18n import register_translatable, t
from chronos.workbench.serve import (  # noqa: F401  (re-exported)
    LOOPBACK, ServeProcess, build_serve_command, redact,
)


_server = ServeProcess()
atexit.register(_server.stop)          # never leave an orphaned server behind the UI


def _status(lang: str | None = None) -> str:
    if not _server.running():
        return t("serve.state.stopped", lang)
    auth = t("serve.auth_on", lang) if _server.auth else t("serve.auth_off", lang)
    return f"{t('serve.state.running', lang)} · {_server.url} · {auth}"


def build_serve_tab():
    with gr.Tab(t("tab.serve")) as tab:
        register_translatable(tab, "tab.serve")
        intro = gr.Markdown(t("serve.intro"))
        register_translatable(intro, "serve.intro")
        with gr.Row():
            models_dir = gr.Textbox(label=t("serve.models_dir"), value="./out", scale=3)
            host = gr.Textbox(label=t("serve.host"), value="127.0.0.1", scale=1)
            port = gr.Number(label=t("serve.port"), value=11435, precision=0, scale=1)
        with gr.Row():
            device = gr.Dropdown(["auto", "cpu", "mps", "cuda"], value="auto", label=t("serve.device"))
            dtype = gr.Dropdown(["auto", "float32", "bfloat16", "float16"], value="auto", label=t("serve.dtype"))
            lazy = gr.Checkbox(value=True, label=t("serve.lazy"))
            ram_fraction = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label=t("serve.ram_fraction"))
        with gr.Row():
            kv_dtype = gr.Dropdown(["checkpoint", "auto", "int8"], value="checkpoint", label=t("serve.kv_dtype"))
            mla_decode = gr.Dropdown(["checkpoint", "auto", "absorbed", "expanded"], value="checkpoint",
                                     label=t("serve.mla_decode"))
            router_scope = gr.Dropdown(["checkpoint", "per_iteration", "shared"], value="checkpoint",
                                       label=t("serve.router_scope"))
            max_resident = gr.Number(label=t("serve.max_resident"), value=0, precision=0)
            ram_budget = gr.Number(label=t("serve.ram_budget"), value=0, precision=0)
        with gr.Row():
            allow_remote = gr.Checkbox(value=False, label=t("serve.allow_remote"))
            api_key = gr.Textbox(label=t("serve.api_key"), type="password", value="")
            monitor_dir = gr.Textbox(label=t("serve.monitor_dir"), value="", placeholder="./runs")
        warning = gr.Markdown(t("serve.warn_noauth"))
        with gr.Row():
            start_btn = gr.Button(t("serve.start"), variant="primary")
            stop_btn = gr.Button(t("serve.stop"), variant="stop")
        status = gr.Textbox(label=t("serve.status"), interactive=False, value=_status())
        log_box = gr.Textbox(label=t("serve.log"), lines=14, interactive=False, autoscroll=True)
        usage = gr.Markdown(t("serve.usage"))
        for comp, key in ((models_dir, "serve.models_dir"), (host, "serve.host"), (port, "serve.port"),
                          (device, "serve.device"), (dtype, "serve.dtype"), (lazy, "serve.lazy"),
                          (ram_fraction, "serve.ram_fraction"), (kv_dtype, "serve.kv_dtype"),
                          (mla_decode, "serve.mla_decode"), (router_scope, "serve.router_scope"),
                          (max_resident, "serve.max_resident"), (ram_budget, "serve.ram_budget"),
                          (allow_remote, "serve.allow_remote"), (api_key, "serve.api_key"),
                          (monitor_dir, "serve.monitor_dir"), (warning, "serve.warn_noauth"),
                          (start_btn, "serve.start"), (stop_btn, "serve.stop"), (status, "serve.status"),
                          (log_box, "serve.log"), (usage, "serve.usage")):
            register_translatable(comp, key)

        def start(md, h, p, dev, dt, lz, rf, kv, mla, rs, mr, rb, ar, key, mon):
            try:
                argv, env = build_serve_command(
                    models_dir=md, host=h, port=int(p or 11435), device=dev, dtype=dt, lazy_experts=bool(lz),
                    ram_expert_fraction=float(rf), allow_remote=bool(ar), api_key=key or "",
                    kv_cache_dtype=kv, mla_decode=mla, router_scope=rs,
                    max_resident_experts=int(mr or 0) or None, ram_budget_mb=float(rb or 0) or None,
                    monitor_runs_dir=mon or "")
            except ValueError as exc:
                return t(f"serve.err.{exc}"), gr.update()
            outcome = _server.start(argv, env, (h or "127.0.0.1").strip(), int(p or 11435))
            if outcome == "busy":
                return t("serve.err.busy"), gr.update()
            return _status(), _server.snapshot()[1]

        def stop():
            _server.stop()
            return _status(), _server.snapshot()[1]

        def poll():
            return _status(), _server.snapshot()[1]

        start_btn.click(fn=start, inputs=[models_dir, host, port, device, dtype, lazy, ram_fraction, kv_dtype,
                                          mla_decode, router_scope, max_resident, ram_budget, allow_remote,
                                          api_key, monitor_dir],
                        outputs=[status, log_box])
        stop_btn.click(fn=stop, outputs=[status, log_box])
        timer = gr.Timer(2.0)
        timer.tick(fn=poll, outputs=[status, log_box])
    return tab
