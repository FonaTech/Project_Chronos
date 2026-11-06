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
import os
import subprocess
import sys
import threading
from collections import deque

from ui.gradio_compat import gr
from ui.i18n import register_translatable, t

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def build_serve_command(*, models_dir: str, host: str = "127.0.0.1", port: int = 11435,
                        device: str = "auto", dtype: str = "auto", lazy_experts: bool = True,
                        ram_expert_fraction: float = 0.5, allow_remote: bool = False,
                        api_key: str = "", kv_cache_dtype: str = "", mla_decode: str = "",
                        router_scope: str = "", max_resident_experts: int | None = None,
                        ram_budget_mb: float | None = None, monitor_runs_dir: str = "") -> tuple[list[str], dict]:
    """Return ``(argv, env_overrides)``; raises ValueError on an unsafe or invalid request."""
    host = (host or "127.0.0.1").strip()
    models_dir = os.path.expanduser((models_dir or "").strip())
    if not models_dir or not os.path.isdir(models_dir):
        raise ValueError("models_dir")
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError("port")
    remote = host not in LOOPBACK
    if remote and not (allow_remote and (api_key or "").strip()):
        raise ValueError("remote_needs_key")
    argv = [sys.executable, "-m", "chronos.serving.ollama_server", "--models-dir", models_dir,
            "--host", host, "--port", str(port), "--device", device or "auto", "--dtype", dtype or "auto",
            "--ram-expert-fraction", str(float(ram_expert_fraction))]
    if not lazy_experts:
        argv.append("--no-lazy-experts")
    if allow_remote:
        argv.append("--allow-remote")
    for flag, value in (("--kv-cache-dtype", kv_cache_dtype), ("--mla-decode", mla_decode),
                        ("--router-scope", router_scope)):
        if value and value != "checkpoint":
            argv += [flag, value]
    if max_resident_experts:
        argv += ["--max-resident-experts", str(int(max_resident_experts))]
    if ram_budget_mb:
        argv += ["--ram-budget-mb", str(float(ram_budget_mb))]
    if monitor_runs_dir and monitor_runs_dir.strip():
        argv += ["--monitor-runs-dir", os.path.expanduser(monitor_runs_dir.strip())]
    env = {"CHRONOS_API_KEY": api_key.strip()} if (api_key or "").strip() else {}
    return argv, env


def redact(argv: list[str]) -> str:
    return " ".join(argv[1:])            # the key is never in argv; drop the interpreter path


class ServeProcess:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: deque[str] = deque(maxlen=500)
        self.lock = threading.Lock()
        self.url = ""
        self.auth = False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _pump(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip())
        with self.lock:
            self.lines.append(f"[exit {proc.wait()}]")

    def start(self, argv: list[str], env_overrides: dict, host: str, port: int) -> str:
        if self.running():
            return "busy"
        env = dict(os.environ)
        env.pop("CHRONOS_API_KEY", None)        # only an explicit key from the form enables auth
        env.update(env_overrides)
        env["PYTHONUNBUFFERED"] = "1"
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.lines.clear()
        self.lines.append(f"$ {redact(argv)}")
        self.proc = subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        shown = f"[{host}]" if ":" in host else host
        self.url, self.auth = f"http://{shown}:{int(port)}", bool(env_overrides.get("CHRONOS_API_KEY"))
        return "started"

    def stop(self, timeout: float = 10.0) -> str:
        if not self.running():
            return "idle"
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return "stopped"

    def snapshot(self) -> tuple[bool, str]:
        with self.lock:
            return self.running(), "\n".join(self.lines)


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
