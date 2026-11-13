"""``chronos serve`` as a child process (argument list, no shell).

Security contract (same as the CLI): loopback by default; any other host
requires *allow remote* plus an API key.  The key is passed through the
``CHRONOS_API_KEY`` environment variable, never on the command line, so it
does not show up in ``ps``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def build_serve_command(*, models_dir: str, host: str = "127.0.0.1", port: int = 11435,
                        device: str = "auto", dtype: str = "auto", lazy_experts: bool = True,
                        ram_expert_fraction: float = 0.5, allow_remote: bool = False,
                        api_key: str = "", kv_cache_dtype: str = "", mla_decode: str = "",
                        router_scope: str = "", max_resident_experts: int | None = None,
                        ram_budget_mb: float | None = None, monitor_runs_dir: str = "",
                        paging: str = "auto", mla_cache_rope: bool = False) -> tuple[list[str], dict]:
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
    if paging not in ("auto", "cluster", "expert"):
        raise ValueError("paging")
    if not lazy_experts:
        argv.append("--no-lazy-experts")
    elif paging != "auto":
        argv += ["--paging", paging]
    if allow_remote:
        argv.append("--allow-remote")
    for flag, value in (("--kv-cache-dtype", kv_cache_dtype), ("--mla-decode", mla_decode),
                        ("--router-scope", router_scope)):
        if value and value != "checkpoint":
            argv += [flag, value]
    if mla_cache_rope:
        argv.append("--mla-cache-rope")
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
