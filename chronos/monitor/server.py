"""Standalone training-monitor web page + JSON API.

    chronos monitor --runs-dir runs                       # http://127.0.0.1:7870/
    chronos serve --models-dir ... --monitor-runs-dir runs   # same page at /monitor/

Security
* Default bind is 127.0.0.1 with **no authentication** (local use only); the
  page says so in its header.  Binding any other address requires
  ``--allow-remote`` *and* an API key.
* With a key, the browser logs in once (``POST api/login``) and receives an
  HttpOnly, SameSite=Strict session cookie; scripts may send
  ``Authorization: Bearer <key>`` instead.  Failed logins are rate-limited per
  client address.  Keys are compared in constant time and never stored in the
  page or the cookie.
* Read-only: no endpoint starts, stops or modifies a run.
"""
from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from collections import deque

try:  # module level: FastAPI resolves ``Request`` annotations by name
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
except ImportError:  # pragma: no cover - serving extra not installed
    FastAPI = HTTPException = Request = FileResponse = JSONResponse = PlainTextResponse = RedirectResponse = None

from chronos.monitor.snapshot import RunMonitor

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATIC_FILES = {"app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8"}
COOKIE = "chronos_monitor_session"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                                "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"),
}


class _Sessions:
    def __init__(self, ttl_s: float):
        self.ttl_s = ttl_s
        self._tokens: dict[str, float] = {}
        self._failures: dict[str, deque] = {}
        self._lock = threading.Lock()

    def issue(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.time()
            self._tokens = {t: exp for t, exp in self._tokens.items() if exp > now}
            self._tokens[token] = now + self.ttl_s
        return token

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            exp = self._tokens.get(token)
            if exp is None or exp < time.time():
                self._tokens.pop(token, None)
                return False
            return True

    def revoke(self, token: str | None) -> None:
        with self._lock:
            self._tokens.pop(token or "", None)

    def blocked(self, client: str, limit: int = 8, window_s: float = 300.0) -> bool:
        with self._lock:
            q = self._failures.setdefault(client, deque())
            while q and q[0] < time.time() - window_s:
                q.popleft()
            return len(q) >= limit

    def fail(self, client: str) -> None:
        with self._lock:
            self._failures.setdefault(client, deque()).append(time.time())
            if len(self._failures) > 4096:
                self._failures.clear()


def create_monitor_app(runs_dir: str, *, api_key: str | None = None, session_hours: float = 12.0,
                       monitor: RunMonitor | None = None):
    """FastAPI app serving the monitor page and its JSON API (mountable)."""
    if FastAPI is None:
        raise ImportError("chronos monitor needs fastapi and uvicorn (pip install 'Project_Chronos[serve]')")
    mon = monitor or RunMonitor(runs_dir)
    sessions = _Sessions(session_hours * 3600)
    app = FastAPI(title="Chronos training monitor", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    def _key_ok(given: str) -> bool:
        return bool(api_key) and hmac.compare_digest(given.encode("utf-8"), api_key.encode("utf-8"))

    def _authorized(request: Request) -> bool:
        if not api_key:
            return True
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and _key_ok(auth[7:]):
            return True
        return sessions.valid(request.cookies.get(COOKIE))

    def _guard(request: Request):
        if not _authorized(request):
            raise HTTPException(status_code=401, detail="login required")

    def _run_or_404(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html; charset=utf-8")

    @app.get("/static/{name}")
    def static(name: str):
        if name not in STATIC_FILES:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(os.path.join(STATIC_DIR, name), media_type=STATIC_FILES[name])

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/session")
    def session(request: Request):
        return {"auth_required": bool(api_key), "authenticated": _authorized(request),
                "local_only": not api_key}

    @app.post("/api/login")
    async def login(request: Request):
        client = request.client.host if request.client else "?"
        if not api_key:
            return {"authenticated": True, "auth_required": False}
        if sessions.blocked(client):
            raise HTTPException(status_code=429, detail="too many failed logins; wait 5 minutes")
        try:
            body = await request.json()
        except ValueError:
            body = {}
        key = body.get("key") if isinstance(body, dict) else None
        if not isinstance(key, str) or not _key_ok(key):
            sessions.fail(client)
            raise HTTPException(status_code=401, detail="invalid key")
        response = JSONResponse({"authenticated": True, "auth_required": True})
        response.set_cookie(COOKIE, sessions.issue(), max_age=int(sessions.ttl_s), httponly=True,
                            samesite="strict", secure=request.url.scheme == "https",
                            path=(request.scope.get("root_path") or "") + "/")
        return response

    @app.post("/api/logout")
    def logout(request: Request):
        sessions.revoke(request.cookies.get(COOKIE))
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(COOKIE, path=(request.scope.get("root_path") or "") + "/")
        return response

    @app.get("/api/runs")
    def runs(request: Request):
        _guard(request)
        return {"root": mon.root, "runs": mon.runs()}

    @app.get("/api/state")
    def state(request: Request, run: str | None = None, history: int = 1):
        _guard(request)
        return _run_or_404(mon.snapshot, run, history=bool(history))

    @app.get("/api/metrics")
    def metrics(request: Request, run: str | None = None, phase: str | None = None, since: int = 0, limit: int = 5000):
        _guard(request)
        return {"rows": _run_or_404(mon.metrics, run, phase, since, limit)}

    @app.get("/api/parameters")
    def parameters(request: Request, run: str | None = None):
        """Training parameters: run spec, model config, precision/runtime, data and tokenizer manifests."""
        _guard(request)
        return _run_or_404(mon.training_parameters, run)

    @app.get("/api/tensors")
    def tensors(request: Request, run: str | None = None):
        _guard(request)
        return _run_or_404(mon.parameters, run) or {"tensors": []}

    @app.get("/api/tensor")
    def tensor(request: Request, name: str, run: str | None = None):
        _guard(request)
        return {"name": name, "history": _run_or_404(mon.tensor_history, run, name)}

    @app.get("/api/log")
    def log(request: Request, run: str | None = None, bytes: int = 16000):  # noqa: A002 - query name
        _guard(request)
        return PlainTextResponse(_run_or_404(mon.log_tail, run, max(1000, min(int(bytes), 400_000))))

    return app


def check_bind(host: str, allow_remote: bool, api_key: str | None) -> str | None:
    """Error message when a bind would expose an unauthenticated monitor."""
    if host in LOOPBACK:
        return None
    if not (allow_remote and api_key):
        return "binding a non-loopback host requires --allow-remote and --api-key"
    return None


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="chronos monitor", description="Read-only Chronos training monitor page + API")
    parser.add_argument("--runs-dir", default="runs", help="directory holding run folders (run_spec.json, metrics.jsonl)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--api-key", default=os.environ.get("CHRONOS_MONITOR_KEY") or os.environ.get("CHRONOS_API_KEY"),
                        help="login key (env CHRONOS_MONITOR_KEY / CHRONOS_API_KEY); required off loopback")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--session-hours", type=float, default=12.0)
    parser.add_argument("--ssl-certfile", default=None, help="serve HTTPS (recommended with --allow-remote)")
    parser.add_argument("--ssl-keyfile", default=None)
    args = parser.parse_args(argv)
    problem = check_bind(args.host, args.allow_remote, args.api_key)
    if problem:
        parser.error(problem)
    import uvicorn
    app = create_monitor_app(args.runs_dir, api_key=args.api_key, session_hours=args.session_hours)
    scheme = "https" if args.ssl_certfile else "http"
    auth = "login required" if args.api_key else "no auth (loopback only)"
    print(f"[chronos monitor] {scheme}://{args.host}:{args.port}/  runs={os.path.abspath(args.runs_dir)}  {auth}", flush=True)
    if args.api_key and scheme == "http" and args.host not in LOOPBACK:
        print("[chronos monitor] warning: the login key crosses the network in clear text; use --ssl-certfile/--ssl-keyfile "
              "or a TLS reverse proxy", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
