"""Chronos Web UI: FastAPI JSON API + a hand-written single-page front end.

    chronos webui                       # http://127.0.0.1:7860/
    chronos webui --lan                 # also reachable from the local network (login for other machines)
    chronos webui --legacy              # the old Gradio console

Security
* Default bind is 127.0.0.1 with **no authentication** (local use only); the
  page says so.  Any other address needs ``--allow-remote`` *and* a key in
  ``CHRONOS_UI_KEY`` / ``CHRONOS_API_KEY`` (never on the command line).
* With a key, the browser logs in once and receives an HttpOnly,
  SameSite=Strict session cookie; scripts may send ``Authorization: Bearer``.
  Failed logins are rate-limited per client; keys are compared in constant time.
* Every request must carry an allowed ``Host`` (DNS-rebinding guard); every
  mutating request must carry ``X-Chronos-Request: 1`` and, when present, an
  ``Origin`` on the same allow-list (CSRF guard).  The page is served with a
  strict CSP (``script-src 'self'``, no inline script).
* ``--lan`` (off by default) listens on every interface and accepts this
  machine's LAN addresses as Host. Browsers on other machines must log in (the key
  comes from ``CHRONOS_UI_KEY`` or is generated and printed at start); a browser
  on this machine (loopback peer *and* loopback Host, no proxy headers) needs no
  login. Do not put ``--lan`` behind a reverse proxy; use ``--allow-remote``.
* File fields: on this machine the server opens the system file dialog and the
  chosen paths are used in place. From another machine files are uploaded
  (streamed, size-capped, disk reserve kept) into ``<workdir>/uploads/<kind>/``.
* This UI starts training, inference, export and servers and reads and writes
  files the server user can access: treat access to it as shell access.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import sys
import time

try:  # module level: FastAPI resolves ``Request`` annotations by name
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:  # pragma: no cover - serving extra not installed
    FastAPI = HTTPException = Request = FileResponse = JSONResponse = run_in_threadpool = None

from chronos.monitor.server import LOOPBACK, SECURITY_HEADERS, _Sessions, check_bind
from chronos.webui import schema
from chronos.webui.jobs import JobBusy, JobRegistry

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
I18N_DIR = os.path.join(HERE, "i18n")
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
COOKIE = "chronos_ui_session"
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml"}
_STATIC_NAME = re.compile(r"^[a-z0-9_\-]+\.(js|css|svg)$")
_LANG_ALIASES = {"zh-tw": "zh-Hant", "zh-hk": "zh-Hant", "zh-mo": "zh-Hant", "zh-hant": "zh-Hant"}
MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
STAGE_ORDER = ("pretrain", "sft", "dpo", "orpo", "grpo", "distill")


def resolve_language(tag: str | None) -> str:
    """Map a BCP-47 tag (or Accept-Language list) to a supported catalog; English otherwise."""
    for part in (tag or "").split(","):
        code = part.split(";")[0].strip().lower().replace("_", "-")
        if not code:
            continue
        if code in _LANG_ALIASES or code.startswith("zh-hant"):
            return "zh-Hant"
        if code == "zh" or code.startswith("zh-"):
            return "zh-Hans"
        if code == "ja" or code.startswith("ja-"):
            return "ja"
        if code == "en" or code.startswith("en-"):
            return "en"
    return schema.DEFAULT_LANGUAGE


def load_catalog(lang: str) -> dict:
    if lang not in schema.LANGUAGES:
        raise KeyError(lang)
    with open(os.path.join(I18N_DIR, f"{lang}.json"), encoding="utf-8") as fh:
        return json.load(fh)


MAX_UPLOAD_MB = 20480                     # per file; --max-upload-mb changes it
UPLOAD_DISK_RESERVE = 8 * 2**30           # uploads stop while less than this (or 5 % of the disk) stays free
LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "localhost"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def lan_addresses() -> list[str]:
    """This machine's IPv4 addresses other than loopback (what a LAN browser types)."""
    import socket
    out = []
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            out += [a.address for a in addrs if a.family == socket.AF_INET and not a.address.startswith("127.")]
    except Exception:  # pragma: no cover - psutil is a dependency; keep a fallback anyway
        try:
            out = [a for a in socket.gethostbyname_ex(socket.gethostname())[2] if not a.startswith("127.")]
        except OSError:
            out = []
    return sorted(set(out), key=lambda a: (not a.startswith(("192.168.", "10.")), a))


def allowed_hosts(bind_host: str, extra: tuple[str, ...] = ()) -> set[str]:
    """Host header values (without port) the server answers to."""
    hosts = {"127.0.0.1", "localhost", "::1", "[::1]"}
    if bind_host and bind_host not in ("0.0.0.0", "::"):
        hosts.add(bind_host.strip("[]"))
        hosts.add(f"[{bind_host.strip('[]')}]" if ":" in bind_host else bind_host)
    hosts.update(h.strip().lower() for h in extra if h.strip())
    return hosts


def _host_only(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("["):
        return value[: value.find("]") + 1] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


# ── config sanitising ──────────────────────────────────────────────────
def _plain_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= 4096)


def _small_json(value, limit: int = 16_384) -> bool:
    try:
        return len(json.dumps(value, allow_nan=False)) <= limit
    except (TypeError, ValueError):
        return False


def sanitize_config(values: dict) -> tuple[dict, list[str]]:
    """Config sent by the page → ``(clean, bad_keys)``.

    Schema fields are validated by :func:`schema.coerce`; ``modalities`` is a
    list (comma text in the form); other canonical Chronos fields (planner or
    preset values such as ``use_moe``) pass when they are small JSON values;
    ``stage_overrides`` is ``{stage: {key: scalar}}``; anything else is dropped."""
    from chronos.config_schema import CANONICAL_FIELDS
    from chronos.model.planner import STAGE_KEYS, STAGES
    values = dict(values or {})
    modalities = values.pop("modalities", None)
    clean, bad = schema.coerce_config(values)
    if modalities is not None:
        items = modalities if isinstance(modalities, list) else str(modalities).split(",")
        names = sorted({str(m).strip().lower() for m in items if str(m).strip()} - {"text"})
        if all(n in ("image", "audio") for n in names):
            clean["modalities"] = names
        else:
            bad.append("modalities")
    for key, value in values.items():
        if key in clean or key in bad:
            continue
        if key == "stage_overrides" and isinstance(value, dict):
            clean[key] = {stage: {k: v for k, v in (row or {}).items() if k in STAGE_KEYS and _plain_scalar(v)}
                          for stage, row in value.items() if stage in STAGES and isinstance(row, dict)}
        elif key in CANONICAL_FIELDS and key not in schema.FIELDS and _small_json(value):
            clean[key] = value
    return clean, bad


def form_values(body: dict, prefix: str) -> tuple[dict, list[str]]:
    """``{"<prefix>.<name>": value}`` from a page form → ``({name: value}, bad_keys)``; defaults fill gaps."""
    out, bad = {}, []
    given = body or {}
    for f in schema.FORM_FIELDS:
        if not f.key.startswith(prefix + "."):
            continue
        name = f.key[len(prefix) + 1:]
        raw = given.get(f.key, given.get(name, f.default))
        try:
            out[name] = schema.coerce(f.key, raw)
        except ValueError:
            bad.append(f.key)
    return out, bad


def probe_sweep(values: dict) -> dict:
    """``probe.*`` form values → a ``ProbeSweep`` dict for the train worker."""
    return {
        "backends": values["backends"], "paging": values["paging"], "residency": values["residency"],
        "ram_ratios": [float(r) for r in values["ram_ratios"]], "miss_policies": values["miss_policies"],
        "residency_decay": [values["residency_decay"]], "decode_wait_s": [values["decode_wait_s"]],
        "verify_wait_s": [values["verify_wait_s"]], "speculative": [bool(s) for s in values["speculative"]],
        "draft_depth": values["draft_depth"], "gamma": values["gamma"],
        "max_new_tokens": values["max_new_tokens"], "max_variants": values["max_variants"],
    }


def config_estimate(cfg: dict) -> dict:
    """Size / memory estimate for the Config page (``ui.estimator``; estimated, not measured)."""
    from ui.estimator import (
        ArchConfig, active_param_bounds, compute_params, estimated_decode_tps, memory_footprint,
        parameter_breakdown, total_params,
    )
    arch = ArchConfig.from_config(cfg)
    lo, hi = active_param_bounds(arch)
    groups = int(arch.recurrent_groups or 0)
    depth = int(arch.executed_depth) if arch.recurrent_enabled else 1
    virtual = (int(arch.prelude_layers) + groups * depth + int(arch.coda_layers)
               if arch.recurrent_enabled and groups else int(arch.num_hidden_layers))
    return {"label": "estimated", "total_params": total_params(arch), "active_params": lo,
            "active_union_upper": hi, "compute_params": compute_params(arch),
            "breakdown": parameter_breakdown(arch), "memory": memory_footprint(arch),
            "decode_tps_estimate": round(float(estimated_decode_tps(arch)), 1), "virtual_layers": virtual}


def plan_request(values: dict):
    """``plan.*`` form values → ``PlanRequest`` (millions / billions in the form); ValueError(err key)."""
    from chronos.model.planner import MEMORY_LEVELS, PlanRequest
    from chronos.workbench.tokenizer import parse_paths
    target = float(values.get("target_m") or 0) * 1e6
    if target <= 0:
        raise ValueError("target")
    use_moe = bool(values.get("use_moe"))
    value = float(values.get("active_value") or 0)
    experts = int(values.get("num_experts") or 1)
    active_params = value * 1e6 if use_moe and values.get("active_mode") == "params" and value > 0 else None
    active_experts = int(value) if use_moe and values.get("active_mode") == "experts" and value >= 1 else None
    if active_experts and active_experts > experts:
        raise ValueError("active_experts")
    return PlanRequest(
        target_total=target, use_moe=use_moe, num_experts=experts, active_params=active_params,
        active_experts=active_experts,
        memory_saving=values.get("memory_saving") if values.get("memory_saving") in MEMORY_LEVELS else "none",
        data_paths=tuple(parse_paths(values.get("data_paths") or "")),
        ram_gb=float(values["ram_gb"]) if values.get("ram_gb") else None,
        token_budget=float(values["token_budget_b"]) * 1e9 if values.get("token_budget_b") else None,
        recurrent_depth=int(values.get("depth") or 4),
        vocab_size=int(values["vocab_size"]) if values.get("vocab_size") else None,
        seq_len=int(values["seq_len"]) if values.get("seq_len") else None,
    )


def system_info() -> dict:
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform(terse=True),
            "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        from chronos import __version__
        info["chronos"] = __version__
    except Exception:                                   # pragma: no cover - partial install
        info["chronos"] = "?"
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        info["ram_gb"] = None
    backends = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            backends.append("cuda")
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            backends.append("xpu")
        if torch.backends.mps.is_available():
            backends.append("mps")
    except Exception:
        pass
    try:
        import mlx.core  # noqa: F401
        backends.append("mlx")
    except Exception:
        pass
    info["backends"] = backends
    return info


def read_presets() -> dict:
    """Planner presets as complete config dicts (``ui.presets`` merges each over the base defaults)."""
    from ui.presets import DEFAULT_PRESET, V3_PRESET_META, V3_PRESETS
    presets = [{"name": name, "config": sanitize_config(cfg)[0], "meta": V3_PRESET_META.get(name, {})}
               for name, cfg in V3_PRESETS.items()]
    names = [p["name"] for p in presets]
    return {"default": DEFAULT_PRESET if DEFAULT_PRESET in names else (names[0] if names else ""),
            "presets": presets}


def base_config() -> dict:
    """What an empty Config page starts from: the default planner preset, else the schema defaults."""
    data = read_presets()
    by_name = {p["name"]: p["config"] for p in data["presets"]}
    return {**schema.config_defaults(), **by_name.get(data["default"], {})}


def _json_safe(value):
    """Plain JSON types (unknown objects → str); NaN / inf become None via :func:`_finite`."""
    return _finite(json.loads(json.dumps(value, default=str))) if value is not None else None


def _finite(value):
    """Replace NaN / inf (not valid JSON) with None, recursively."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


class UIState:
    """Everything one server instance owns: jobs, the serve child, the tokenizer job, the tuner."""

    def __init__(self, workdir: str, runs_dir: str):
        from chronos.monitor.snapshot import RunMonitor
        from chronos.workbench.serve import ServeProcess
        from chronos.workbench.tokenizer import TokenizerJob
        self.workdir = os.path.abspath(workdir)
        self.runs_dir = os.path.abspath(os.path.join(self.workdir, runs_dir))
        os.makedirs(self.runs_dir, exist_ok=True)
        self.jobs = JobRegistry()
        self.serve = ServeProcess()
        self.tokenizer = TokenizerJob()
        self.monitor = RunMonitor(self.runs_dir)
        self.tuner = None
        self.tune_log: list[str] = []

    def path(self, value: str) -> str:
        """User path relative to the work directory, ``~`` expanded."""
        value = os.path.expanduser((value or "").strip())
        return value if not value or os.path.isabs(value) else os.path.join(self.workdir, value)

    def paths(self, value) -> list[str]:
        """Several user paths (one per line, or a list), each resolved like :meth:`path`."""
        from chronos.workbench.files import parse_paths
        return [self.path(p) for p in parse_paths(value)]

    def shutdown(self) -> None:
        self.jobs.stop_all()
        self.serve.stop(timeout=5.0)
        if self.tuner is not None and self.tuner.is_running():
            self.tuner.stop()


def create_app(workdir: str = ".", *, runs_dir: str = "runs", api_key: str | None = None,
               bind_host: str = "127.0.0.1", extra_hosts: tuple[str, ...] = (), session_hours: float = 12.0,
               max_upload_mb: int = MAX_UPLOAD_MB, trust_local: bool = False, native_dialog: bool = True):
    """The Web UI FastAPI app; ``app.state.ui`` holds the :class:`UIState`."""
    if FastAPI is None:
        raise ImportError("chronos webui needs fastapi and uvicorn (pip install 'Project_Chronos[serve]')")
    ui = UIState(workdir, runs_dir)
    sessions = _Sessions(session_hours * 3600)
    hosts = allowed_hosts(bind_host, extra_hosts)
    app = FastAPI(title="Chronos Web UI", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ui = ui
    app.add_event_handler("shutdown", ui.shutdown)
    public = {"/", "/health", "/api/session", "/api/login", "/api/logout", "/api/i18n"}

    def _key_ok(given: str) -> bool:
        return bool(api_key) and hmac.compare_digest(given.encode("utf-8"), api_key.encode("utf-8"))

    def _authorized(request) -> bool:
        if not api_key or (trust_local and _is_local(request)):
            return True
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and _key_ok(auth[7:]):
            return True
        return sessions.valid(request.cookies.get(COOKIE))

    def _is_local(request) -> bool:
        """The browser runs on this machine: loopback peer *and* a loopback Host (not a LAN URL)."""
        peer = request.client.host if request.client else ""
        if any(h in request.headers for h in ("forwarded", "x-forwarded-for", "x-real-ip")):
            return False                           # came through a proxy: the real browser is elsewhere
        return peer in LOOPBACK_PEERS and _host_only(request.headers.get("host", "")) in LOOPBACK_HOSTS

    def _deny(status: int, detail: str):
        response = JSONResponse({"detail": detail}, status_code=status)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if _host_only(request.headers.get("host", "")) not in hosts:
            return _deny(421, "unexpected Host header")
        path = request.url.path
        if request.method in MUTATING:
            if request.headers.get("x-chronos-request") != "1":
                return _deny(403, "missing X-Chronos-Request header")
            origin = request.headers.get("origin")
            if origin and _host_only(origin.split("://", 1)[-1].split("/", 1)[0]) not in hosts:
                return _deny(403, "cross-origin request")
        if path.startswith("/api/") and not (path in public or path.startswith("/api/i18n/")) \
                and not _authorized(request):
            return _deny(401, "login required")
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    async def _body(request) -> dict:
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid JSON") from None
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="expected a JSON object")
        return data

    def _check_data(paths: list[str], key: str) -> None:
        """Every selected dataset path exists and at least one readable data file is behind them."""
        from chronos.workbench.files import data_files
        if not paths:
            return
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            _bad([key], "not_found", missing[0])
        if not data_files(paths):
            _bad([key], "no_files")

    def _bad(keys, code: str = "invalid", message: str | None = None):
        detail = {"error": code, "fields": list(keys)}
        if message:
            detail["message"] = message
        raise HTTPException(status_code=422, detail=detail)

    # ── page, session, catalogs ────────────────────────────────────────
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"), media_type="text/html; charset=utf-8")

    @app.get("/static/{name}")
    def static(name: str):
        ext = os.path.splitext(name)[1]
        full = os.path.join(STATIC_DIR, name)
        if not _STATIC_NAME.match(name) or not os.path.isfile(full):
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(full, media_type=STATIC_TYPES[ext])

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/session")
    def session(request: Request):
        from chronos.webui.native_dialog import backend
        local = _is_local(request)
        return {"auth_required": bool(api_key) and not (trust_local and local), "authenticated": _authorized(request),
                "local_only": not api_key, "language": resolve_language(request.headers.get("accept-language")),
                "access": "local" if local else "lan", "lan_enabled": bool(api_key) and bind_host not in LOOPBACK,
                "native_dialog": bool(local and native_dialog and backend()), "max_upload_mb": max_upload_mb}

    @app.post("/api/login")
    async def login(request: Request):
        client = request.client.host if request.client else "?"
        if not api_key:
            return {"authenticated": True, "auth_required": False}
        if sessions.blocked(client):
            raise HTTPException(status_code=429, detail="too many failed logins; wait 5 minutes")
        body = await _body(request)
        key = body.get("key")
        if not isinstance(key, str) or not _key_ok(key):
            sessions.fail(client)
            raise HTTPException(status_code=401, detail="invalid key")
        response = JSONResponse({"authenticated": True, "auth_required": True})
        response.set_cookie(COOKIE, sessions.issue(), max_age=int(sessions.ttl_s), httponly=True,
                            samesite="strict", secure=request.url.scheme == "https", path="/")
        return response

    @app.post("/api/logout")
    def logout(request: Request):
        sessions.revoke(request.cookies.get(COOKIE))
        response = JSONResponse({"authenticated": False})
        response.delete_cookie(COOKIE, path="/")
        return response

    @app.get("/api/i18n/{lang}")
    def i18n(lang: str):
        try:
            return {"language": lang, "messages": load_catalog(lang)}
        except KeyError:
            raise HTTPException(status_code=404, detail="unsupported language") from None

    @app.get("/api/schema")
    def get_schema():
        return schema.schema_payload()

    @app.get("/api/system")
    def system():
        return {**system_info(), "workdir": ui.workdir, "runs_dir": ui.runs_dir,
                "auth_required": bool(api_key)}

    @app.get("/api/presets")
    def presets():
        return read_presets()

    # ── config, planner, data, tokenizer ───────────────────────────────
    @app.post("/api/config/estimate")
    async def estimate(request: Request):
        clean, bad = sanitize_config((await _body(request)).get("config") or {})
        if bad:
            _bad(bad)
        try:
            return _finite(config_estimate({**base_config(), **clean}))
        except (ValueError, ZeroDivisionError) as exc:
            raise HTTPException(status_code=422, detail={"error": "invalid", "message": str(exc)}) from exc

    @app.post("/api/config/save")
    async def config_save(request: Request):
        body = await _body(request)
        clean, bad = sanitize_config(body.get("config") or {})
        if bad:
            _bad(bad)
        path = ui.path(body.get("path") or "./out/chronos_config.json")
        if not path.endswith(".json"):
            _bad(["path"])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2, ensure_ascii=False)
        return {"path": path}

    @app.post("/api/config/load")
    async def config_load(request: Request):
        path = ui.path((await _body(request)).get("path") or "")
        if not path.endswith(".json") or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="not_found")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and isinstance(raw.get("config"), dict):       # a saved plan
            raw = {**raw["config"], "stage_overrides": raw.get("stage_overrides") or raw["config"].get(
                "stage_overrides")}
        clean, bad = sanitize_config(raw if isinstance(raw, dict) else {})
        return {"config": clean, "ignored": bad, "path": path}

    @app.post("/api/plan")
    async def plan(request: Request):
        from chronos.model.planner import plan_training
        values, bad = form_values(await _body(request), "plan")
        if bad:
            _bad(bad)
        try:
            req = plan_request(values)
        except ValueError as exc:
            _bad([str(exc)], code=str(exc))
        return _finite(_json_safe(plan_training(req)))

    # ── file picker: lists server folders (the UI is local; typed paths reach the same files) ──
    @app.get("/api/fs/list")
    def fs_list(path: str = "", kind: str = "any", hidden: bool = False):
        from chronos.workbench.files import list_dir
        if len(path) > schema.MAX_TEXT or "\x00" in path:
            _bad(["path"])
        try:
            return _json_safe(list_dir(ui.path(path) if path else ui.workdir, kind=kind, show_hidden=hidden))
        except ValueError:
            _bad(["kind"])
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail={"error": "not_found"}) from None

    # The browser hides real file paths. On this machine the server opens the system dialog and the
    # chosen paths are used in place; from another machine the files are uploaded instead.
    @app.post("/api/fs/native")
    async def fs_native(request: Request):
        from chronos.webui.native_dialog import DialogBusy, backend, choose
        if not _is_local(request):
            raise HTTPException(status_code=403, detail={"error": "native_local_only"})
        if not native_dialog or backend() is None:
            raise HTTPException(status_code=501, detail={"error": "native_unavailable"})
        body = await _body(request)
        kind = body.get("kind") or "any"
        if kind not in ("dir", "data", "checkpoint", "config", "any"):
            _bad(["kind"])
        folder = kind == "dir" or bool(body.get("folder"))
        start = ui.path(str(body.get("start") or "")) or ui.workdir
        start = start if os.path.isdir(start) else os.path.dirname(start)
        try:
            paths = await run_in_threadpool(choose, folder=folder, multiple=bool(body.get("multiple")) and not
                                            (folder and sys.platform.startswith("win")), start=start,
                                            title=str(body.get("title") or "")[:200])
        except DialogBusy:
            raise HTTPException(status_code=409, detail={"error": "dialog_open"}) from None
        except RuntimeError:
            raise HTTPException(status_code=501, detail={"error": "native_unavailable"}) from None
        return {"paths": paths or [], "cancelled": not paths}

    @app.put("/api/fs/upload")
    async def fs_upload(request: Request, kind: str, batch: str, name: str):
        from chronos.workbench.files import upload_path
        try:
            dst = upload_path(ui.workdir, kind, batch, name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"error": f"upload_{exc}", "fields": ["upload"]}) from None
        limit = max_upload_mb * 2**20
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise HTTPException(status_code=413, detail={"error": "upload_too_large"})
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        free = shutil.disk_usage(os.path.dirname(dst)).free
        reserve = max(UPLOAD_DISK_RESERVE, int(0.05 * shutil.disk_usage(os.path.dirname(dst)).total))
        expected = int(declared) if declared and declared.isdigit() else 0
        if free - expected < reserve:                 # keep room for checkpoints of running jobs
            raise HTTPException(status_code=507, detail={"error": "upload_disk_full"})
        tmp = f"{dst}.part-{os.getpid()}-{time.monotonic_ns()}"
        size, next_check = 0, 256 * 2**20
        try:
            with open(tmp, "wb") as fh:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(status_code=413, detail={"error": "upload_too_large"})
                    if size >= next_check:                # re-check free space every 256 MiB written
                        next_check += 256 * 2**20
                        if shutil.disk_usage(os.path.dirname(dst)).free < reserve:
                            raise HTTPException(status_code=507, detail={"error": "upload_disk_full"})
                    fh.write(chunk)
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        from chronos.workbench.files import upload_root
        return {"path": dst, "size": size, "root": upload_root(ui.workdir, kind, batch)}

    @app.post("/api/data/scan")
    async def data_scan(request: Request):
        from chronos.workbench.data import scan_dataset
        body = await _body(request)
        raw = body.get("path") or ""
        if not isinstance(raw, (str, list)) or len(str(raw)) > schema.MAX_MULTI_TEXT:
            _bad(["path"])
        paths = ui.paths(raw)
        if not paths:
            _bad(["path"], "no_paths")
        try:
            limit = max(100, min(int(body.get("max_records") or 50_000), 500_000))
        except (TypeError, ValueError):
            _bad(["max_records"])
        summary, lengths, samples = await run_in_threadpool(scan_dataset, paths, max_records=limit)
        if "error" in summary:
            raise HTTPException(status_code=422, detail={"error": "no_files", "message": summary["error"]})
        return _finite(_json_safe({"summary": summary, "histogram": _histogram(lengths),
                                   "samples": samples[:5]}))

    @app.post("/api/tokenizer/start")
    async def tokenizer_start(request: Request):
        from chronos.workbench.tokenizer import parse_paths
        values, bad = form_values(await _body(request), "tok")
        if bad:
            _bad(bad)
        fields = [f.strip() for f in (values["fields"] or "").split(",") if f.strip()]
        result = ui.tokenizer.start([ui.path(p) for p in parse_paths(values["paths"])], ui.path(values["out_dir"]),
                                    values["vocab_size"], values["per_source"], fields, values["temperature"])
        if result in ("no_paths", "no_out", "no_files"):
            _bad(["tok.paths" if result != "no_out" else "tok.out_dir"], result)
        if result == "busy":
            raise HTTPException(status_code=409, detail="busy")
        return tokenizer_state()

    @app.get("/api/tokenizer/state")
    def tokenizer_state():
        from chronos.workbench.tokenizer import read_report, report_rows, report_summary
        state, log = ui.tokenizer.snapshot()
        report = read_report(ui.tokenizer.out_dir) if ui.tokenizer.out_dir and state == "done" else {}
        return _finite(_json_safe({"state": state, "log": log, "error": ui.tokenizer.error or None,
                                   "out_dir": ui.tokenizer.out_dir, "summary": report_summary(report),
                                   "rows": report_rows((report or {}).get("heldout_eval") or {})}))

    # ── training jobs (subprocess worker, run dir read by the monitor) ──
    @app.post("/api/train/start")
    async def train_start(request: Request):
        from chronos.workbench.lazy_probe import ProbeSweep, expand_variants
        body = await _body(request)
        cfg, bad = sanitize_config(body.get("config") or {})
        train, bad_t = form_values(body.get("train") or {}, "train")
        probe, bad_p = form_values(body.get("probe") or {}, "probe")
        if bad or bad_t or bad_p:
            _bad(bad + bad_t + bad_p)
        sweep = probe_sweep(probe)
        try:
            ProbeSweep.from_dict(sweep)
        except ValueError as exc:
            _bad(["probe"], "probe_invalid", str(exc))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"webui_{stamp}"
        run_dir = os.path.join(ui.runs_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        cfg = {**base_config(), **cfg}
        cfg["save_dir"] = ui.path(cfg.get("save_dir") or "./out")
        stages = []
        modes = [m for m in STAGE_ORDER if m in train["stages"]]      # canonical order, not click order
        first_data = ui.paths(train["data_path"])
        _check_data(first_data, "train.data_path")
        for i, mode in enumerate(modes):
            own = ui.paths(train.get(f"data_{mode}") or "") if i else []
            _check_data(own, f"train.data_{mode}")
            data = own or first_data
            stages.append({"mode": mode, "data_paths": data, "data_path": data[0] if len(data) == 1 else "",
                           "init_weight": ui.path(train["init_weight"]) if i == 0 else "",
                           "teacher_path": ui.path(train["teacher_path"]) if mode == "distill" else "",
                           "max_steps": train["max_steps"], "val_ratio": train["val_ratio"],
                           "sample_prompt": train["sample_prompt"]})
        job_spec = {"run_dir": run_dir, "config": cfg, "stages": stages, "backend": train["backend"],
                    "lazy_probe": {"enabled": bool(probe["enabled"]), "sweep": sweep}}
        job_path = os.path.join(run_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(job_spec, fh, indent=2, ensure_ascii=False)
        variants, total = expand_variants(ProbeSweep.from_dict(sweep), lambda p: p)
        try:
            job = ui.jobs.spawn("train", [sys.executable, "-m", "chronos.workbench.train_worker", job_path],
                                cwd=REPO_ROOT, title=" → ".join(modes),
                                meta={"run": run_name, "run_dir": run_dir, "stages": modes})
        except JobBusy:
            raise HTTPException(status_code=409, detail="busy") from None
        return {"job": job.view(log_lines=0), "run": run_name,
                "probe": {"enabled": bool(probe["enabled"]), "variants": len(variants), "requested": total}}

    @app.post("/api/probe/preview")
    async def probe_preview(request: Request):
        """How many combinations a probe form expands to (before ``auto`` paging resolves)."""
        from chronos.workbench.lazy_probe import ProbeSweep, expand_variants, variant_label
        probe, bad = form_values(await _body(request), "probe")
        if bad:
            _bad(bad)
        try:
            variants, total = expand_variants(ProbeSweep.from_dict(probe_sweep(probe)), lambda p: p)
        except ValueError as exc:
            _bad(["probe"], "probe_invalid", str(exc))
        return {"variants": len(variants), "requested": total, "labels": [variant_label(v) for v in variants]}

    @app.get("/api/jobs")
    def jobs(kind: str | None = None):
        return {"jobs": ui.jobs.list(kind)}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, log: int = 200):
        try:
            return _finite(_json_safe(ui.jobs.get(job_id).view(log_lines=max(0, min(int(log), 2000)))))
        except KeyError:
            raise HTTPException(status_code=404, detail="not_found") from None

    @app.post("/api/jobs/{job_id}/stop")
    def job_stop(job_id: str):
        try:
            return ui.jobs.stop(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="not_found") from None

    # ── run monitor (any run under runs_dir, including the CLI long runs) ──
    def _mon(fn, *args, **kwargs):
        try:
            return _finite(_json_safe(fn(*args, **kwargs)))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/monitor/runs")
    def monitor_runs():
        return {"root": ui.monitor.root, "runs": _mon(ui.monitor.runs)}

    @app.get("/api/monitor/state")
    def monitor_state(run: str | None = None, history: int = 1):
        return _mon(ui.monitor.snapshot, run, history=bool(history))

    @app.get("/api/monitor/metrics")
    def monitor_metrics(run: str | None = None, phase: str | None = None, since: int = 0, limit: int = 5000):
        return {"rows": _mon(ui.monitor.metrics, run, phase, since, max(1, min(int(limit), 20000)))}

    @app.get("/api/monitor/log")
    def monitor_log(run: str | None = None, bytes: int = 16000):  # noqa: A002 - query name
        return {"text": _mon(ui.monitor.log_tail, run, max(1000, min(int(bytes), 400_000)))}

    @app.get("/api/monitor/parameters")
    def monitor_parameters(run: str | None = None):
        return _mon(ui.monitor.training_parameters, run)

    @app.get("/api/monitor/lazy_probes")
    def monitor_probes(run: str | None = None, limit: int = 200):
        return {"probes": _mon(ui.monitor.lazy_probes, run, max(1, min(int(limit), 1000)))}

    # ── inference (thread job; architecture comes from the checkpoint sidecar) ──
    @app.post("/api/infer")
    async def infer(request: Request):
        from chronos.workbench.inference import generate_api
        body = await _body(request)
        cfg, bad = sanitize_config(body.get("config") or {})
        values, bad_i = form_values(body.get("infer") or {}, "infer")
        prompt = body.get("prompt")
        if bad or bad_i:
            _bad(bad + bad_i)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32_000:
            _bad(["prompt"])
        model_path = ui.path(values["model_path"])
        if not model_path or not os.path.exists(model_path):
            _bad(["infer.model_path"], "not_found")

        def work(job):
            job.log(f"{values['mode']} · {values['backend']} · {os.path.basename(model_path)}")
            out = generate_api({**base_config(), **cfg}, values["backend"], values["mode"], model_path,
                               prompt, values["max_tokens"], values["temperature"], values["raw_prompt"],
                               values["ram_load_ratio"], values["sweep_ram_ratios"], values["miss_policy"],
                               values["speculative"], values["draft_depth"], values["gamma"],
                               values["residency_planner"], values["paging"], values["residency_decay"],
                               values["decode_wait_s"], values["verify_wait_s"])
            if out.get("traceback"):
                job.log(out.pop("traceback"))
            if out.get("error"):
                raise RuntimeError(out["error"])
            return _finite(_json_safe(out))

        try:
            job = ui.jobs.run("inference", work, title=os.path.basename(model_path))
        except JobBusy:
            raise HTTPException(status_code=409, detail="busy") from None
        return {"job": job.view(log_lines=0)}

    # ── export ──────────────────────────────────────────────────────────
    @app.post("/api/export")
    async def export(request: Request):
        values, bad = form_values(await _body(request), "export")
        if bad:
            _bad(bad)
        model_path = ui.path(values["model_path"])
        if not model_path or not os.path.isfile(model_path):
            _bad(["export.model_path"], "not_found")
        if not values["output_dir"]:
            _bad(["export.output_dir"], "no_out")

        calibration = ui.paths(values["calibration_data_path"])
        if calibration and not all(os.path.exists(p) for p in calibration):
            _bad(["export.calibration_data_path"], "not_found")

        def work(job):
            from chronos.export import export_checkpoint, format_export_report
            from chronos.workbench.files import merge_datasets
            calibration_path = (merge_datasets(calibration, os.path.join(ui.workdir, "out", "merged_data"),
                                               log=job.log) if calibration else None)
            results = export_checkpoint(
                model_path, ui.path(values["output_dir"]), formats=values["formats"],
                config_path=ui.path(values["config_path"]) or None,
                expert_cache_dir=ui.path(values["expert_cache_dir"]) or None,
                copy_expert_cache=values["copy_cache"], auto_cluster=values["auto_cluster"],
                calibration_data_path=calibration_path,
                cluster_max_batches=values["cluster_max_batches"], cluster_batch_size=values["cluster_batch_size"],
                cluster_max_seq_len=values["cluster_max_seq_len"], cluster_device=values["cluster_device"])
            job.log(format_export_report(results))
            return [{"format": r.format, "path": r.path, "size_mb": round(r.bytes / 2**20, 2),
                     "tensors": r.tensors} for r in results]

        try:
            job = ui.jobs.run("export", work, title=os.path.basename(model_path))
        except JobBusy:
            raise HTTPException(status_code=409, detail="busy") from None
        return {"job": job.view(log_lines=0)}

    # ── Ollama / OpenAI-compatible server child ─────────────────────────
    @app.post("/api/serve/start")
    async def serve_start(request: Request):
        from chronos.workbench.serve import build_serve_command
        values, bad = form_values(await _body(request), "serve")
        if bad:
            _bad(bad)
        serve_key = os.environ.get("CHRONOS_API_KEY", "") if values["allow_remote"] else ""
        try:
            argv, env = build_serve_command(
                models_dir=ui.path(values["models_dir"]), host=values["host"], port=values["port"],
                device=values["device"], dtype=values["dtype"], lazy_experts=values["lazy_experts"],
                ram_expert_fraction=values["ram_expert_fraction"], allow_remote=values["allow_remote"],
                api_key=serve_key, kv_cache_dtype=values["kv_cache_dtype"] or "",
                mla_decode=values["mla_decode"] or "", router_scope=values["router_scope"] or "",
                max_resident_experts=values["max_resident_experts"], ram_budget_mb=values["ram_budget_mb"],
                monitor_runs_dir=ui.runs_dir, paging=values["paging"], mla_cache_rope=values["mla_cache_rope"])
        except ValueError as exc:
            _bad([f"serve.{exc}" if str(exc) in ("models_dir", "port", "paging") else "serve.host"], str(exc))
        if ui.serve.start(argv, env, values["host"], values["port"]) == "busy":
            raise HTTPException(status_code=409, detail="busy")
        return serve_state()

    @app.post("/api/serve/stop")
    def serve_stop():
        ui.serve.stop()
        return serve_state()

    @app.get("/api/serve/state")
    def serve_state():
        running, log = ui.serve.snapshot()
        url = ui.serve.url if running else ""
        addr = url.split("://", 1)[-1] if url else ""
        return {"running": running, "url": url, "addr": addr, "auth": ui.serve.auth if running else False,
                "log": log}

    # ── benchmark (benchmark_compare.py; external baseline optional) ──
    bench_script = os.path.join(REPO_ROOT, "benchmark_compare.py")
    bench_results = os.path.join(REPO_ROOT, "benchmark_results.json")

    def _bench_read(_job=None):
        if not os.path.isfile(bench_results):
            return None
        with open(bench_results, encoding="utf-8") as fh:
            return _finite(json.load(fh))

    @app.post("/api/benchmark/start")
    def benchmark_start():
        if not os.path.isfile(bench_script):
            raise HTTPException(status_code=404, detail="not_found")
        try:
            job = ui.jobs.spawn("benchmark", [sys.executable, bench_script], cwd=REPO_ROOT,
                                title="benchmark_compare", on_exit=_bench_read)
        except JobBusy:
            raise HTTPException(status_code=409, detail="busy") from None
        return {"job": job.view(log_lines=0)}

    @app.get("/api/benchmark/results")
    def benchmark_results():
        return {"available": os.path.isfile(bench_script), "results": _bench_read(),
                "metrics": [{"key": k, "label": label, "lower_is_better": low} for k, label, low in BENCH_METRICS]}

    # ── auto-tune (Optuna TPE; one search per server) ───────────────────
    @app.post("/api/autotune/start")
    async def autotune_start(request: Request):
        from chronos.tuning.chronos_auto_tuner import ChronosAutoTuner, ChronosSearchSpaceConfig
        body = await _body(request)
        values, bad = form_values(body.get("tune") or {}, "tune")
        cfg, bad_c = sanitize_config(body.get("config") or {})
        if bad or bad_c:
            _bad(bad + bad_c)
        from chronos.workbench.files import data_files
        data = ui.paths(values["data_path"])
        if not data or not all(os.path.exists(p) for p in data):
            _bad(["tune.data_path"], "not_found")
        if not data_files(data):
            _bad(["tune.data_path"], "no_files")
        data_path = data[0]
        if ui.tuner is not None and ui.tuner.is_running():
            raise HTTPException(status_code=409, detail="busy")
        cfg = {**base_config(), **cfg}
        # The loop knobs are probed at the depth the user plans to train at: at least 2, at most 8 (the probe is short).
        probe_depth = max(2, min(8, int(cfg.get("recurrent_depth") or 1)))
        space = ChronosSearchSpaceConfig(
            probe_recurrent_depth=probe_depth,
            **{f"tune_{name}": name in values["search"] for name in schema.FIELDS["tune.search"].choices})
        out_dir = os.path.join(ui.workdir, "auto_tune_cache")
        os.makedirs(out_dir, exist_ok=True)
        ui.tuner, ui.tune_log = ChronosAutoTuner(), []
        ui.tuner.start(model_id=os.path.join(ui.path(cfg.get("save_dir") or "./out"),
                                             f"chronos_{cfg.get('hidden_size', 256)}_moe.pth"),
                       dataset_path=data_path, train_ratio=0.95, prompt_template="alpaca", think_mode="keep",
                       search_space=space, n_trials=values["n_trials"], probe_steps=values["probe_steps"],
                       output_dir=out_dir, seed=42,
                       tokenizer_dir=ui.path(cfg["tokenizer_path"]) if cfg.get("tokenizer_path") else None)
        return autotune_state()

    @app.post("/api/autotune/stop")
    def autotune_stop():
        if ui.tuner is not None and ui.tuner.is_running():
            ui.tuner.stop()
        return autotune_state()

    @app.get("/api/autotune/state")
    def autotune_state():
        if ui.tuner is None:
            return {"state": "idle", "log": [], "trials": [], "best": {}}
        for event in ui.tuner.poll():
            if event.get("type") == "log":
                ui.tune_log.append(str(event.get("line", "")))
        del ui.tune_log[:-500]
        trials = [{"trial": tr.trial_number + 1, "params": tr.params, "loss": tr.train_loss,
                   "duration_s": tr.duration_s, "status": tr.status} for tr in list(ui.tuner.trials)[-200:]]
        best = dict(ui.tuner.result.best_params) if ui.tuner.result and ui.tuner.result.best_params else {}
        return _finite(_json_safe({"state": ui.tuner.status, "log": ui.tune_log[-200:],
                                   "trials": trials, "best": best}))

    # ── live IO / cache metrics of this process ─────────────────────────
    @app.get("/api/io")
    def io_metrics(samples: int = 120):
        from chronos.runtime.metrics import bus
        keep = max(1, min(int(samples), 2000))
        return _finite({"series": {name: [[round(t, 3), v] for t, v in points[-keep:]]
                                   for name, points in sorted(bus.snapshot().items())}})

    return app


BENCH_METRICS = (("params_m", "bench.params", False), ("ram_model_gb", "bench.ram_model", True),
                 ("ram_train_gb", "bench.ram_train", True), ("ram_infer_gb", "bench.ram_infer", True),
                 ("tokens_per_sec", "bench.tps", False))


def _histogram(lengths, bins: int = 24) -> dict:
    """Record-length histogram (characters) for the Data page; log-spaced bins."""
    import math
    values = [int(v) for v in lengths or () if v is not None and v >= 0]
    if not values:
        return {"edges": [], "counts": []}
    lo, hi = max(1, min(values)), max(values)
    if hi <= lo:
        return {"edges": [lo, lo + 1], "counts": [len(values)]}
    edges = sorted({int(round(math.exp(math.log(lo) + (math.log(hi + 1) - math.log(lo)) * i / bins)))
                    for i in range(bins + 1)})
    counts = [0] * (len(edges) - 1)
    for v in values:
        idx = min(len(counts) - 1, max(0, next((i for i, e in enumerate(edges[1:]) if v < e), len(counts) - 1)))
        counts[idx] += 1
    return {"edges": edges, "counts": counts}


def main(argv=None) -> int:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--legacy" in argv:                        # the Gradio console, unchanged
        argv.remove("--legacy")
        from chronos.app import main as legacy_main
        sys.argv = ["chronos-ui"] + argv
        return legacy_main() or 0
    parser = argparse.ArgumentParser(prog="chronos webui", description="Chronos Web UI (FastAPI + single-page app)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--workdir", default=".", help="base for relative paths (data, out/, runs/)")
    parser.add_argument("--runs-dir", default="runs", help="run folders shown in the monitor (relative to --workdir)")
    parser.add_argument("--lan", action="store_true",
                        help="also open the UI to this machine's local network (off by default). Browsers on other "
                             "machines log in with CHRONOS_UI_KEY, or with a key printed here when none is set; "
                             "this machine needs no login. Not for use behind a reverse proxy")
    parser.add_argument("--allow-remote", action="store_true",
                        help="listen beyond loopback; needs a key in CHRONOS_UI_KEY or CHRONOS_API_KEY")
    parser.add_argument("--max-upload-mb", type=int, default=MAX_UPLOAD_MB,
                        help="largest single file a LAN browser may upload (MiB)")
    parser.add_argument("--no-native-dialog", action="store_true",
                        help="never open the system file dialog on this machine (e.g. when run over SSH)")
    parser.add_argument("--allowed-host", action="append", default=[],
                        help="extra Host header value to accept (e.g. a reverse-proxy name); repeatable")
    parser.add_argument("--session-hours", type=float, default=12.0)
    parser.add_argument("--ssl-certfile", default=None)
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--legacy", action="store_true", help="start the old Gradio console instead")
    parser.add_argument("--cpu-threads", default="auto", help="PyTorch/BLAS CPU threads for in-process work")
    parser.add_argument("--cpu-budget-percent", type=float, default=100,
                        help="CPU physical-core budget used when --cpu-threads=auto")
    parser.add_argument("--share", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.share:
        parser.error("--share (a public Gradio tunnel) is not offered by this UI; use --allow-remote with a key "
                     "and TLS, or --legacy --share for the old Gradio console")
    from chronos.trainer.device_utils import configure_cpu_threads
    configure_cpu_threads(args.cpu_threads, budget_percent=args.cpu_budget_percent)
    api_key = os.environ.get("CHRONOS_UI_KEY") or os.environ.get("CHRONOS_API_KEY") or None
    extra, generated, lan = list(args.allowed_host), False, []
    if args.lan:
        import secrets
        import socket
        if args.host in LOOPBACK:
            args.host = "0.0.0.0"
        args.allow_remote = True
        if not api_key:
            api_key, generated = secrets.token_urlsafe(18), True
        lan = lan_addresses()
        name = socket.gethostname().lower()
        extra += lan + [name, f"{name.split('.')[0]}.local"]
    if args.max_upload_mb < 1:
        parser.error("--max-upload-mb must be at least 1")
    problem = check_bind(args.host, args.allow_remote, api_key)
    if problem:
        parser.error("binding a non-loopback host requires --allow-remote and a key in CHRONOS_UI_KEY "
                     "or CHRONOS_API_KEY")
    import uvicorn
    app = create_app(args.workdir, runs_dir=args.runs_dir, api_key=api_key,
                     bind_host=args.host, extra_hosts=tuple(extra), session_hours=args.session_hours,
                     max_upload_mb=args.max_upload_mb, trust_local=args.lan, native_dialog=not args.no_native_dialog)
    scheme = "https" if args.ssl_certfile else "http"
    if args.lan:
        print(f"[chronos webui] this machine: {scheme}://127.0.0.1:{args.port}/  (no login)  "
              f"workdir={os.path.abspath(args.workdir)}", flush=True)
        for addr in lan or ["<this machine's LAN address>"]:
            print(f"[chronos webui] LAN:          {scheme}://{addr}:{args.port}/  (login required)", flush=True)
        if generated:
            print(f"[chronos webui] LAN login key (generated for this run; set CHRONOS_UI_KEY to choose one): "
                  f"{api_key}", flush=True)
    else:
        shown = f"[{args.host}]" if ":" in args.host else args.host
        auth = "login required" if api_key else "no auth (loopback only)"
        print(f"[chronos webui] {scheme}://{shown}:{args.port}/  workdir={os.path.abspath(args.workdir)}  {auth}",
              flush=True)
    if api_key and scheme == "http" and args.host not in LOOPBACK:
        print("[chronos webui] warning: the login key crosses the network in clear text; use --ssl-certfile/"
              "--ssl-keyfile or a TLS reverse proxy", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
