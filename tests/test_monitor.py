"""Read-only training monitor: JSONL reader, run snapshots, API auth, Gradio tab, telemetry."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---- fixtures ----------------------------------------------------------------
def _write_jsonl(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _synthetic_run(root, name="demo", *, final=False, steps=10):
    """A run directory as an old (pre-telemetry) longrun leaves it: spec + metrics only."""
    path = os.path.join(root, name)
    os.makedirs(os.path.join(path, "checkpoints"))
    spec = {"run_dir": path, "budget_hours": 10.0, "eval_reserve_fraction": 0.1,
            "phases": [{"name": "pretrain", "budget_fraction": 0.7}, {"name": "sft", "budget_fraction": 0.3}]}
    with open(os.path.join(path, "run_spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    t0 = datetime(2026, 11, 6, 0, 0, 0)
    rows = [{"phase": "pretrain", "event": "phase_start", "step": 0, "time": t0.isoformat()}]
    for i in range(1, steps + 1):
        rows.append({"phase": "pretrain", "step": i, "total": 100, "loss": 5.0 - 0.1 * i, "primary_loss": 4.5 - 0.1 * i,
                     "lr": 1e-3, "grad_norm": 1.0, "tokens_per_s": 1000.0, "step_s": 2.0,
                     "time": (t0 + timedelta(seconds=2 * i)).isoformat()})
    rows.append({"phase": "pretrain", "event": "validation", "step": 4, "valid_ce": 3.2})
    rows.append({"phase": "pretrain", "event": "validation", "step": 8, "valid_ce": 2.9})
    rows.append({"phase": "pretrain", "event": "validation", "step": 9, "valid_ce": 3.0})
    if final:
        rows.append({"phase": "pretrain", "event": "final", "step": steps})
    _write_jsonl(os.path.join(path, "metrics.jsonl"), rows)
    return path


# ---- JsonLines -----------------------------------------------------------------
def test_jsonlines_incremental_partial_and_truncate(tmp_path):
    from chronos.monitor import JsonLines
    path = str(tmp_path / "m.jsonl")
    reader = JsonLines(path)
    assert reader.read() == []                              # missing file
    _write_jsonl(path, [{"a": 1}, {"a": 2}])
    assert [r["a"] for r in reader.read()] == [1, 2]
    with open(path, "a", encoding="utf-8") as fh:            # in-flight write: no newline yet
        fh.write('{"a": 3}\n{"a": 4')
    assert [r["a"] for r in reader.read()] == [1, 2, 3]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('}\nnot json\n')
    assert [r["a"] for r in reader.read()] == [1, 2, 3, 4]
    assert reader.bad_lines == 1
    _write_jsonl(path, [{"a": 9}])                          # truncated / rewritten
    assert [r["a"] for r in reader.read()] == [9]
    capped = JsonLines(path, max_rows=2)
    _write_jsonl(path, [{"a": i} for i in range(5)])
    assert [r["a"] for r in capped.read()] == [3, 4]


# ---- snapshot ------------------------------------------------------------------
def test_snapshot_derives_status_progress_and_best_valid(tmp_path):
    from chronos.monitor import RunMonitor
    _synthetic_run(str(tmp_path), "demo", steps=10)
    os.makedirs(tmp_path / "not_a_run")
    mon = RunMonitor(str(tmp_path))
    assert [r["name"] for r in mon.runs()] == ["demo"]
    snap = mon.snapshot("demo")
    status = snap["status"]
    assert status["source"] == "derived" and not status["alive"] and status["state"] == "stopped"
    assert status["log_interval_s"] == pytest.approx(2.0)
    assert snap["latest"]["step"] == 10
    assert snap["best_valid"]["pretrain"] == {"step": 8, "valid_ce": 2.9, "time": None}
    progress = snap["progress"]
    assert [p["state"] for p in progress["phases"]] == ["active", "pending"]
    assert progress["phase_fraction"] == pytest.approx(0.1)
    assert progress["phase_remaining_s"] == pytest.approx(90 * 2.0)
    # Later phase budget (0.3 · 9 h training) and the eval reserve (1 h) are added.
    assert progress["run_remaining_s"] == pytest.approx(180 + 0.3 * 9 * 3600 + 3600)
    assert {e["event"] for e in snap["events"]} == {"phase_start", "validation"}
    assert len(snap["history"]["pretrain"]) == 10
    assert [r["step"] for r in mon.metrics("demo", "pretrain", since_step=8)] == [9, 10]


def test_snapshot_final_event_marks_finished(tmp_path):
    from chronos.monitor import RunMonitor
    _synthetic_run(str(tmp_path), "done", final=True)
    snap = RunMonitor(str(tmp_path)).snapshot("done")
    assert snap["status"]["state"] == "finished"


def test_status_json_heartbeat_wins_over_derivation(tmp_path):
    from chronos.monitor import RunMonitor
    path = _synthetic_run(str(tmp_path), "hb")
    with open(os.path.join(path, "status.json"), "w", encoding="utf-8") as fh:
        json.dump({"state": "failed", "pid": 2 ** 22 + 12345, "phase": "pretrain", "error": "boom",
                   "phase_wall_s": 3600.0, "phase_budget_s": 4 * 3600.0}, fh)
    snap = RunMonitor(str(tmp_path)).snapshot("hb")
    assert snap["status"]["source"] == "status.json"
    assert snap["status"]["state"] == "failed" and snap["status"]["error"] == "boom"
    assert snap["progress"]["phase_fraction"] == pytest.approx(0.25)


@pytest.mark.parametrize("name", ["../etc", ".hidden", "a/b", "..", "missing"])
def test_unknown_or_unsafe_run_names_are_rejected(tmp_path, name):
    from chronos.monitor import RunMonitor
    _synthetic_run(str(tmp_path), "demo")
    with pytest.raises(KeyError):
        RunMonitor(str(tmp_path)).snapshot(name)


def test_mach_tick_scale_is_positive():
    from chronos.monitor.process import _mach_tick_ns
    assert _mach_tick_ns() > 0


# ---- standalone API ------------------------------------------------------------
def _client(root, **kwargs):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from chronos.monitor.server import create_monitor_app
    return TestClient(create_monitor_app(str(root), **kwargs))


def test_api_open_on_loopback_without_key(tmp_path):
    _synthetic_run(str(tmp_path), "demo")
    client = _client(tmp_path)
    assert client.get("/api/session").json() == {"auth_required": False, "authenticated": True, "local_only": True}
    state = client.get("/api/state", params={"run": "demo"})
    assert state.status_code == 200 and state.json()["run"] == "demo"
    assert client.get("/api/state", params={"run": "../demo"}).status_code == 404
    assert [r["step"] for r in client.get("/api/metrics", params={"run": "demo", "since": 8}).json()["rows"]] == [9, 10]
    assert client.get("/api/parameters", params={"run": "demo"}).json()["spec"]["budget_hours"] == 10.0
    assert client.get("/api/tensors", params={"run": "demo"}).json() == {"tensors": []}
    page = client.get("/")
    assert page.status_code == 200 and "static/app.js" in page.text
    for header in ("Content-Security-Policy", "X-Content-Type-Options", "X-Frame-Options", "Cache-Control"):
        assert header in page.headers
    assert "script-src 'self'" in page.headers["Content-Security-Policy"]
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/app.css").status_code == 200
    for name in ("index.html", "..%2Fserver.py", "server.py"):
        assert client.get(f"/static/{name}").status_code == 404


def test_api_key_login_cookie_bearer_and_rate_limit(tmp_path):
    from chronos.monitor.server import COOKIE
    _synthetic_run(str(tmp_path), "demo")
    key = "correct horse battery staple"
    client = _client(tmp_path, api_key=key)
    assert client.get("/").status_code == 200                       # login page shell is public
    assert client.get("/api/session").json()["auth_required"] is True
    for path in ("/api/runs", "/api/state", "/api/metrics", "/api/parameters", "/api/tensors", "/api/log"):
        assert client.get(path).status_code == 401, path
    assert client.get("/api/runs", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get("/api/runs", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = client.post("/api/login", json={"key": key})
    assert ok.status_code == 200
    cookie = ok.headers["set-cookie"].lower()
    assert COOKIE in cookie and "httponly" in cookie and "samesite=strict" in cookie and key not in cookie
    assert client.get("/api/runs").status_code == 200                # session cookie
    client.post("/api/logout")
    client.cookies.clear()
    assert client.get("/api/runs").status_code == 401
    other = _client(tmp_path, api_key=key)
    codes = [other.post("/api/login", json={"key": f"wrong{i}"}).status_code for i in range(8)]
    assert codes == [401] * 8
    assert other.post("/api/login", json={"key": key}).status_code == 429   # blocked even with the right key
    assert other.post("/api/login", content=b"not json").status_code == 429


def test_check_bind_requires_remote_flag_and_key():
    from chronos.monitor.server import check_bind
    assert check_bind("127.0.0.1", False, None) is None
    assert check_bind("::1", False, None) is None
    assert check_bind("0.0.0.0", False, "k")
    assert check_bind("0.0.0.0", True, None)
    assert check_bind("0.0.0.0", True, "k") is None


def test_serve_mounts_monitor_with_same_key(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from chronos.serving.ollama_server import create_app

    class _Runtime:
        loaded, last_used, keep_alive_s = {}, {}, 300

        def available(self):
            return {}

    _synthetic_run(str(tmp_path), "demo")
    key = "k-123"
    client = TestClient(create_app(_Runtime(), api_key=key, monitor_runs_dir=str(tmp_path)))
    assert client.get("/api/tags").status_code == 401
    assert client.get("/api/tags", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get("/monitor", follow_redirects=False).status_code in (301, 302, 307, 308)
    page = client.get("/monitor/")
    assert page.status_code == 200 and "static/app.css" in page.text
    assert client.get("/monitor/api/state").status_code == 401
    login = client.post("/monitor/api/login", json={"key": key})
    assert login.status_code == 200 and "path=/monitor/" in login.headers["set-cookie"].lower()
    assert client.get("/monitor/api/state", params={"run": "demo"}).json()["run"] == "demo"
    no_monitor = TestClient(create_app(_Runtime(), api_key=key))
    assert no_monitor.get("/monitor/").status_code == 401


# ---- longrun telemetry -> monitor -> Gradio tab ---------------------------------
def _telemetry_run(tmp_path):
    """Three CPU steps of a tiny MLX run with telemetry on (writes status/runtime/parameters)."""
    mx = pytest.importorskip("mlx.core")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_longrun import _spec, _tiny_shards, _train
    from chronos.mlx.training.longrun import LongRun
    shards = str(tmp_path / "shards"); _tiny_shards(shards)
    run_dir = str(tmp_path / "runs" / "tiny")
    with mx.stream(mx.cpu):
        run = LongRun(_spec(run_dir, shards), log=lambda *_: None)
        _train(run, 3)
    return run_dir


def test_longrun_telemetry_feeds_monitor_and_gradio_tab(tmp_path):
    run_dir = _telemetry_run(tmp_path)
    for name in ("status.json", "runtime.json", "parameters.jsonl", "metrics.jsonl", "run_spec.json"):
        assert os.path.isfile(os.path.join(run_dir, name)), name
    with open(os.path.join(run_dir, "runtime.json"), encoding="utf-8") as fh:
        runtime = json.load(fh)
    assert runtime["precision"]["compute"] and runtime["code"]["sha256"]
    from chronos.monitor import RunMonitor
    mon = RunMonitor(os.path.dirname(run_dir))
    snap = mon.snapshot("tiny")
    assert snap["status"]["source"] == "status.json" and snap["status"]["pid"] == os.getpid()
    assert snap["latest"]["step"] == 3
    summary = snap["parameters"]
    assert summary["step"] == 1 and summary["tensor_count"] > 0 and "tensors" not in summary
    assert {"experts", "attention", "embedding"} <= set(summary["groups"])
    full = mon.parameters("tiny")
    first = full["tensors"][0]
    assert {"name", "group", "shape", "rms", "grad_rms", "update_ratio"} <= set(first)
    assert mon.tensor_history("tiny", first["name"])[0]["step"] == 1
    assert summary["routing"] and summary["routing"]["load"]

    pytest.importorskip("gradio")
    from ui.tabs.monitor_tab import refresh
    out = refresh(os.path.dirname(run_dir), "tiny", "pretrain", "primary_loss")
    assert len(out) == 31
    status_html = out[2]
    assert "tiny" in status_html and "<script" not in status_html


def test_gradio_monitor_tab_builds_and_handles_empty_dir(tmp_path):
    gr = pytest.importorskip("gradio")
    from ui.tabs.monitor_tab import build_monitor_tab, refresh
    with gr.Blocks():
        with gr.Tabs():
            tab = build_monitor_tab()
    fn, inputs, outputs = tab.monitor_wiring
    assert fn is refresh and len(outputs) == 31 and len(inputs) == 5
    empty = refresh(str(tmp_path / "nothing"), None, None, "primary_loss")
    assert len(empty) == 31


def test_heatmap_html_escapes_labels():
    pytest.importorskip("gradio")
    from ui.tabs.monitor_tab import _heat_html
    html = _heat_html("<b>t</b>", ["<img src=x>"], [[0.1, 0.9]])
    assert "<img" not in html and "&lt;img" in html and "&lt;b&gt;" in html


def test_monitor_i18n_keys_present_for_all_languages():
    from ui.i18n import LANGUAGE_CHOICES, TRANSLATIONS
    keys = {k for k in TRANSLATIONS["en"] if k == "tab.monitor" or k.startswith("monitor.")}
    assert len(keys) > 90
    for _, code in LANGUAGE_CHOICES:
        missing = keys - set(TRANSLATIONS[code])
        assert not missing, (code, sorted(missing)[:5])


def test_monitor_static_assets_parse():
    static = os.path.join(_repo_root, "chronos", "monitor", "static")
    for name in ("index.html", "app.js", "app.css"):
        assert os.path.isfile(os.path.join(static, name))
    with open(os.path.join(static, "index.html"), encoding="utf-8") as fh:
        page = fh.read()
    assert "<script>" not in page and " style=" not in page and "onclick" not in page   # CSP: no inline code
    with open(os.path.join(static, "app.js"), encoding="utf-8") as fh:
        app = fh.read()
    assert '"lazy"' in app and "function panelLazy" in app and "api/lazy_probes" in app   # lazy-expert tab
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    result = subprocess.run([node, "--check", os.path.join(static, "app.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
