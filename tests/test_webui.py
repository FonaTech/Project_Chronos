"""Chronos Web UI (FastAPI + SPA): catalogs, language detection, security, forms, jobs."""
import json
import os
import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from chronos.webui import schema  # noqa: E402
from chronos.webui import server  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "chronos", "webui", "static")
PH = re.compile(r"\{(\w+)\}")
MUTATE = {"X-Chronos-Request": "1"}


def _catalog(lang):
    with open(os.path.join(ROOT, "chronos", "webui", "i18n", f"{lang}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _client(tmp_path, **kw):
    from fastapi.testclient import TestClient
    app = server.create_app(str(tmp_path), runs_dir="runs", **kw)
    return TestClient(app, base_url="http://127.0.0.1"), app


# ── catalogs ────────────────────────────────────────────────────────────
def test_catalogs_have_identical_keys_placeholders_and_no_empty_values():
    en = _catalog("en")
    for lang in schema.LANGUAGES:
        cat = _catalog(lang)
        assert set(cat) == set(en), (lang, sorted(set(en) ^ set(cat))[:10])
        assert not [k for k, v in cat.items() if not str(v).strip()], lang
        bad = [k for k in en if set(PH.findall(en[k])) != set(PH.findall(cat[k]))]
        assert not bad, (lang, bad[:10])


def test_every_schema_field_group_choice_and_rule_is_translated():
    need = {f"group.{g}" for g in schema.GROUPS} | {f"source.{s}" for s in schema.SOURCES}
    need |= {f"rule.{r}" for r in schema.RULE_REFS} | set(schema.choice_keys())
    for f in schema.ALL_FIELDS:
        need |= {f"field.{f.key}.label", f"field.{f.key}.help"}
    for lang in schema.LANGUAGES:
        missing = sorted(need - set(_catalog(lang)))
        assert not missing, (lang, missing[:10])


def test_static_files_use_only_catalog_keys():
    en = _catalog("en")
    call = re.compile(r"""\b(?:t|button|pageHead|has)\(\s*["']([a-z0-9_.]+)["'](?:\s*,\s*["']([a-z0-9_.]+)["'])?""")
    used = set()
    for name in os.listdir(STATIC):
        text = open(os.path.join(STATIC, name), encoding="utf-8").read()
        if name.endswith(".html"):
            used |= set(re.findall(r'data-i18n="([^"]+)"', text))
        elif name.endswith(".js"):
            for a, b in call.findall(text):
                used |= {k for k in (a, b) if k and "." in k}
            used |= set(re.findall(r'"(probe\.skip\.[a-z_]+)"', text))
    assert used, "no keys found"
    assert not sorted(used - set(en))
    nav = re.findall(r'\["([a-z]+)",\s*render', open(os.path.join(STATIC, "app.js"), encoding="utf-8").read())
    assert len(nav) == 11 and all(f"nav.{n}" in en for n in nav)


def test_catalog_endpoint_and_unknown_language(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/i18n/ja").json()
    assert body["language"] == "ja" and body["messages"] == _catalog("ja")
    assert client.get("/api/i18n/fr").status_code == 404


# ── language detection ──────────────────────────────────────────────────
@pytest.mark.parametrize("tag,expected", [
    ("zh-CN,zh;q=0.9", "zh-Hans"), ("zh", "zh-Hans"), ("zh-SG", "zh-Hans"), ("zh-Hans-CN", "zh-Hans"),
    ("zh-TW", "zh-Hant"), ("zh-HK", "zh-Hant"), ("zh-MO", "zh-Hant"), ("zh-Hant", "zh-Hant"),
    ("ja-JP", "ja"), ("en-GB", "en"), ("fr-FR,de;q=0.8", "en"), ("fr-FR,ja;q=0.5", "ja"), ("", "en"), (None, "en"),
])
def test_resolve_language(tag, expected):
    assert server.resolve_language(tag) == expected


def test_session_reports_browser_language(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/api/session", headers={"Accept-Language": "zh-TW,zh;q=0.8"}).json()
    # TestClient's peer is "testclient", not loopback: the server treats it as another machine
    assert r == {"auth_required": False, "authenticated": True, "local_only": True, "language": "zh-Hant",
                 "access": "lan", "lan_enabled": False, "native_dialog": False, "max_upload_mb": server.MAX_UPLOAD_MB}


# ── security ────────────────────────────────────────────────────────────
def test_host_header_allowlist(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/health").status_code == 200
    assert client.get("/health", headers={"Host": "evil.example"}).status_code == 421


def test_mutations_need_marker_header_and_same_origin(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/api/logout").status_code == 403
    assert client.post("/api/logout", headers={**MUTATE, "Origin": "http://evil.example"}).status_code == 403
    assert client.post("/api/logout", headers={**MUTATE, "Origin": "http://127.0.0.1"}).status_code == 200


def test_security_headers_and_csp(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    csp = r.headers.get("content-security-policy", "")
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("style-src")[0]
    html = r.text
    assert "<script>" not in html and 'style="' not in html


def test_login_flow_and_rate_limit(tmp_path):
    client, _ = _client(tmp_path, api_key="s3cret-key")
    assert client.get("/api/schema").status_code == 401
    assert client.get("/api/session").json()["authenticated"] is False
    assert client.post("/api/login", json={"key": "wrong"}, headers=MUTATE).status_code == 401
    ok = client.post("/api/login", json={"key": "s3cret-key"}, headers=MUTATE)
    assert ok.status_code == 200 and server.COOKIE in ok.cookies
    assert client.get("/api/schema").status_code == 200
    client.post("/api/logout", headers=MUTATE)
    client.cookies.clear()
    assert client.get("/api/schema").status_code == 401
    codes = [client.post("/api/login", json={"key": "nope"}, headers=MUTATE).status_code for _ in range(10)]
    assert codes[-1] == 429
    assert client.post("/api/login", json={"key": "s3cret-key"}, headers=MUTATE).status_code == 429


def test_remote_bind_needs_flag_and_key():
    assert server.check_bind("127.0.0.1", False, None) is None
    assert server.check_bind("0.0.0.0", False, "k")
    assert server.check_bind("0.0.0.0", True, None)
    assert server.check_bind("0.0.0.0", True, "k") is None


@pytest.mark.parametrize("name,status", [
    ("app.js", 200), ("app.css", 200), ("icon.svg", 200), ("index.html", 404), ("..%2Fserver.py", 404),
    ("../server.py", 404), ("App.js", 404), ("missing.js", 404),
])
def test_static_name_filter(tmp_path, name, status):
    client, _ = _client(tmp_path)
    assert client.get(f"/static/{name}").status_code == status


# ── config and forms ────────────────────────────────────────────────────
def test_sanitize_config_validates_and_drops_unknown_keys():
    clean, bad = server.sanitize_config({"hidden_size": 384, "num_experts": "many", "__proto__": 1,
                                         "modalities": "image, text", "stage_overrides": {"sft": {"learning_rate": 1e-4,
                                         "evil": "x"}, "nope": {}}})
    assert clean["hidden_size"] == 384 and "num_experts" in bad
    assert "__proto__" not in clean and clean["modalities"] == ["image"]
    assert clean["stage_overrides"] == {"sft": {"learning_rate": 1e-4}}


def test_form_values_fill_defaults_and_flag_bad_values():
    values, bad = server.form_values({"train.max_steps": 5, "train.stages": ["sft", "pretrain"]}, "train")
    assert values["max_steps"] == 5 and values["stages"] == ["sft", "pretrain"] and not bad
    assert values["data_sft"] == "" and values["backend"] == "auto"
    _, bad = server.form_values({"train.stages": [], "train.max_steps": -1}, "train")
    assert set(bad) == {"train.stages", "train.max_steps"}


def test_presets_are_complete_and_estimate_matches_meta(tmp_path):
    client, _ = _client(tmp_path)
    data = client.get("/api/presets").json()
    names = [p["name"] for p in data["presets"]]
    assert data["default"] in names and len(names) == 5
    base = server.base_config()
    for preset in data["presets"]:
        est = client.post("/api/config/estimate", json={"config": {**base, **preset["config"]}}, headers=MUTATE).json()
        meta = preset["meta"]
        assert est["label"] == "estimated"
        assert est["total_params"] == meta["total_params"] and est["active_params"] == meta["active_params"]
        assert est["active_params"] <= est["active_union_upper"] <= est["total_params"]


def test_estimate_rejects_bad_field(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/api/config/estimate", json={"config": {"hidden_size": "big"}}, headers=MUTATE)
    assert r.status_code == 422 and r.json()["detail"]["fields"] == ["hidden_size"]


# ── lazy-loading probe ──────────────────────────────────────────────────
def test_probe_preview_counts_variants(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/api/probe/preview", json={"probe.paging": ["auto", "expert"], "probe.ram_ratios": [0.5, 1.0],
                                                "probe.miss_policies": ["sync_on_demand", "on_demand"]}, headers=MUTATE)
    assert r.status_code == 200
    body = r.json()
    assert body["variants"] == len(body["labels"]) >= 2 and body["requested"] >= body["variants"]
    bad = client.post("/api/probe/preview", json={"probe.ram_ratios": [7]}, headers=MUTATE)
    assert bad.status_code == 422 and bad.json()["detail"]["fields"] == ["probe.ram_ratios"]


# ── training job ────────────────────────────────────────────────────────
def test_train_start_orders_stages_and_routes_per_stage_data(tmp_path, monkeypatch):
    client, app = _client(tmp_path)
    seen = {}

    class _Job:
        id = "train-1"

        def view(self, **_kw):
            return {"id": self.id, "state": "starting"}

    def fake_spawn(kind, argv, **kw):
        seen.update(kind=kind, argv=argv, **kw)
        return _Job()

    monkeypatch.setattr(app.state.ui.jobs, "spawn", fake_spawn)
    (tmp_path / "data").mkdir()
    for name in ("pre.jsonl", "sft.jsonl"):
        (tmp_path / "data" / name).write_text('{"text": "x"}\n', encoding="utf-8")
    train = {"train.stages": ["dpo", "pretrain", "sft"], "train.data_path": "data/pre.jsonl",
             "train.data_sft": "data/sft.jsonl", "train.init_weight": "", "train.max_steps": 3}
    r = client.post("/api/train/start", json={"config": {}, "train": train, "probe": {"probe.enabled": False}},
                    headers=MUTATE)
    assert r.status_code == 200, r.text
    assert seen["kind"] == "train" and seen["meta"]["stages"] == ["pretrain", "sft", "dpo"]
    spec = json.load(open(seen["argv"][-1], encoding="utf-8"))
    paths = {s["mode"]: s["data_path"] for s in spec["stages"]}
    assert [s["mode"] for s in spec["stages"]] == ["pretrain", "sft", "dpo"]
    assert paths["pretrain"] == str(tmp_path / "data/pre.jsonl")
    assert paths["sft"] == str(tmp_path / "data/sft.jsonl")
    assert paths["dpo"] == paths["pretrain"]                        # empty row reuses the first dataset
    assert spec["lazy_probe"]["enabled"] is False and r.json()["probe"]["enabled"] is False
    assert {s["mode"]: s["data_paths"] for s in spec["stages"]}["sft"] == [str(tmp_path / "data/sft.jsonl")]


def test_autotune_state_lists_trials_from_the_tuner(tmp_path):
    from types import SimpleNamespace
    client, app = _client(tmp_path)
    assert client.get("/api/autotune/state").json()["state"] == "idle"
    trial = SimpleNamespace(trial_number=0, params={"lr": 1e-3}, train_loss=2.5, duration_s=1.2, status="complete")
    app.state.ui.tuner = SimpleNamespace(poll=lambda: [{"type": "log", "line": "trial 1"}], trials=[trial],
                                         result=SimpleNamespace(best_params={"lr": 1e-3}), status="finished",
                                         is_running=lambda: False)
    s = client.get("/api/autotune/state").json()
    assert s["state"] == "finished" and s["best"] == {"lr": 1e-3} and s["log"] == ["trial 1"]
    assert s["trials"] == [{"trial": 1, "params": {"lr": 1e-3}, "loss": 2.5, "duration_s": 1.2, "status": "complete"}]


def test_entry_points_start_the_new_ui():
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    assert 'chronos-ui    = "chronos.webui.server:main"' in text
    assert '"chronos.webui" = [' in text and "i18n/*.json" in text
    assert "from chronos.webui.server import main" in open(os.path.join(ROOT, "chronos_app.py"), encoding="utf-8").read()


# ── file picker and dataset merging ─────────────────────────────────────
def _tree(root):
    (root / "sub").mkdir()
    (root / ".hidden").mkdir()
    (root / "a.jsonl").write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")
    (root / "b.csv").write_text("text\nthree\n", encoding="utf-8")
    (root / "notes.md").write_text("# not data\n", encoding="utf-8")
    (root / "model.pth").write_bytes(b"\0")
    (root / "sub" / "c.txt").write_text("four\nfive\n", encoding="utf-8")


def test_list_dir_filters_by_kind(tmp_path):
    from chronos.workbench.files import list_dir
    _tree(tmp_path)
    names = lambda kind, **kw: [e["name"] for e in list_dir(str(tmp_path), kind, **kw)["entries"]]  # noqa: E731
    assert names("data") == ["sub", "a.jsonl", "b.csv"]                 # folders first, then matching files
    assert names("dir") == ["sub"]
    assert names("checkpoint") == ["sub", "model.pth"]
    assert ".hidden" in names("dir", show_hidden=True)
    view = list_dir(str(tmp_path / "missing" / "deeper"), "any")       # walks up to the nearest folder
    assert view["path"] == str(tmp_path) and view["parent"] == str(tmp_path.parent)
    with pytest.raises(ValueError):
        list_dir(str(tmp_path), "exe")


def test_merge_datasets_streams_every_source_and_reuses_the_merge(tmp_path):
    from chronos.workbench.files import merge_datasets
    _tree(tmp_path)
    single = merge_datasets([str(tmp_path / "a.jsonl")], str(tmp_path / "out"))
    assert single == str(tmp_path / "a.jsonl")                          # one JSONL is used in place
    lines = []
    dst = merge_datasets([str(tmp_path / "a.jsonl"), str(tmp_path / "b.csv"), str(tmp_path / "sub")],
                         str(tmp_path / "out"), log=lines.append)
    rows = [json.loads(x) for x in open(dst, encoding="utf-8")]
    texts = [r.get("text") for r in rows]
    assert texts[:3] == ["one", "two", "three"] and set(texts[3:]) == {"four", "five"}
    manifest = json.load(open(dst + ".sources.json", encoding="utf-8"))
    assert manifest["records"] == 5 and len(manifest["sources"]) == 3
    again = []
    assert merge_datasets([str(tmp_path / "a.jsonl"), str(tmp_path / "b.csv"), str(tmp_path / "sub")],
                          str(tmp_path / "out"), log=again.append) == dst
    assert any("reusing" in x for x in again)
    with pytest.raises(FileNotFoundError):
        merge_datasets([str(tmp_path / "notes.md")], str(tmp_path / "out"))


def test_fs_list_endpoint(tmp_path):
    client, _ = _client(tmp_path)
    _tree(tmp_path)
    body = client.get("/api/fs/list", params={"path": str(tmp_path), "kind": "data"}).json()
    names = [e["name"] for e in body["entries"] if e["name"] != "runs"]     # create_app makes runs/
    assert names == ["sub", "a.jsonl", "b.csv"] and body["roots"]
    assert client.get("/api/fs/list").json()["path"] == str(tmp_path)  # default: the workbench folder
    assert client.get("/api/fs/list", params={"kind": "exe"}).status_code == 422
    assert client.get("/api/fs/list", params={"path": "a\x00b"}).status_code == 422
    locked, _ = _client(tmp_path, api_key="k3y-value")
    assert locked.get("/api/fs/list").status_code == 401


def test_schema_declares_picker_for_every_path_field():
    fields = {f["key"]: f for f in schema.schema_payload()["fields"]}
    for key, f in fields.items():
        if f["type"] == "path":
            assert "pick" in f, key
    assert set(schema.PICK) <= set(fields)
    assert fields["train.data_path"]["pick"] == {"kind": "data", "multiple": True, "merge": True}
    assert fields["tok.paths"]["pick"]["merge"] is False and fields["save_dir"]["pick"]["kind"] == "dir"
    assert schema.coerce("train.data_path", "x\n" * 5000)                # multi-path fields take long lists
    with pytest.raises(ValueError):
        schema.coerce("save_dir", "x" * 5000)


def test_train_start_merges_several_datasets_and_rejects_missing(tmp_path, monkeypatch):
    client, app = _client(tmp_path)
    _tree(tmp_path)
    seen = {}

    class _Job:
        id = "train-2"

        def view(self, **_kw):
            return {"id": self.id}

    monkeypatch.setattr(app.state.ui.jobs, "spawn", lambda kind, argv, **kw: seen.update(argv=argv) or _Job())
    train = {"train.stages": ["pretrain"], "train.data_path": "a.jsonl\nb.csv\n\nsub\n", "train.max_steps": 2}
    r = client.post("/api/train/start", json={"config": {}, "train": train, "probe": {"probe.enabled": False}},
                    headers=MUTATE)
    assert r.status_code == 200, r.text
    stage = json.load(open(seen["argv"][-1], encoding="utf-8"))["stages"][0]
    assert stage["data_paths"] == [str(tmp_path / n) for n in ("a.jsonl", "b.csv", "sub")] and stage["data_path"] == ""
    from chronos.workbench.train_worker import RunWriter, stage_data_path
    merged = stage_data_path(stage, RunWriter(str(tmp_path / "run")), {})
    assert merged.startswith(str(tmp_path / "run" / "data")) and sum(1 for _ in open(merged)) == 5
    bad = client.post("/api/train/start", json={"config": {}, "train": {**train, "train.data_path": "nope.jsonl"},
                                                "probe": {"probe.enabled": False}}, headers=MUTATE)
    assert bad.status_code == 422 and bad.json()["detail"]["fields"] == ["train.data_path"]
    md = client.post("/api/train/start", json={"config": {}, "train": {**train, "train.data_path": "notes.md"},
                                               "probe": {"probe.enabled": False}}, headers=MUTATE)
    assert md.status_code == 422 and md.json()["detail"]["error"] == "no_files"


def test_data_scan_accepts_several_paths(tmp_path):
    client, _ = _client(tmp_path)
    _tree(tmp_path)
    r = client.post("/api/data/scan", json={"path": "a.jsonl\nsub", "max_records": 100}, headers=MUTATE)
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["files"] == 2 and r.json()["summary"]["scanned"] == 4


# ── uploads, native dialog, LAN access ──────────────────────────────────
def _local(tmp_path, **kw):
    """A client whose peer and Host are loopback: the browser runs on the server's own machine."""
    from fastapi.testclient import TestClient
    app = server.create_app(str(tmp_path), runs_dir="runs", **kw)
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)), app


def test_upload_path_refuses_traversal_hidden_and_executables(tmp_path):
    from chronos.workbench.files import upload_path
    ok = upload_path(str(tmp_path), "data", "batch01", "corpus/part-1.jsonl")
    assert ok == str(tmp_path / "uploads" / "datasets" / "batch01" / "corpus" / "part-1.jsonl")
    assert upload_path(str(tmp_path), "checkpoint", "batch01", "m.config.json").endswith("checkpoints/batch01/m.config.json")
    for kind, batch, name, why in [
        ("data", "batch01", "../x.jsonl", "name"), ("data", "batch01", "/etc/x.jsonl", "name"),
        ("data", "batch01", "a/../../x.jsonl", "name"), ("data", "batch01", ".env/x.jsonl", "name"),
        ("data", "batch01", "a\x00.jsonl", "name"), ("data", "batch01", "/".join("d" * 9) + ".jsonl", "name"),
        ("data", "batch01", "run.py", "type"), ("dir", "batch01", "tools/run.sh", "type"),
        ("config", "batch01", "model.pth", "type"), ("data", "../b", "x.jsonl", "batch"),
        ("data", "ab", "x.jsonl", "batch"), ("exe", "batch01", "x.jsonl", "kind")]:
        with pytest.raises(ValueError) as err:
            upload_path(str(tmp_path), kind, batch, name)
        assert str(err.value) == why, (kind, batch, name)


def test_upload_endpoint_streams_and_validates(tmp_path):
    client, _ = _client(tmp_path, max_upload_mb=1)
    body = b'{"text": "hi"}\n' * 100
    q = {"kind": "data", "batch": "b-123456", "name": "set/a.jsonl"}
    r = client.put("/api/fs/upload", params=q, content=body, headers=MUTATE)
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["path"] == str(tmp_path / "uploads" / "datasets" / "b-123456" / "set" / "a.jsonl")
    assert got["size"] == len(body) and open(got["path"], "rb").read() == body
    assert got["root"] == str(tmp_path / "uploads" / "datasets" / "b-123456")
    assert not [n for n in os.listdir(os.path.dirname(got["path"])) if ".part-" in n]
    assert client.put("/api/fs/upload", params=q, content=body).status_code == 403          # no marker header
    bad = client.put("/api/fs/upload", params={**q, "name": "../a.jsonl"}, content=body, headers=MUTATE)
    assert bad.status_code == 422 and bad.json()["detail"]["error"] == "upload_name"
    typ = client.put("/api/fs/upload", params={**q, "name": "x.sh"}, content=body, headers=MUTATE)
    assert typ.json()["detail"]["error"] == "upload_type"
    big = client.put("/api/fs/upload", params={**q, "name": "big.jsonl"}, content=b"x" * (2**20 + 1), headers=MUTATE)
    assert big.status_code == 413 and big.json()["detail"]["error"] == "upload_too_large"
    assert not os.path.exists(tmp_path / "uploads" / "datasets" / "b-123456" / "big.jsonl")
    locked, _ = _client(tmp_path, api_key="k3y-value")
    assert locked.put("/api/fs/upload", params=q, content=body, headers=MUTATE).status_code == 401


def test_upload_refuses_when_disk_reserve_would_be_crossed(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)
    usage = type("U", (), {"free": 2 * 2**30, "total": 100 * 2**30})
    monkeypatch.setattr(server.shutil, "disk_usage", lambda _p: usage)
    r = client.put("/api/fs/upload", params={"kind": "data", "batch": "b-123456", "name": "a.jsonl"},
                   content=b"{}\n", headers=MUTATE)
    assert r.status_code == 507 and r.json()["detail"]["error"] == "upload_disk_full"


def test_native_dialog_only_for_the_servers_own_machine(tmp_path, monkeypatch):
    from chronos.webui import native_dialog
    seen = {}
    monkeypatch.setattr(native_dialog, "backend", lambda: "osascript")
    monkeypatch.setattr(native_dialog, "choose", lambda **kw: seen.update(kw) or ["/data/a.jsonl", "/data/b.csv"])
    remote, _ = _client(tmp_path)
    r = remote.post("/api/fs/native", json={"kind": "data", "multiple": True}, headers=MUTATE)
    assert r.status_code == 403 and r.json()["detail"]["error"] == "native_local_only"
    local, _ = _local(tmp_path)
    r = local.post("/api/fs/native", json={"kind": "data", "multiple": True, "title": "选择数据集"}, headers=MUTATE)
    assert r.status_code == 200 and r.json() == {"paths": ["/data/a.jsonl", "/data/b.csv"], "cancelled": False}
    assert seen["folder"] is False and seen["multiple"] is True and seen["title"] == "选择数据集"
    assert seen["start"] == str(tmp_path)
    local.post("/api/fs/native", json={"kind": "dir"}, headers=MUTATE)
    assert seen["folder"] is True
    # a proxy in front means the browser is elsewhere
    proxied = local.post("/api/fs/native", json={"kind": "data"}, headers={**MUTATE, "X-Forwarded-For": "10.0.0.9"})
    assert proxied.status_code == 403
    monkeypatch.setattr(native_dialog, "choose", lambda **kw: None)
    assert local.post("/api/fs/native", json={"kind": "data"}, headers=MUTATE).json() == {"paths": [], "cancelled": True}

    def busy(**_kw):
        raise native_dialog.DialogBusy("open")
    monkeypatch.setattr(native_dialog, "choose", busy)
    assert local.post("/api/fs/native", json={"kind": "data"}, headers=MUTATE).status_code == 409
    off, _ = _local(tmp_path, native_dialog=False)
    assert off.post("/api/fs/native", json={"kind": "data"}, headers=MUTATE).status_code == 501
    assert local.post("/api/fs/native", json={"kind": "exe"}, headers=MUTATE).status_code == 422


def test_lan_mode_trusts_only_the_local_browser(tmp_path, monkeypatch):
    from chronos.webui import native_dialog
    monkeypatch.setattr(native_dialog, "backend", lambda: "osascript")
    kw = dict(api_key="k3y-value", trust_local=True, bind_host="0.0.0.0", extra_hosts=["192.168.1.20"])
    local, _ = _local(tmp_path, **kw)
    assert local.get("/api/jobs").status_code == 200                       # this machine: no login
    s = local.get("/api/session").json()
    assert s["access"] == "local" and s["auth_required"] is False and s["lan_enabled"] is True and s["native_dialog"]
    from fastapi.testclient import TestClient
    app = server.create_app(str(tmp_path), runs_dir="runs", **kw)
    lan = TestClient(app, base_url="http://192.168.1.20", client=("192.168.1.30", 50000))
    assert lan.get("/api/jobs").status_code == 401                         # another machine: key required
    s = lan.get("/api/session").json()
    assert s["access"] == "lan" and s["auth_required"] is True and s["native_dialog"] is False
    # a loopback peer that typed the LAN address is not "local" (Host is not loopback)
    via_lan_url = TestClient(app, base_url="http://192.168.1.20", client=("127.0.0.1", 50000))
    assert via_lan_url.get("/api/jobs").status_code == 401
    assert lan.post("/api/login", json={"key": "k3y-value"}, headers=MUTATE).status_code == 200
    assert lan.get("/api/jobs").status_code == 200
    strict, _ = _local(tmp_path, api_key="k3y-value")                     # --allow-remote without --lan
    assert strict.get("/api/jobs").status_code == 401


def test_lan_addresses_are_plain_ipv4():
    addrs = server.lan_addresses()
    assert isinstance(addrs, list)
    assert all(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", a) and not a.startswith("127.") for a in addrs)


def test_every_word_like_choice_has_a_translation():
    """Chips and dropdown options show words in the user's language; only device / format ids stay as they are.

    Mirrors forms.js:choiceLabel (field-scoped key, plain key, stage names, auto-tune field names).
    """
    technical = re.compile(r"(cpu|cuda|mps|mlx|xpu|float16|bfloat16|float32|int8|pt|safetensors|[\w.-]+-gguf|"
                           r"[\w.-]+-safetensors)")
    for lang in schema.LANGUAGES:
        cat = _catalog(lang)
        missing = []
        for f in schema.schema_payload()["fields"]:
            if f["type"] not in ("choice", "multi"):
                continue
            short = f["key"].split(".", 1)[-1]
            for c in f.get("choices") or []:
                if not isinstance(c, str) or not c or technical.fullmatch(c):
                    continue
                keys = [f"choice.{f['key']}.{c}", f"choice.{short}.{c}", f"choice.{c}"]
                keys += [f"stage.{c}"] if f["key"] == "train.stages" else []
                keys += [f"field.{c}.label"] if f["key"] == "tune.search" else []
                if not any(k in cat for k in keys):
                    missing.append((f["key"], c))
        assert not missing, (lang, missing)
