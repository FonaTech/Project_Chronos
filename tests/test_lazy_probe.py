"""Mid-training lazy-expert probe (chronos.workbench.lazy_probe) and its train-worker wiring."""
from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.workbench.lazy_probe import (  # noqa: E402
    ProbeScheduler, ProbeSweep, agreement, expand_variants, newest_checkpoint, variant_label,
)


def test_sweep_axes_collapse_per_backend_and_dedupe_by_resolved_layout(monkeypatch):
    # Backends resolve as requested here, so the test does not depend on MLX being installed (Linux CI).
    from chronos.workbench import lazy_probe
    monkeypatch.setattr(lazy_probe, "resolve_backend", lambda b: b)
    sweep = ProbeSweep.from_dict({"backends": ["cpu", "mlx"], "paging": ["auto", "cluster", "expert"],
                                  "residency": ["planned", "window"], "ram_ratios": [0.5],
                                  "miss_policies": ["sync_on_demand"], "speculative": [False, True],
                                  "max_variants": 256})
    variants, total = expand_variants(sweep, lambda p: "cluster" if p == "auto" else p)
    assert total == len(variants)
    torch_v = [v for v in variants if v["backend"] == "cpu"]
    mlx_v = [v for v in variants if v["backend"] == "mlx"]
    assert len(torch_v) == 2 * 2                          # {cluster, expert} × spec; residency n/a
    assert len(mlx_v) == 2 * 2 * 2                        # layouts × residency × spec
    assert all("residency" not in v and v["miss_policy"] == "sync_on_demand" for v in torch_v)
    assert all("residency_decay" in v for v in mlx_v if v["residency"] == "planned")
    assert all("residency_decay" not in v for v in mlx_v if v["residency"] == "window")
    assert all(("verify_wait_s" in v) == v["speculative"] for v in mlx_v)
    capped, total = expand_variants(ProbeSweep.from_dict({**sweep.as_dict(), "max_variants": 3}),
                                    lambda p: p)
    assert len(capped) == 3 and total > 3
    assert variant_label(mlx_v[0]).startswith("mlx/cluster/r0.5/planned/d0.7/w12ms")
    # A backend this machine lacks resolves to one it has; duplicates then collapse into one set.
    monkeypatch.setattr(lazy_probe, "resolve_backend", lambda b: "cpu")
    fallback, _ = expand_variants(sweep, lambda p: "cluster" if p == "auto" else p)
    assert {v["backend"] for v in fallback} == {"cpu"} and len(fallback) == len(torch_v)


@pytest.mark.parametrize("bad", [dict(paging=["louvain"]), dict(residency=["queue"]), dict(ram_ratios=[0]),
                                 dict(residency_decay=[1.0]), dict(decode_wait_s=[-1]),
                                 dict(miss_policies=["drop"]), dict(max_new_tokens=0), dict(gamma=9)])
def test_sweep_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        ProbeSweep.from_dict(bad)


def test_agreement_and_newest_checkpoint(tmp_path):
    assert agreement([1, 2, 3], [1, 2, 3]) == {"exact": True, "prefix": 3, "positionwise": 1.0, "ref_len": 3}
    a = agreement([1, 9, 3, 4], [1, 2, 3])
    assert not a["exact"] and a["prefix"] == 1 and a["positionwise"] == pytest.approx(2 / 3)
    old, new = tmp_path / "a.pth", tmp_path / "b.pth"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    os.utime(old, (1, 1))
    assert newest_checkpoint([str(old), str(new), str(tmp_path / "missing.pth")]) == str(new)
    assert newest_checkpoint([]) is None


torch = pytest.importorskip("torch")


@pytest.fixture
def trained_ckpt(tmp_path, monkeypatch):
    from test_lazy_experts import _save
    monkeypatch.setenv("CHRONOS_EXPERT_CACHE_ROOT", str(tmp_path / "cache"))
    return _save(tmp_path, 0.3, "trained")


def test_probe_torch_sync_is_exact_and_fallback_is_reported(trained_ckpt):
    from chronos.workbench.lazy_probe import run_probe
    res = run_probe(trained_ckpt, {"backends": ["cpu"], "paging": ["auto", "expert"], "ram_ratios": [0.5],
                                   "miss_policies": ["sync_on_demand", "fallback_diagnostic"],
                                   "max_new_tokens": 8})
    assert res["auto_paging"] == "cluster" and res["conditions"]["prompt_source"] == "synthetic"
    by = {v["label"]: v for v in res["variants"]}
    assert set(by) == {"cpu/cluster/r0.5/sync_on_demand", "cpu/cluster/r0.5/fallback_diagnostic",
                       "cpu/expert/r0.5/sync_on_demand", "cpu/expert/r0.5/fallback_diagnostic"}
    for label, v in by.items():
        assert "error" not in v, v
        m = v["measured"]
        assert 0.0 <= m["resident_hit_rate"] <= 1.0 and m["tokens"] > 0 and m["tokens_per_s"] > 0
        assert m["paging_resolved"] == label.split("/")[1]
    for layout in ("cluster", "expert"):
        sync = by[f"cpu/{layout}/r0.5/sync_on_demand"]
        assert sync["fidelity"]["exact_match_rate"] == 1.0        # a miss loads: identical to full memory
        assert sync["measured"]["on_demand_loads"] > 0
        assert by[f"cpu/{layout}/r0.5/fallback_diagnostic"]["measured"]["on_demand_loads"] == 0
    assert res["references"]["cpu"]["measured"]["tokens_per_s"] > 0


def test_probe_mlx_variants_are_exact(trained_ckpt):
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        from chronos.workbench.lazy_probe import run_probe
        res = run_probe(trained_ckpt, {"backends": ["mlx"], "paging": ["auto", "expert"], "ram_ratios": [0.5],
                                       "residency": ["planned", "window"], "speculative": [False, True],
                                       "max_new_tokens": 8})
    finally:
        mx.set_default_device(previous)
    assert len(res["variants"]) == 8 and not res["truncated"]
    for v in res["variants"]:
        assert "error" not in v, v
        assert v["fidelity"]["exact_match_rate"] == 1.0, v["label"]
        assert v["measured"]["prediction_hit_rate"] is None or 0 <= v["measured"]["prediction_hit_rate"] <= 1


def test_dense_checkpoint_is_skipped(tmp_path, monkeypatch):
    from chronos.model.checkpoint import save_state_dict_with_config
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.workbench.lazy_probe import run_probe
    from test_lazy_experts import _tiny
    cfg = _tiny(0.0)
    cfg.use_moe = False
    path = str(tmp_path / "dense.pth")
    save_state_dict_with_config(ChronosForCausalLM(cfg).eval(), path, cfg, stage="pretrain")
    assert "skipped" in run_probe(path, {"backends": ["cpu"]})


def test_scheduler_snapshots_runs_in_child_and_cleans_up(trained_ckpt, tmp_path):
    run_dir = tmp_path / "run"
    sched = ProbeScheduler(str(run_dir), {"backends": ["cpu"], "paging": ["auto"], "ram_ratios": [1.0],
                                          "max_new_tokens": 4}, nice=0)
    assert "skipped" in sched.request(str(tmp_path / "nope.pth"), phase="pretrain", step=1)
    assert sched.request(trained_ckpt, phase="pretrain", step=10)["started"]
    assert "skipped" in sched.request(trained_ckpt, phase="pretrain", step=11)   # one probe at a time
    entries = sched.wait(timeout_s=300)
    assert len(entries) == 1 and "error" not in entries[0], entries
    variant = entries[0]["result"]["variants"][0]
    assert variant["label"] == "cpu/cluster/r1/sync_on_demand" and variant["fidelity"]["exact_match_rate"] == 1.0
    assert not (run_dir / "lazy_probe").exists() or not os.listdir(run_dir / "lazy_probe")
    lines = [json.loads(x) for x in (run_dir / "lazy_probe.jsonl").read_text().splitlines()]
    assert [("skipped" in x, x["step"]) for x in lines] == [(True, 1), (True, 11), (False, 10)]
    assert os.path.exists(trained_ckpt)                    # the original checkpoint is untouched


class _StubSession:
    """TrainSession stand-in: writes the stage checkpoint, then reports a validation."""

    def __init__(self, source_ckpt: str, delay_s: float = 0.0):
        self.source, self.delay = source_ckpt, delay_s
        self.status, self.step, self.loss, self.total_steps = "idle", 0, 0.0, 4
        self.last_sample, self.last_sample_step, self.effective_config = "", 0, {}
        self._metrics, self._thread = [], None

    def start(self, cfg, mode):
        import shutil
        import threading
        import time

        def run():
            self.status = "running"
            for step in (1, 2, 3, 4):
                self.step, self.loss = step, 5.0 - step
                self._metrics.append({"step": step, "total": self.loss, "primary": self.loss,
                                      "secondary": 0.1, "tertiary": 0.0, "tps": 10.0})
            time.sleep(0.6)
            # MLX order: validation first, the checkpoint shortly after.
            self._metrics.append({"step": 4, "total": 1.5, "val": True, "val_loss": 1.5})
            time.sleep(self.delay)
            os.makedirs(cfg["save_dir"], exist_ok=True)
            sidecar = os.path.splitext(self.source)[0] + ".config.json"
            if os.path.exists(sidecar):                    # save_state_dict_with_config writes both
                shutil.copy(sidecar, os.path.join(cfg["save_dir"], "chronos_32_moe.config.json"))
            shutil.copy(self.source, os.path.join(cfg["save_dir"], "chronos_32_moe.pth"))
            time.sleep(0.6)
            self.status = "finished"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        pass

    def drain_log(self):
        return ""

    def get_metrics(self):
        return list(self._metrics)

    def get_effective_config(self):
        return {}


def test_worker_probes_checkpoint_after_validation_and_monitor_shows_it(trained_ckpt, tmp_path, monkeypatch):
    from chronos.monitor.snapshot import RunMonitor
    from chronos.workbench import train_worker
    from chronos.workbench import training

    monkeypatch.setattr(train_worker, "PROBE_GRACE_S", 0.5)
    monkeypatch.setattr(training, "TrainSession", lambda: _StubSession(trained_ckpt, delay_s=1.0))
    run_dir = tmp_path / "runs" / "webui_job"
    job = {"run_dir": str(run_dir), "backend": "cpu",
           "config": {"save_dir": str(tmp_path / "out"), "hidden_size": 32},
           "stages": [{"mode": "pretrain", "max_steps": 4, "val_ratio": 0.1}],
           "lazy_probe": {"enabled": True, "sweep": {"backends": ["cpu"], "paging": ["auto", "expert"],
                                                      "ram_ratios": [0.5], "max_new_tokens": 4}}}
    assert train_worker.run_job(job) == 0
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    kinds = [r.get("event") for r in rows if r.get("event")]
    assert kinds.index("validation") < kinds.index("lazy_probe") < kinds.index("final")
    probe_row = next(r for r in rows if r.get("event") == "lazy_probe")
    assert probe_row["step"] == 4 and probe_row["probe_variants"] == 2 and "error" not in probe_row
    assert probe_row["probe_min_exact_match"] == 1.0                 # sync_on_demand matches full memory
    train_row = next(r for r in rows if "event" not in r)
    assert train_row["primary_loss"] == 4.0 and train_row["balance_loss"] == 0.1

    snap = RunMonitor(str(tmp_path / "runs")).snapshot("webui_job")
    assert snap["status"]["state"] == "finished"
    (probe,) = snap["lazy_probe"]
    assert probe["step"] == 4 and probe["auto_paging"] == "cluster"
    assert {v["label"] for v in probe["variants"]} == {"cpu/cluster/r0.5/sync_on_demand",
                                                       "cpu/expert/r0.5/sync_on_demand"}
    for v in probe["variants"]:
        assert 0 <= v["measured"]["resident_hit_rate"] <= 1 and v["fidelity"]["exact_match_rate"] == 1.0
        assert set(RunMonitor.PROBE_MEASURED) <= set(v["measured"])          # every lazy metric reaches the UI
        assert set(RunMonitor.PROBE_FIDELITY) <= set(v["fidelity"])
        assert v["measured"]["prefill_time_s"] >= 0 and v["measured"]["on_demand_load_time_s"] >= 0
    assert probe["conditions"]["label"] == "measured"
    ref = probe["reference_measured"]["cpu"]
    assert ref["tokens_per_s"] > 0 and ref["tokens_per_s"] == probe["references"]["cpu"]

    from fastapi.testclient import TestClient
    from chronos.monitor.server import create_monitor_app
    client = TestClient(create_monitor_app(str(tmp_path / "runs")))
    body = client.get("/api/lazy_probes", params={"run": "webui_job"}).json()
    assert body["probes"] == snap["lazy_probe"]
    assert client.get("/api/lazy_probes", params={"run": "missing"}).status_code == 404


def test_worker_records_skip_when_validation_has_no_checkpoint(tmp_path, monkeypatch):
    from chronos.workbench import train_worker
    from chronos.workbench import training

    class NoSave(_StubSession):
        def start(self, cfg, mode):
            super().start({**cfg, "save_dir": str(tmp_path / "elsewhere")}, mode)

    src = tmp_path / "not_a_model.pth"
    src.write_bytes(b"x")
    monkeypatch.setattr(train_worker, "PROBE_GRACE_S", 0.2)
    monkeypatch.setattr(training, "TrainSession", lambda: NoSave(str(src)))
    run_dir = tmp_path / "run"
    job = {"run_dir": str(run_dir), "backend": "cpu", "config": {"save_dir": str(tmp_path / "out")},
           "stages": [{"mode": "pretrain", "max_steps": 4}], "lazy_probe": {"sweep": {"backends": ["cpu"]}}}
    assert train_worker.run_job(job) == 0
    lines = [json.loads(x) for x in (run_dir / "lazy_probe.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and "skipped" in lines[0] and lines[0]["step"] == 4
