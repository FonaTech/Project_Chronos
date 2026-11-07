"""Training job worker: ``python -m chronos.workbench.train_worker <job.json>``.

Runs one or more ``TrainSession`` stages in sequence and writes a run
directory that ``chronos.monitor.snapshot.RunMonitor`` (and the Web UI live
monitor) can read:

* ``run_spec.json``  – what was asked for (config, stages, backend);
* ``status.json``    – heartbeat: pid, state, current phase, step, error;
* ``metrics.jsonl``  – one row per logged step, plus ``validation`` /
  ``phase_start`` / ``phase_end`` / ``final`` events;
* ``train.log``      – the session log;
* ``samples.jsonl``  – live generation samples (prompt → output);
* ``lazy_probe.jsonl`` – after each validation, the lazy-expert probe
  (``chronos.workbench.lazy_probe``): efficiency, hit rate and fidelity of
  the just-saved checkpoint under every requested paging / residency /
  RAM-ratio / miss-policy / wait / speculative setting.

The job file is written by the Web UI server; it is JSON with
``{"run_dir", "config", "stages": [{"mode", "data_path", "data_paths", "init_weight",
"teacher_path", "max_steps", "val_ratio", "sample_prompt"}], "backend",
"lazy_probe": {"enabled": true, "sweep": {...ProbeSweep fields}}}``.
SIGTERM (the UI stop button) stops the current stage and skips the rest.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime

STAGE_MODES = ("pretrain", "sft", "dpo", "orpo", "grpo", "distill")

# TrainSession reports three stage-specific terms; name them for the monitor.
TERM_NAMES = {
    "pretrain": ("primary_loss", "balance_loss", "temporal_loss"),
    "sft": ("primary_loss", "lookahead_loss", "anchor_loss"),
    "dpo": ("primary_loss", "anchor_loss", "lookahead_loss"),
    "orpo": ("primary_loss", "or_loss", "anchor_loss"),
    "grpo": ("primary_loss", "kl_loss", "reward_mean"),
    "distill": ("primary_loss", "ce_loss", "anchor_loss"),
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


class RunWriter:
    """Appends monitor-format rows and keeps the heartbeat current."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        self.log = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")
        self.samples_path = os.path.join(run_dir, "samples.jsonl")
        self.status = {"pid": os.getpid(), "state": "starting", "started": _now()}
        self.t0 = time.monotonic()

    def row(self, payload: dict) -> None:
        self.metrics.write(json.dumps({"time": _now(), **payload}, ensure_ascii=False, default=float) + "\n")
        self.metrics.flush()

    def lines(self, text: str) -> None:
        if text:
            self.log.write(text.rstrip("\n") + "\n")
            self.log.flush()

    def sample(self, payload: dict) -> None:
        with open(self.samples_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": _now(), **payload}, ensure_ascii=False) + "\n")

    def heartbeat(self, **fields) -> None:
        self.status.update(fields, updated=_now(), run_wall_s=round(time.monotonic() - self.t0, 1))
        _write_json(os.path.join(self.run_dir, "status.json"), self.status)

    def close(self) -> None:
        self.metrics.close()
        self.log.close()


def metric_row(mode: str, metric: dict, total_steps: int) -> dict:
    """Map one ``TrainSession`` metric dict onto monitor keys."""
    if metric.get("val"):
        return {"event": "validation", "phase": mode, "step": int(metric.get("step", 0)),
                "valid_ce": float(metric.get("val_loss", metric.get("total", 0.0)))}
    names = TERM_NAMES.get(mode, TERM_NAMES["pretrain"])
    row = {"phase": mode, "step": int(metric.get("step", 0)), "total": int(total_steps or 0),
           "loss": float(metric.get("total", 0.0))}
    for name, key in zip(names, ("primary", "secondary", "tertiary")):
        if metric.get(key) is not None:
            row[name] = float(metric[key])
    if metric.get("tokens_per_s"):
        row["tokens_per_s"] = float(metric["tokens_per_s"])
    if metric.get("tps"):
        row["steps_per_s"] = float(metric["tps"])
        row["step_s"] = 1.0 / float(metric["tps"])
    for key in ("grad_norm", "lr", "lookahead_union_recall", "router_adjacent_jaccard", "recurrent_loss"):
        value = metric.get(key)
        if isinstance(value, (int, float)):
            row[key] = float(value)
    return row


PROBE_GRACE_S = 3.0


def _fresh_checkpoint(candidates, newer_than: float) -> str | None:
    from chronos.workbench.lazy_probe import newest_checkpoint
    fresh = []
    for path in candidates:
        try:
            if os.path.getmtime(path) > newer_than:
                fresh.append(path)
        except OSError:
            continue
    return newest_checkpoint(fresh)


def probe_event(entry: dict) -> dict:
    """A compact ``lazy_probe`` event row for metrics.jsonl (the full result is in lazy_probe.jsonl)."""
    row = {"event": "lazy_probe", "phase": entry.get("phase"), "step": entry.get("step"),
           "wall_s": entry.get("wall_s")}
    if "error" in entry:
        row["error"] = str(entry["error"])[:200]
    result = entry.get("result") or {}
    if result.get("skipped"):
        row["skipped"] = str(result["skipped"])[:200]
    failed = [v for v in result.get("variants") or [] if "error" in v]
    if failed:
        row["probe_errors"] = len(failed)
    variants = [v for v in result.get("variants") or [] if "fidelity" in v]
    if variants:
        row["probe_variants"] = len(variants)
        row["probe_min_exact_match"] = min(v["fidelity"]["exact_match_rate"] for v in variants)
        hits = [v["measured"]["resident_hit_rate"] for v in variants
                if v.get("measured", {}).get("resident_hit_rate") is not None]
        if hits:
            row["probe_mean_resident_hit_rate"] = round(sum(hits) / len(hits), 4)
    return row


def stage_checkpoints(run_cfg: dict, mode: str) -> list[str]:
    """Checkpoint files this stage writes (``<prefix>_<H>_moe[.best].pth`` in ``save_dir``)."""
    import glob
    from chronos.workbench.training import STAGE_CHECKPOINT_PREFIX
    prefix = STAGE_CHECKPOINT_PREFIX.get(mode, mode)
    return glob.glob(os.path.join(run_cfg.get("save_dir", "./out"), f"{prefix}_*_moe*.pth"))


def stage_data_path(stage: dict, writer: RunWriter, cfg: dict) -> str:
    """The one JSONL file a stage trains on.

    ``data_paths`` (several files / folders, any supported format) are merged into
    ``<run_dir>/data/merged_<hash>.jsonl``; a single ``.jsonl`` file is used in place. An empty
    selection keeps the old behaviour (``data_path`` or the config's, else a synthetic smoke run).
    """
    paths = [p for p in stage.get("data_paths") or [] if p]
    if not paths:
        return stage.get("data_path") or cfg.get("data_path", "")
    from chronos.workbench.files import merge_datasets
    try:
        return merge_datasets(paths, os.path.join(writer.run_dir, "data"), log=writer.lines)
    except FileNotFoundError:
        raise FileNotFoundError(f"no readable records in {', '.join(paths)}") from None


def run_stage(session, writer: RunWriter, cfg: dict, stage: dict, backend: str, stop, probe=None) -> str:
    from chronos.workbench.training import _prepare_training_run_config
    mode = stage["mode"]
    data_path = stage_data_path(stage, writer, cfg)
    run_cfg, effective = _prepare_training_run_config(
        cfg, mode, backend, data_path, None,
        stage.get("init_weight", ""), stage.get("max_steps", cfg.get("max_steps", 0)),
        stage.get("val_ratio", 0.0), stage.get("teacher_path", ""), stage.get("sample_prompt", ""))
    if stage.get("max_steps") is not None:
        run_cfg["max_steps"] = int(stage["max_steps"] or 0)
    run_cfg["cleanup_gradio_temp_dataset"] = False          # never delete a user dataset
    session.effective_config = effective
    writer.row({"event": "phase_start", "phase": mode, "step": 0})
    writer.heartbeat(state="running", phase=mode, phase_started=_now())
    session.start(run_cfg, mode)
    seen, last_sample, last_beat = 0, "", 0.0
    pending = None                       # validation step whose checkpoint has not been probed yet
    probed_mtime = time.time()           # only checkpoints newer than this belong to this stage / probe
    while True:
        alive = session.is_running()
        writer.lines(session.drain_log())
        rows = session.get_metrics()
        for metric in rows[seen:]:
            row = metric_row(mode, metric, session.total_steps)
            writer.row(row)
            if probe is not None and row.get("event") == "validation":
                if pending is not None:          # validations arrive faster than checkpoints settle
                    probe.skip(phase=mode, step=pending["step"], reason=f"superseded by step {row['step']}")
                pending = {"step": row["step"], "t": time.monotonic()}
        seen = len(rows)
        if probe is not None and pending is not None:
            # Torch saves before validating; MLX validates, then may save ``.best`` and the final
            # checkpoint.  Give the trainer a short grace period, then probe the newest checkpoint
            # written since the last probe (the weights that were just validated).
            ckpt = _fresh_checkpoint(stage_checkpoints(run_cfg, mode), probed_mtime)
            due = time.monotonic() - pending["t"] >= PROBE_GRACE_S or not alive
            if due and (ckpt is not None or not alive):
                probe.request(ckpt, phase=mode, step=pending["step"])
                if ckpt is not None:
                    probed_mtime = os.path.getmtime(ckpt)
                pending = None
        if probe is not None:
            for entry in probe.poll():
                writer.row(probe_event(entry))
        if session.last_sample and session.last_sample != last_sample:
            last_sample = session.last_sample
            writer.sample({"phase": mode, "step": session.last_sample_step,
                           "prompt": run_cfg.get("sample_prompt") or "", "output": last_sample})
        if time.monotonic() - last_beat > 2.0 or not alive:
            writer.heartbeat(step=session.step, total=session.total_steps, loss=session.loss,
                             effective_config=session.get_effective_config())
            last_beat = time.monotonic()
        if stop["requested"] and alive:
            session.stop()
        if not alive:
            break
        time.sleep(0.5)
    state = session.status if session.status in ("finished", "stopped", "error") else "finished"
    writer.row({"event": "phase_end", "phase": mode, "step": session.step, "state": state})
    return state


def run_job(job: dict) -> int:
    from chronos.workbench.training import TrainSession
    run_dir = os.path.abspath(job["run_dir"])
    stages = [s for s in job.get("stages", []) if s.get("mode") in STAGE_MODES]
    if not stages:
        raise ValueError("no valid stages")
    writer = RunWriter(run_dir)
    _write_json(os.path.join(run_dir, "run_spec.json"), {
        "run_dir": run_dir, "source": "chronos-webui", "backend": job.get("backend", "auto"),
        "phases": [{"name": s["mode"], "budget_fraction": round(1 / len(stages), 4),
                    "data_path": s.get("data_path", ""), "data_paths": s.get("data_paths") or [],
                    "max_steps": s.get("max_steps")} for s in stages],
        "model_config": job.get("config", {}), "created": _now(),
    })
    stop = {"requested": False}
    probe = None
    probe_cfg = job.get("lazy_probe") or {}
    if probe_cfg.get("enabled", True):
        from chronos.workbench.lazy_probe import ProbeScheduler
        probe = ProbeScheduler(run_dir, probe_cfg.get("sweep"), tokenizer_dir=job.get("config", {}).get(
            "tokenizer_path") or None, log=writer.lines)

    def _on_term(_sig, _frm):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    writer.heartbeat(state="starting", stages=[s["mode"] for s in stages])
    final, error = "finished", None
    try:
        for stage in stages:
            session = TrainSession()
            state = run_stage(session, writer, dict(job.get("config", {})), stage, job.get("backend", "auto"),
                              stop, probe)
            if state == "error":
                final, error = "failed", f"stage {stage['mode']} failed (see train.log)"
                break
            if stop["requested"] or state == "stopped":
                final = "stopped"
                break
    except Exception as exc:                      # recorded for the UI, then re-raised for the exit code
        final, error = "failed", f"{type(exc).__name__}: {exc}"
        writer.lines(error)
        raise
    finally:
        if probe is not None:
            if stop["requested"]:
                probe.stop()
            for entry in probe.wait():
                writer.row(probe_event(entry))
        writer.row({"event": "final", "state": final, **({"error": error} if error else {})})
        writer.heartbeat(state="finished" if final == "finished" else ("failed" if final == "failed" else "stopped"),
                         error=error)
        writer.close()
    return 0 if final == "finished" else (1 if final == "failed" else 0)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m chronos.workbench.train_worker <job.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as fh:
        return run_job(json.load(fh))


if __name__ == "__main__":
    raise SystemExit(main())
