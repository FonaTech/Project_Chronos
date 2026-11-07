"""Run snapshot: everything the monitor pages show, as plain JSON.

``RunMonitor(root).snapshot(run)`` merges
* the full run spec (phases, LR recipe, model config, budget) — training
  parameters are always visible, through the API and both pages;
* status: ``status.json`` heartbeat when the run writes one, otherwise
  derived from the metrics stream plus the process table (older runs);
* progress, ETA and wall-budget accounting per phase;
* the newest metric row, a downsampled history per phase, validation,
  rollback and final events;
* per-tensor parameter statistics (``parameters.jsonl``), evaluation samples
  (``samples.jsonl``, any task), offline inspection reports, checkpoints on disk;
* the evaluation metrics the run actually logs, discovered from its events
  (:func:`chronos.monitor.evals.eval_metrics`), so no page assumes a task;
* OS-level memory of the live training process.
"""
from __future__ import annotations

import glob
import os
import statistics
import threading
import time
from datetime import datetime

from chronos.monitor.evals import eval_metrics, is_context_key, sample_view
from chronos.monitor.jsonl import JsonLines, read_json, tail_text
from chronos.monitor.process import PROCESS_MEMORY, find_run_pid, pid_alive, process_state

METRIC_GROUPS = {
    "loss": ("loss", "primary_loss", "balance_loss", "lookahead_loss", "lookahead_topk_loss",
             "lookahead_union_loss", "temporal_loss", "locality_loss", "anchor_loss",
             "recurrent_ponder_loss", "recurrent_distill_loss", "recurrent_stability_loss"),
    "optimizer": ("lr", "grad_norm", "update_ratio"),
    "throughput": ("tokens_per_s", "processed_tokens_per_s", "step_s", "tokens_seen"),
    "memory": ("peak_mem_gb", "active_mem_gb", "cache_mem_gb", "footprint_gb"),
    "recurrent": ("rec_depth_mean", "rec_contraction", "rec_early_exit_ratio", "rec_ponder_loss",
                  "rec_distill_loss", "rec_stability_loss"),
    "routing": ("route_entropy", "load_cv", "dead_experts", "max_expert_load"),
}
EVENT_KINDS = ("validation", "rollback", "final", "phase_start", "phase_end", "resume")


def downsample(rows: list[dict], limit: int) -> list[dict]:
    """Evenly strided subset that always keeps the newest row."""
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    stride = len(rows) / limit
    picked = [rows[int(i * stride)] for i in range(limit - 1)]
    return picked + [rows[-1]]


def _age_s(stamp) -> float | None:
    if not stamp:
        return None
    try:
        if isinstance(stamp, (int, float)):
            return max(0.0, time.time() - float(stamp))
        return max(0.0, time.time() - datetime.fromisoformat(str(stamp)).timestamp())
    except (TypeError, ValueError):
        return None


def _numeric_leaves(obj, prefix: str, out: dict, depth: int = 0) -> None:
    if not isinstance(obj, dict) or depth > 3:
        return
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _numeric_leaves(v, key, out, depth + 1)
        elif (isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")
              and not is_context_key(str(k))):
            out[key] = v


def _safe_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..") and not name.startswith(".")


class RunMonitor:
    """Read-only view over ``<root>/<run>/`` directories."""

    def __init__(self, root: str, repo_root: str | None = None, history_points: int = 600):
        self.root = os.path.abspath(root)
        self.repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.history_points = history_points
        self._streams: dict[str, JsonLines] = {}
        self._lock = threading.Lock()

    # ---- discovery ---------------------------------------------------------
    def runs(self) -> list[dict]:
        out = []
        try:
            names = sorted(os.listdir(self.root))
        except OSError:
            return out
        for name in names:
            path = os.path.join(self.root, name)
            if not _safe_name(name) or not os.path.isdir(path):
                continue
            if not any(os.path.exists(os.path.join(path, f)) for f in ("run_spec.json", "metrics.jsonl", "status.json")):
                continue
            try:
                mtime = max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))
            except (OSError, ValueError):
                mtime = 0.0
            out.append({"name": name, "updated": mtime})
        out.sort(key=lambda r: -r["updated"])
        return out

    def run_dir(self, run: str | None) -> str | None:
        runs = self.runs()
        if not runs:
            return None
        name = run or runs[0]["name"]
        if not _safe_name(name) or name not in {r["name"] for r in runs}:
            raise KeyError(f"unknown run {name!r}")
        return os.path.join(self.root, name)

    def _stream(self, path: str, max_rows: int | None = None) -> JsonLines:
        with self._lock:
            reader = self._streams.get(path)
            if reader is None:
                reader = self._streams[path] = JsonLines(path, max_rows=max_rows)
            return reader

    # ---- status --------------------------------------------------------------
    def _status(self, path: str, spec: dict | None, rows: list[dict]) -> dict:
        heartbeat = read_json(os.path.join(path, "status.json")) or {}
        pid = heartbeat.get("pid") if isinstance(heartbeat.get("pid"), int) else None
        source = "status.json" if heartbeat else "derived"
        if pid is None or not pid_alive(pid):
            found = find_run_pid(path, self.repo_root)
            pid = found if found is not None else pid
        alive = pid_alive(pid)
        ps = process_state(pid) if alive else None
        last = rows[-1] if rows else {}
        finished = any(r.get("event") == "final" for r in rows[-50:])
        declared = heartbeat.get("state")
        if declared in ("finished", "failed") and (not alive or finished):
            # A heartbeat that declares the end, backed by a ``final`` event, wins even when the pid
            # is still alive (the writer runs inside a longer-lived process, or the pid was reused).
            state = declared
        elif finished and not alive:
            state = "finished"
        elif alive and ps and ps.startswith("T"):
            state = "paused"                # SIGSTOP (e.g. benchmark pause)
        elif alive:
            state = declared if declared in ("starting", "evaluating", "checkpointing") else "running"
        else:
            state = "stopped" if rows or heartbeat else "pending"
        stamps = [r.get("time") for r in rows[-6:] if r.get("time") and "event" not in r]
        intervals = []
        for a, b in zip(stamps, stamps[1:]):
            try:
                intervals.append(datetime.fromisoformat(b).timestamp() - datetime.fromisoformat(a).timestamp())
            except ValueError:
                pass
        interval = statistics.median(intervals) if intervals else None
        age = _age_s(heartbeat.get("updated") or last.get("time"))
        stale = bool(state == "running" and age is not None and age > max(180.0, 3 * (interval or 60.0)))
        return {"state": state, "source": source, "pid": pid, "alive": alive, "ps_state": ps,
                "last_update_age_s": age, "log_interval_s": interval, "stale": stale,
                "heartbeat": heartbeat or None, "error": heartbeat.get("error")}

    # ---- progress / ETA ------------------------------------------------------
    @staticmethod
    def _progress(spec: dict | None, status: dict, latest: dict | None, rows: list[dict]) -> dict:
        phases = (spec or {}).get("phases") or []
        hb = status.get("heartbeat") or {}
        budget_h = float((spec or {}).get("budget_hours") or 0.0)
        reserve = float((spec or {}).get("eval_reserve_fraction") or 0.0)
        train_s = budget_h * 3600 * (1 - reserve)
        current = hb.get("phase") or (latest or {}).get("phase")
        names = [p.get("name") for p in phases]
        index = names.index(current) if current in names else 0
        out = []
        for i, phase in enumerate(phases):
            prows = [r for r in rows if r.get("phase") == phase.get("name") and "event" not in r]
            last = prows[-1] if prows else {}
            entry = {"name": phase.get("name"), "budget_h": round(train_s * float(phase.get("budget_fraction", 0)) / 3600, 2),
                     "step": last.get("step", 0), "total": last.get("total"),
                     "state": "complete" if i < index else ("active" if i == index else "pending")}
            if i == index and status.get("state") == "finished":
                entry["state"] = "complete"
            out.append(entry)
        result = {"phases": out, "current": current, "budget_hours": budget_h, "train_budget_hours": round(train_s / 3600, 2)}
        if hb.get("phase_wall_s") is not None and hb.get("phase_budget_s"):
            result["phase_fraction"] = min(1.0, float(hb["phase_wall_s"]) / float(hb["phase_budget_s"]))
            result["phase_remaining_s"] = max(0.0, float(hb["phase_budget_s"]) - float(hb["phase_wall_s"]))
        elif latest and latest.get("total"):
            result["phase_fraction"] = min(1.0, latest["step"] / max(1, latest["total"]))
            result["phase_remaining_s"] = max(0, latest["total"] - latest["step"]) * float(latest.get("step_s") or 0)
        if result.get("phase_remaining_s") is not None:
            later = sum(e["budget_h"] for e in out[index + 1:]) * 3600
            result["run_remaining_s"] = result["phase_remaining_s"] + later + budget_h * 3600 * reserve
            result["eta"] = datetime.fromtimestamp(time.time() + result["run_remaining_s"]).isoformat(timespec="minutes")
        if hb.get("run_wall_s") is not None:
            result["run_wall_s"] = hb["run_wall_s"]
        return result

    # ---- full snapshot -------------------------------------------------------
    def snapshot(self, run: str | None = None, *, history: bool = True) -> dict:
        path = self.run_dir(run)
        if path is None:
            return {"runs": [], "run": None, "status": {"state": "pending"}, "root": self.root}
        rows = self._stream(os.path.join(path, "metrics.jsonl")).read()
        spec = read_json(os.path.join(path, "run_spec.json"))
        status = self._status(path, spec, rows)
        train_rows = [r for r in rows if "event" not in r]
        events = [r for r in rows if r.get("event") in EVENT_KINDS or (r.get("event") and r.get("event") not in EVENT_KINDS)]
        latest = train_rows[-1] if train_rows else None
        memory = PROCESS_MEMORY.read(status["pid"]) if status["alive"] else None
        best = {}
        for r in events:
            if r.get("event") == "validation" and r.get("valid_ce") is not None:
                key = r.get("phase", "?")
                if key not in best or r["valid_ce"] < best[key]["valid_ce"]:
                    best[key] = {"step": r.get("step"), "valid_ce": r["valid_ce"], "time": r.get("time")}
        params = self._stream(os.path.join(path, "parameters.jsonl"), max_rows=64).read()
        latest_params = params[-1] if params else None
        out = {
            "root": self.root, "runs": self.runs(), "run": os.path.basename(path), "generated": time.time(),
            "spec": spec, "runtime": read_json(os.path.join(path, "runtime.json")), "status": status,
            "progress": self._progress(spec, status, latest, rows),
            "latest": latest, "memory": memory,
            "events": events[-200:], "best_valid": best, "evals": eval_metrics(events),
            "groups": {k: list(v) for k, v in METRIC_GROUPS.items()},
            "parameters": self._parameter_summary(latest_params),
            "samples": [sample_view(x) for x in
                        self._stream(os.path.join(path, "samples.jsonl"), max_rows=200).read()[-40:] if isinstance(x, dict)],
            "inspections": self._inspections(path),
            "checkpoints": self._checkpoints(path),
            "lazy_probe": self._lazy_probes(path, limit=20),
            "counts": {"metric_rows": len(train_rows), "events": len(events), "parameter_rows": len(params)},
        }
        if history:
            out["history"] = {name: downsample([r for r in train_rows if r.get("phase") == name], self.history_points)
                              for name in dict.fromkeys(r.get("phase") for r in train_rows)}
        return out

    @staticmethod
    def _parameter_summary(row: dict | None) -> dict | None:
        """Latest telemetry row without the per-tensor list (fetched separately:
        ~700 tensors are ~170 KiB, too much for every status poll)."""
        if not row:
            return None
        return {k: row.get(k) for k in ("step", "phase", "time", "lr", "grad_norm", "stats", "groups",
                                         "experts", "routing", "grad_note", "telemetry_s")} | {
            "tensor_count": len(row.get("tensors") or [])}

    def parameters(self, run: str | None) -> dict | None:
        """Full latest ``parameters.jsonl`` row (every tensor)."""
        path = self.run_dir(run)
        if path is None:
            return None
        rows = self._stream(os.path.join(path, "parameters.jsonl"), max_rows=64).read()
        return rows[-1] if rows else None

    def metrics(self, run: str | None, phase: str | None = None, since_step: int = 0, limit: int = 5000) -> list[dict]:
        path = self.run_dir(run)
        if path is None:
            return []
        rows = [r for r in self._stream(os.path.join(path, "metrics.jsonl")).read()
                if "event" not in r and (phase is None or r.get("phase") == phase) and int(r.get("step") or 0) > since_step]
        return downsample(rows, max(1, min(int(limit), 20000)))

    def training_parameters(self, run: str | None) -> dict:
        """Run spec (recipe, LR schedule, model config) + runtime facts
        (precision policy, device, versions, effective batch, code hash)."""
        path = self.run_dir(run)
        if path is None:
            return {}
        spec = read_json(os.path.join(path, "run_spec.json")) or {}
        out = {"run": os.path.basename(path), "spec": spec, "runtime": read_json(os.path.join(path, "runtime.json"))}
        # Small manifests the run points at (paths come from the run's own spec).
        for key, folder, name in (("data", spec.get("shards_dir"), "shards_manifest.json"),
                                  ("tokenizer", spec.get("tokenizer_dir"), "tokenizer_report.json")):
            if isinstance(folder, str) and folder:
                file = os.path.join(folder, name)
                if os.path.isfile(file) and os.path.getsize(file) < 512 * 1024:
                    data = read_json(file)
                    if isinstance(data, dict) and key == "tokenizer":
                        data = {k: v for k, v in data.items() if k != "special_tokens"} | {
                            "special_tokens_count": len(data.get("special_tokens") or [])}
                    out[key] = data
        return out

    def tensor_history(self, run: str | None, name: str) -> list[dict]:
        path = self.run_dir(run)
        if path is None:
            return []
        rows = self._stream(os.path.join(path, "parameters.jsonl"), max_rows=64).read()
        out = []
        for row in rows:
            for t in row.get("tensors") or []:
                if t.get("name") == name:
                    out.append({"step": row.get("step"), "phase": row.get("phase"), **t})
                    break
        return out

    def log_tail(self, run: str | None, max_bytes: int = 16000) -> str:
        path = self.run_dir(run)
        return tail_text(os.path.join(path, "train.log"), max_bytes) if path else ""

    @staticmethod
    def _inspections(path: str) -> list[dict]:
        """Offline reports: identity fields plus every numeric leaf, flattened
        (``held_out.valid_ce``, ``probe.dev.bleu``, ``qa.accuracy`` …)."""
        meta = ("time", "phase", "step", "tokens_seen", "device", "compute_dtype", "train_dtype", "elapsed_s")
        out = []
        for file in sorted(glob.glob(os.path.join(path, "inspections", "*.json")))[-12:]:
            data = read_json(file)
            if isinstance(data, dict):
                metrics: dict[str, float] = {}
                _numeric_leaves({k: v for k, v in data.items() if k not in meta + ("precision", "run_dir")}, "", metrics)
                out.append({"file": os.path.basename(file), **{k: data.get(k) for k in meta}, "metrics": metrics})
        return out

    # Columns the monitor table shows per probe variant (the full result stays in lazy_probe.jsonl).
    PROBE_MEASURED = ("tokens_per_s", "decode_tokens_per_s", "speed_vs_full", "tokens",
                      "prefill_time_s", "decode_time_s",
                      "resident_hit_rate", "cache_hits", "cache_misses",
                      "prediction_hit_rate", "prediction_hits", "prediction_total",
                      "cold_miss_predict_hit_rate", "cold_prediction_hits", "cold_prediction_total",
                      "on_demand_loads", "on_demand_load_time_s", "sync_ssd_loads", "cluster_loads",
                      "prefetch_wait_time_s", "peak_rss_gb", "spec_acceptance_rate",
                      "paging_resolved", "num_pages", "num_clusters", "storage_format", "eviction_policy",
                      "effective_expert_budget", "ideal_active_experts")
    PROBE_FIDELITY = ("exact_match_rate", "positionwise_agreement", "min_matching_prefix", "reference_tokens")

    def _lazy_probes(self, path: str, limit: int = 20) -> list[dict]:
        """Mid-training lazy-expert probe results (``lazy_probe.jsonl``), newest last.

        Each entry keeps phase / step / wall time and, per variant, the label,
        settings, measured efficiency + hit rates and fidelity versus the
        full-memory reference; skipped requests keep their reason."""
        out = []
        for row in self._stream(os.path.join(path, "lazy_probe.jsonl"), max_rows=500).read()[-limit:]:
            if not isinstance(row, dict):
                continue
            entry = {k: row.get(k) for k in ("time", "phase", "step", "checkpoint", "wall_s", "skipped", "error") if k in row}
            result = row.get("result") or {}
            if result.get("skipped"):
                entry["skipped"] = result["skipped"]
            if result.get("variants") is not None:
                entry.update({k: result.get(k) for k in ("trained_with_clusters", "auto_paging", "num_experts",
                                                          "top_k", "truncated", "variants_requested", "elapsed_s",
                                                          "conditions")})
                refs = result.get("references") or {}
                entry["references"] = {b: (r.get("measured") or {}).get("tokens_per_s") for b, r in refs.items()}
                entry["reference_measured"] = {
                    b: ({"error": r["error"]} if "error" in r else
                        {k: (r.get("measured") or {}).get(k) for k in self.PROBE_MEASURED})
                    for b, r in refs.items()}
                entry["variants"] = [{
                    "label": v.get("label"), "settings": v.get("settings"),
                    **({"error": v["error"]} if "error" in v else {}),
                    "measured": {k: (v.get("measured") or {}).get(k) for k in self.PROBE_MEASURED},
                    "fidelity": {k: (v.get("fidelity") or {}).get(k) for k in self.PROBE_FIDELITY},
                } for v in result["variants"]]
            out.append(entry)
        return out

    def lazy_probes(self, run: str | None, limit: int = 200) -> list[dict]:
        path = self.run_dir(run)
        return self._lazy_probes(path, limit=limit) if path else []

    @staticmethod
    def _checkpoints(path: str) -> list[dict]:
        files = glob.glob(os.path.join(path, "*.pth")) + glob.glob(os.path.join(path, "checkpoints", "*.pkl"))
        out = []
        for file in files:
            try:
                st = os.stat(file)
            except OSError:
                continue
            out.append({"file": os.path.relpath(file, path), "size_gb": round(st.st_size / 1e9, 3),
                        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                        "age_s": round(time.time() - st.st_mtime, 1)})
        out.sort(key=lambda r: r["modified"], reverse=True)
        return out
