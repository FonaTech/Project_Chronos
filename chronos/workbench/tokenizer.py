"""Chronos Tokenizer Builder jobs and report helpers (no UI dependency)."""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

EVAL_COLUMNS = ["bucket", "texts", "bytes/token", "chars/token", "tokens/word",
                "byte_fallback_rate", "roundtrip_failures"]


def parse_paths(text: str) -> list[str]:
    """One path per line or comma; ``~`` expanded; blanks dropped."""
    out = []
    for chunk in (text or "").replace(",", "\n").splitlines():
        chunk = chunk.strip().strip('"').strip("'")
        if chunk:
            out.append(os.path.expanduser(chunk))
    return out


def report_rows(report: dict) -> list[list]:
    """Per-bucket rows from a ``heldout_eval`` dict (``vocab_utilization`` skipped)."""
    rows = []
    for bucket, stats in (report or {}).items():
        if not isinstance(stats, dict):
            continue
        rows.append([bucket, stats.get("texts"), stats.get("bytes_per_token"), stats.get("chars_per_token"),
                     stats.get("tokens_per_word"), stats.get("byte_fallback_rate"),
                     stats.get("roundtrip_failures")])
    return rows


def read_report(out_dir: str) -> dict:
    path = os.path.join(os.path.expanduser(out_dir or ""), "tokenizer_report.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def report_summary(report: dict) -> dict:
    if not report:
        return {}
    keep = ("vocab_size", "fingerprint", "algorithm", "elapsed_s", "special_tokens")
    out = {k: report[k] for k in keep if k in report}
    heldout = report.get("heldout_eval") or {}
    if "vocab_utilization" in heldout:
        out["vocab_utilization"] = heldout["vocab_utilization"]
    if report.get("sources"):
        out["sources"] = report["sources"]
    return out


class TokenizerJob:
    """One background build at a time; log lines are kept in a bounded deque."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.lines: deque[str] = deque(maxlen=400)
        self.state = "idle"
        self.out_dir = ""
        self.error = ""

    def log(self, message) -> None:
        with self.lock:
            self.lines.append(f"{time.strftime('%H:%M:%S')} {message}")

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, paths: list[str], out_dir: str, vocab_size: int, per_source: int,
              fields: list[str] | None, temperature: float) -> str:
        if self.running():
            return "busy"
        if not paths:
            return "no_paths"
        if not out_dir:
            return "no_out"
        from chronos.tokenizer.extract import expand
        files = expand(paths)
        if not files:
            return "no_files"
        self.lines.clear()
        self.state, self.out_dir, self.error = "running", os.path.expanduser(out_dir), ""
        self.log(f"{len(files)} files: " + ", ".join(os.path.basename(f) for f in files[:8])
                 + (" …" if len(files) > 8 else ""))

        def work():
            from chronos.tokenizer import BPETrainerConfig, ExtractConfig, build_tokenizer
            try:
                build_tokenizer(
                    paths, self.out_dir, vocab_size=int(vocab_size),
                    extract=ExtractConfig(per_source_texts=int(per_source), fields=fields or None),
                    trainer=BPETrainerConfig(vocab_size=int(vocab_size),
                                             language_temperature=float(temperature)),
                    log=self.log,
                )
                self.state = "done"
            except Exception as exc:          # surfaced in the UI, never raised into Gradio
                self.error = f"{type(exc).__name__}: {exc}"
                self.log(f"error: {self.error}")
                self.state = "failed"

        self.thread = threading.Thread(target=work, name="chronos-tokenizer", daemon=True)
        self.thread.start()
        return "started"

    def snapshot(self) -> tuple[str, str]:
        with self.lock:
            return self.state, "\n".join(self.lines)
