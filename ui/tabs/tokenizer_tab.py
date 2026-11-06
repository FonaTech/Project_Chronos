"""ui/tabs/tokenizer_tab.py — Chronos Tokenizer Builder.

Trains a tokenizer from user-selected datasets (jsonl / parquet / csv / tsv /
txt, files, directories or globs) with ``chronos.tokenizer.build_tokenizer``
in a background thread, then shows the held-out evaluation it wrote to
``tokenizer_report.json``.  The same builder backs ``chronos tokenizer``.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

from ui.gradio_compat import gr
from ui.i18n import register_translatable, t

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


_job = TokenizerJob()


def _status_text(state: str, lang: str | None = None) -> str:
    return t(f"tok.state.{state}", lang)


def build_tokenizer_tab():
    with gr.Tab(t("tab.tokenizer")) as tab:
        register_translatable(tab, "tab.tokenizer")
        intro = gr.Markdown(t("tok.intro"))
        register_translatable(intro, "tok.intro")

        with gr.Row():
            paths = gr.Textbox(label=t("tok.paths"), lines=4, scale=3,
                               placeholder="./dataset/corpus/\n./dataset/*.jsonl\n./dataset/*.parquet")
            with gr.Column(scale=2):
                out_dir = gr.Textbox(label=t("tok.out_dir"), value="./out/tokenizer")
                fields = gr.Textbox(label=t("tok.fields"), placeholder="text, messages, prompt, response")
        with gr.Row():
            vocab = gr.Number(label=t("tok.vocab_size"), value=16384, precision=0)
            per_source = gr.Number(label=t("tok.per_source"), value=400_000, precision=0)
            temperature = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label=t("tok.temperature"))
        with gr.Row():
            build_btn = gr.Button(t("tok.build"), variant="primary")
            load_btn = gr.Button(t("tok.load_report"))
        status = gr.Textbox(label=t("tok.status"), interactive=False)
        log_box = gr.Textbox(label=t("tok.log"), lines=12, interactive=False, autoscroll=True)
        summary = gr.JSON(label=t("tok.summary"), value={})
        table = gr.Dataframe(headers=EVAL_COLUMNS, label=t("tok.eval"), interactive=False)
        for comp, key in ((paths, "tok.paths"), (out_dir, "tok.out_dir"), (fields, "tok.fields"),
                          (vocab, "tok.vocab_size"), (per_source, "tok.per_source"),
                          (temperature, "tok.temperature"), (build_btn, "tok.build"),
                          (load_btn, "tok.load_report"), (status, "tok.status"), (log_box, "tok.log"),
                          (summary, "tok.summary"), (table, "tok.eval")):
            register_translatable(comp, key)

        def start(p, o, v, n, f, temp):
            field_list = [x.strip() for x in (f or "").split(",") if x.strip()]
            outcome = _job.start(parse_paths(p), (o or "").strip(), int(v or 16384), int(n or 400_000),
                                 field_list, float(temp or 0.5))
            if outcome != "started":
                return t(f"tok.err.{outcome}")
            return _status_text("running")

        def show(o):
            report = read_report(o)
            if not report:
                return t("tok.err.no_report"), {}, []
            return t("tok.state.loaded"), report_summary(report), report_rows(report.get("heldout_eval"))

        def poll(o):
            state, text = _job.snapshot()
            if state in ("done",) and _job.out_dir:
                report = read_report(_job.out_dir)
                return (_status_text(state), text, report_summary(report),
                        report_rows(report.get("heldout_eval")))
            message = _status_text(state) + (f" — {_job.error}" if _job.error else "")
            return message, text, gr.update(), gr.update()

        build_btn.click(fn=start, inputs=[paths, out_dir, vocab, per_source, fields, temperature],
                        outputs=[status])
        load_btn.click(fn=show, inputs=[out_dir], outputs=[status, summary, table])
        timer = gr.Timer(2.0)
        timer.tick(fn=poll, inputs=[out_dir], outputs=[status, log_box, summary, table])
    return tab
