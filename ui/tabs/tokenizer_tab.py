"""ui/tabs/tokenizer_tab.py — Chronos Tokenizer Builder.

Trains a tokenizer from user-selected datasets (jsonl / parquet / csv / tsv /
txt, files, directories or globs) with ``chronos.tokenizer.build_tokenizer``
in a background thread, then shows the held-out evaluation it wrote to
``tokenizer_report.json``.  The same builder backs ``chronos tokenizer``.
"""
from __future__ import annotations

from ui.gradio_compat import gr
from ui.i18n import register_translatable, t
from chronos.workbench.tokenizer import (  # noqa: F401  (re-exported)
    EVAL_COLUMNS, TokenizerJob, parse_paths, read_report, report_rows, report_summary,
)

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
