"""
ui/tabs/data_tab.py — Dataset inspector for any task.

Streams a file, a directory or a glob (jsonl / json / parquet / csv / tsv /
txt; never a full load) and reports, without assuming what the data is for:
  - record count and the record schemas found (chat, instruction, prompt /
    response, preference, parallel pair, plain text, other), with examples
  - the field inventory (dotted paths, types, how often each is present)
  - char-length distribution (min / mean / median / p95 / max)
  - script mix of the text (Han, Latin, Kana, Hangul, Cyrillic …)
  - unique-record ratio (cheap dedup heuristic via record hash)
  - media modalities (image / audio) and the first few previews
  - a histogram of char lengths

The point: before training, the user can verify the corpus is what they
expect — right fields, sane lengths, low duplication — whatever the model
is meant to do.
"""
from __future__ import annotations

from ui.gradio_compat import gr

import chronos.deps  # noqa: F401
from ui.i18n import t, register_translatable
from chronos.workbench.data import (  # noqa: F401  (re-exported)
    _ANSWER_KEYS, _CHAT_KEYS, _PROMPT_KEYS, _extract_text, _fields, _scan, record_schema,
)

def _hist(lens):
    if not lens:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.hist(lens, bins=40, color="#39d98a", edgecolor="#1a1a2e")
    ax.set_xlabel("chars per record (all text fields)", color="#aaa")
    ax.set_ylabel("frequency", color="#aaa")
    ax.tick_params(colors="#aaa")
    for s in ax.spines.values():
        s.set_edgecolor("#333")
    return fig


def build_data_tab():
    with gr.Tab(t("tab.data")) as tab:
        register_translatable(tab, "tab.data")
        gr.Markdown(f"### {t('data.title')}")

        with gr.Row():
            data_path = gr.Textbox(
                label=t("data.path"), scale=4,
                placeholder="./dataset/train.jsonl  |  ./data/  |  ./data/*.parquet",
            )
            scan_btn = gr.Button(t("data.scan"), variant="primary", scale=1)
            register_translatable(data_path, "data.path")
            register_translatable(scan_btn, "data.scan")

        summary_box = gr.JSON(label=t("data.summary"), value={})
        register_translatable(summary_box, "data.summary")

        preview_box = gr.Textbox(
            label=t("data.preview"), lines=6, interactive=False, value="",
        )
        register_translatable(preview_box, "data.preview")

        hist_plot = gr.Plot(label=t("data.length_hist"))
        register_translatable(hist_plot, "data.length_hist")

        def do_scan(path):
            summary, lens, samples = _scan(path)
            preview = "\n---\n".join(samples) if samples else ""
            fig = _hist(lens)
            return summary, preview, fig

        scan_btn.click(fn=do_scan, inputs=[data_path],
                       outputs=[summary_box, preview_box, hist_plot])

    return tab
