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

import hashlib
import json
import os
import statistics
from collections import Counter

from ui.gradio_compat import gr

import chronos.deps  # noqa: F401
from ui.i18n import t, register_translatable

_CHAT_KEYS = ("conversations", "messages", "dialog", "dialogue")
_PROMPT_KEYS = ("prompt", "question", "query", "input", "instruction")
_ANSWER_KEYS = ("response", "answer", "completion", "output", "target", "chosen")


def record_schema(rec) -> str:
    """Coarse shape of one record; never task-specific beyond the field names."""
    if isinstance(rec, str):
        return "plain_text"
    if not isinstance(rec, dict):
        return "other"
    if any(isinstance(rec.get(k), list) for k in _CHAT_KEYS):
        return "chat"
    if "chosen" in rec and "rejected" in rec:
        return "preference"
    if "instruction" in rec and any(k in rec for k in ("output", "response", "answer")):
        return "instruction"
    if any(k in rec for k in _PROMPT_KEYS) and any(k in rec for k in _ANSWER_KEYS):
        return "prompt_response"
    nested = [v for v in rec.values() if isinstance(v, dict)
              and len(v) >= 2 and all(isinstance(x, str) for x in v.values())]
    short = [k for k, v in rec.items() if isinstance(v, str) and len(k) <= 3]
    if nested or len(short) >= 2:
        return "parallel_pair"
    if any(isinstance(rec.get(k), str) for k in ("text", "content", "document", "passage", "article", "story")):
        return "plain_text"
    return "other"


def _extract_text(rec) -> str:
    """All text a record carries, in field order (chat turns, pairs, any schema)."""
    from chronos.tokenizer.extract import _texts_from_record
    parts = list(_texts_from_record(rec, None))
    if not parts and isinstance(rec, dict):
        parts = [str(v) for v in rec.values() if isinstance(v, str)]
    return " ".join(parts)


def _fields(rec, prefix: str = "", out: dict | None = None, depth: int = 0) -> dict:
    out = {} if out is None else out
    if not isinstance(rec, dict) or depth > 2:
        return out
    for k, v in rec.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _fields(v, key, out, depth + 1)
        else:
            out[key] = type(v).__name__ if v is not None else "null"
    return out


def _scan(path: str, max_records: int = 50_000):
    """Stream up to ``max_records`` records; return summary dict + length list + previews."""
    from chronos.tokenizer.extract import expand, iter_records, script_bucket
    if not path or not path.strip():
        return {"error": "no path given"}, [], []
    path = os.path.expanduser(path.strip())
    files = [path] if os.path.isfile(path) else expand([path])
    if not files:
        return {"error": f"no readable data files at: {path!r} (jsonl, json, parquet, csv, tsv, txt)"}, [], []
    char_lens, samples = [], []
    schemas, scripts, field_hits, field_types = Counter(), Counter(), Counter(), {}
    examples: dict[str, str] = {}
    seen, dup, n_total, errors = set(), 0, 0, []
    modal = Counter()
    try:
        from chronos.data.flexible_dataset import detect_modalities
    except Exception:  # pragma: no cover - torch-free environments
        detect_modalities = None
    for file in files:
        if n_total >= max_records:
            break
        try:
            for rec in iter_records(file):
                if n_total >= max_records:
                    break
                n_total += 1
                kind = record_schema(rec)
                schemas[kind] += 1
                for key, typ in _fields(rec).items():
                    field_hits[key] += 1
                    field_types.setdefault(key, typ)
                if detect_modalities is not None and isinstance(rec, dict):
                    for mod in detect_modalities(rec):
                        modal[mod] += 1
                text = _extract_text(rec)
                char_lens.append(len(text))
                if text:
                    scripts[script_bucket(text[:400])] += 1
                digest = hashlib.md5(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str)
                                     .encode("utf-8")).hexdigest()
                if digest in seen:
                    dup += 1
                else:
                    seen.add(digest)
                if kind not in examples:
                    examples[kind] = json.dumps(rec, ensure_ascii=False, default=str)[:300]
                if len(samples) < 3:
                    samples.append(text[:200])
        except Exception as exc:  # unreadable file: report it, keep scanning the rest
            errors.append(f"{os.path.basename(file)}: {type(exc).__name__}: {exc}"[:200])
    if not char_lens:
        return {"error": "no parseable records", "files": len(files), "read_errors": errors}, [], samples
    ordered = sorted(char_lens)
    summary = {
        "files":         len(files),
        "scanned":       n_total,
        "dup_ratio":     round(dup / max(n_total, 1), 4),
        "schemas":       dict(schemas.most_common()),
        "schema_examples": examples,
        "fields":        {k: {"type": field_types[k], "present": round(c / n_total, 3)}
                          for k, c in field_hits.most_common(24)},
        "scripts":       {k: round(c / max(1, sum(scripts.values())), 3) for k, c in scripts.most_common()},
        "char_min":      ordered[0],
        "char_mean":     round(statistics.mean(char_lens), 1),
        "char_median":   int(statistics.median(char_lens)),
        "char_p95":      int(ordered[int(0.95 * (len(ordered) - 1))]),
        "char_max":      ordered[-1],
        "size_mb":       round(sum(os.path.getsize(f) for f in files) / (1024 * 1024), 1),
        "note":          (f"sampled first {max_records} records" if n_total >= max_records else "full scan"),
    }
    if modal:
        summary["modalities"] = dict(modal)
    if errors:
        summary["read_errors"] = errors[:8]
    return summary, char_lens, samples


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
