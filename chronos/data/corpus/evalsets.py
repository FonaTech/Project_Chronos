"""Held-out evaluation sets (never trained on; also used for leakage filtering)."""
from __future__ import annotations

import ast
import json
import os


def _flores(raw: str, split: str):
    import pyarrow.parquet as pq
    path = os.path.join(raw, "facebook__flores__eng-zho", "data", "pair", "eng_Latn-zho_Hans",
                        f"{split}-00000-of-00001.parquet")
    for r in pq.read_table(path, columns=["sentence_eng_Latn", "sentence_zho_Hans"]).to_pylist():
        yield r["sentence_eng_Latn"], r["sentence_zho_Hans"]


def _wmt22(raw: str):
    import pyarrow.parquet as pq
    folder = os.path.join(raw, "haoranxu__WMT22-Test__en-zh", "en-zh")
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".parquet"):
            continue
        for r in pq.read_table(os.path.join(folder, name)).to_pylist():
            value = r["en-zh"]
            pair = value if isinstance(value, dict) else ast.literal_eval(value)
            yield pair["en"], pair["zh"]


def _wmt24pp(raw: str):
    path = os.path.join(raw, "google__wmt24pp__en-zh_CN", "en-zh_CN.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if str(r.get("is_bad_source", "")).strip().lower() == "true":
                continue  # documented low-quality sources
            yield r["source"], r["target"]


def _opus_test(raw: str):
    import pyarrow.parquet as pq
    path = os.path.join(raw, "Helsinki-NLP__opus-100", "en-zh", "test-00000-of-00001.parquet")
    for r in pq.read_table(path, columns=["translation"]).to_pylist():
        yield r["translation"]["en"], r["translation"]["zh"]


EVAL_SETS = {
    "flores_dev": lambda raw: _flores(raw, "dev"),
    "flores_devtest": lambda raw: _flores(raw, "devtest"),
    "wmt22_enzh": _wmt22,
    "wmt24pp_enzh": _wmt24pp,
    "opus100_test": _opus_test,
}


def load_eval_sets(raw_root: str, names=None) -> dict[str, list[tuple[str, str]]]:
    """Return {name: [(en, zh), ...]} for every available set."""
    out = {}
    for name, loader in EVAL_SETS.items():
        if names and name not in names:
            continue
        try:
            out[name] = list(loader(raw_root))
        except FileNotFoundError:
            continue
    return out
