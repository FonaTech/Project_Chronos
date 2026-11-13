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


# ---- document-level sets (doc-mix runs) ------------------------------------
def _flores_docs(raw: str, split: str):
    """FLORES+ sentences grouped by source article (URL), in id order."""
    import pyarrow.parquet as pq
    path = os.path.join(raw, "facebook__flores__eng-zho", "data", "pair", "eng_Latn-zho_Hans",
                        f"{split}-00000-of-00001.parquet")
    rows = pq.read_table(path, columns=["id", "URL", "sentence_eng_Latn", "sentence_zho_Hans"]).to_pylist()
    docs: dict = {}
    for r in sorted(rows, key=lambda r: int(r["id"])):
        docs.setdefault(r["URL"], []).append((r["sentence_eng_Latn"], r["sentence_zho_Hans"]))
    return [{"doc_id": url, "segments": segs} for url, segs in docs.items()]


def _wmt24pp_docs(raw: str):
    """WMT24++ en→zh_CN segments grouped by document_id, in segment order (bad sources dropped)."""
    path = os.path.join(raw, "google__wmt24pp__en-zh_CN", "en-zh_CN.jsonl")
    docs: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if str(r.get("is_bad_source", "")).strip().lower() == "true":
                continue
            docs.setdefault(r["document_id"], []).append((int(r["segment_id"]), r["source"], r["target"],
                                                          r.get("domain")))
    out = []
    for doc_id, segs in docs.items():
        segs.sort()
        out.append({"doc_id": doc_id, "domain": segs[0][3], "segments": [(s, t) for _i, s, t, _d in segs]})
    return out


EVAL_DOCUMENTS = {
    "flores_dev_docs": lambda raw: _flores_docs(raw, "dev"),
    "flores_devtest_docs": lambda raw: _flores_docs(raw, "devtest"),
    "wmt24pp_docs": _wmt24pp_docs,
}


def load_eval_documents(raw_root: str, names=None) -> dict[str, list[dict]]:
    """Return {name: [{"doc_id", "segments": [(en, zh), ...]}, ...]} for every available set.

    A document's source is its English segments joined by " ", its reference
    the Chinese segments joined without separator (same joins as doc-mix).
    """
    out = {}
    for name, loader in EVAL_DOCUMENTS.items():
        if names and name not in names:
            continue
        try:
            out[name] = loader(raw_root)
        except FileNotFoundError:
            continue
    return out
