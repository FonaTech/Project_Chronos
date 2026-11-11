"""Automatic text extraction for tokenizer training from user-selected datasets.

Accepts any mix of files/directories (parquet, jsonl, json, csv, tsv, txt).
For each record it finds text fields automatically (conversations, pairs,
``text``/``content``, nested string fields …) unless a field list is given, buckets
text by script, and keeps a seeded per-source reservoir so every source is
represented regardless of size.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Iterator

_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

TEXT_KEYS = ("text", "content", "sentence", "source", "target", "prompt", "response",
             "chosen", "rejected", "question", "answer", "instruction", "input", "output")
SUFFIXES = (".parquet", ".jsonl", ".json", ".csv", ".tsv", ".txt")


def script_bucket(text: str) -> str:
    counts = {"zh": len(_HAN.findall(text)), "ja": len(_KANA.findall(text)) * 2,
              "ko": len(_HANGUL.findall(text)), "latin": len(_LATIN.findall(text)),
              "cyrillic": len(_CYRILLIC.findall(text))}
    best = max(counts, key=counts.get)
    return best if counts[best] else "other"


def _texts_from_record(record, fields: list[str] | None) -> Iterator[str]:
    if fields:
        for f in fields:
            value = record
            for part in f.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if isinstance(value, str):
                yield value
        return
    if isinstance(record, str):
        yield record
        return
    if not isinstance(record, dict):
        return
    for key in ("conversations", "messages"):
        msgs = record.get(key)
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    yield m["content"]
            return
    for key, value in record.items():
        if isinstance(value, str) and (key in TEXT_KEYS or len(key) <= 3 or key.startswith("sentence")):
            yield value
        elif isinstance(value, dict):  # nested string fields, e.g. {"a": ..., "b": ...}
            for sub in value.values():
                if isinstance(sub, str):
                    yield sub


def iter_records(path: str) -> Iterator[object]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            yield from batch.to_pylist()
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif path.endswith(".json"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        yield from (data if isinstance(data, list) else [data])
    elif path.endswith((".csv", ".tsv")):
        csv.field_size_limit(1 << 30)
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh, delimiter="\t" if path.endswith(".tsv") else ","):
                yield {k if k and not k.isdigit() else f"c{k}": v for k, v in row.items()}
    elif path.endswith(".txt"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    yield line.rstrip("\n")


def expand(paths: list[str]) -> list[str]:
    import glob
    out = []
    for p in paths:
        if any(ch in p for ch in "*?["):
            out.extend(f for f in glob.glob(p) if f.endswith(SUFFIXES))
        elif os.path.isdir(p):
            for root, _d, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in sorted(files) if f.endswith(SUFFIXES))
        elif p.endswith(SUFFIXES):
            out.append(p)
    return sorted(out)


@dataclass
class ExtractConfig:
    per_source_texts: int = 400_000       # reservoir size per selected path
    max_records_per_source: int | None = 4_000_000
    fields: list[str] | None = None       # explicit dotted field list, else auto
    min_chars: int = 2
    max_chars: int = 2000
    seed: int = 1234
    exclude_substrings: tuple[str, ...] = ("/valid.", "/test")


@dataclass
class ExtractResult:
    texts_by_bucket: dict[str, list[str]]
    per_source: dict[str, dict] = field(default_factory=dict)


def _capped(iterator, limit):
    for i, item in enumerate(iterator):
        if limit is not None and i >= limit:
            return
        yield item


def extract_texts(paths: list[str], config: ExtractConfig, normalize=None, log=None) -> ExtractResult:
    """Seeded reservoir sample of text per source, bucketed by script."""
    rng = random.Random(config.seed)
    buckets: dict[str, list[str]] = {}
    per_source = {}
    # Each argument is one user-selected dataset (file, directory or glob):
    # one reservoir per dataset, so a corpus split into many shards is not
    # over-weighted.  Records are capped per file so every shard is sampled.
    for source in paths:
        files = [f for f in expand([source]) if not any(x in f for x in config.exclude_substrings)]
        if not files:
            continue
        per_file = None
        if config.max_records_per_source:
            per_file = max(1, config.max_records_per_source // len(files))
        reservoir: list[str] = []
        seen = records = 0
        for record in (r for f in files for r in _capped(iter_records(f), per_file)):
            records += 1
            for text in _texts_from_record(record, config.fields):
                if normalize is not None:
                    text = normalize(text)
                if not (config.min_chars <= len(text) <= config.max_chars):
                    continue
                seen += 1
                if len(reservoir) < config.per_source_texts:
                    reservoir.append(text)
                else:
                    j = rng.randrange(seen)
                    if j < config.per_source_texts:
                        reservoir[j] = text
        counts = {}
        for text in reservoir:
            b = script_bucket(text)
            buckets.setdefault(b, []).append(text)
            counts[b] = counts.get(b, 0) + 1
        per_source[source] = {"files": len(files), "records": records, "texts_seen": seen,
                              "sampled": len(reservoir), "buckets": counts}
        if log:
            log(f"[tokenizer] {source}: files={len(files)} records={records} sampled={len(reservoir)} {counts}")
    return ExtractResult(texts_by_bucket=buckets, per_source=per_source)
