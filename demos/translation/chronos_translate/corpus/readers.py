"""Streaming readers that yield (source_text, target_text) pairs or plain text.

Every reader is lazy and bounded in memory, so multi-GB CSV/parquet sources
never load fully.  Field specs use dotted paths (``translation.zh``).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Iterator

csv.field_size_limit(min(sys.maxsize, 1 << 30))


def _get(record, dotted: str):
    value = record
    for part in str(dotted).split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, (list, tuple)) and part.isdigit():
            value = value[int(part)] if int(part) < len(value) else None
        else:
            return None
    return value


def iter_pairs(path: str, kind: str, fields: tuple[str, str], *, batch_rows: int = 65536,
               start: int = 0, stop: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield (a, b) pairs for ``fields=(a_field, b_field)``.

    kind: pairs_parquet | pairs_csv | pairs_jsonl | pairs_tsv.
    ``start``/``stop`` select a row range (for parallel shards).
    """
    a_field, b_field = fields
    row = 0
    if kind == "pairs_parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        top = sorted({a_field.split(".")[0], b_field.split(".")[0]})
        for batch in pf.iter_batches(batch_size=batch_rows, columns=top):
            n = batch.num_rows
            if stop is not None and row >= stop:
                return
            if row + n <= start:
                row += n
                continue
            for record in batch.to_pylist():
                if start <= row and (stop is None or row < stop):
                    a, b = _get(record, a_field), _get(record, b_field)
                    if isinstance(a, str) and isinstance(b, str):
                        yield a, b
                row += 1
        return
    if kind in ("pairs_csv", "pairs_tsv"):
        delimiter = "\t" if kind == "pairs_tsv" else ","
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            header = next(reader, None) or []
            ia = header.index(a_field) if a_field in header else int(a_field)
            ib = header.index(b_field) if b_field in header else int(b_field)
            for record in reader:
                if stop is not None and row >= stop:
                    return
                if row >= start and len(record) > max(ia, ib):
                    yield record[ia], record[ib]
                row += 1
        return
    if kind == "pairs_jsonl":
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if stop is not None and row >= stop:
                    return
                if row >= start and line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        row += 1
                        continue
                    a, b = _get(record, a_field), _get(record, b_field)
                    if isinstance(a, str) and isinstance(b, str):
                        yield a, b
                row += 1
        return
    raise ValueError(f"unknown pair source kind {kind!r}")


def iter_csv_byte_range(path: str, fields: tuple[str, str], byte_start: int, byte_stop: int,
                        delimiter: str = ",") -> Iterator[tuple[str, str]]:
    """Pairs whose line starts in [byte_start, byte_stop).  Ranges must come
    from ``csv_byte_ranges`` (newline aligned); the header line is skipped."""
    with open(path, "rb") as raw:
        header = raw.readline().decode("utf-8", errors="replace")
        names = next(csv.reader([header], delimiter=delimiter))
        a_field, b_field = fields
        ia = names.index(a_field) if a_field in names else int(a_field)
        ib = names.index(b_field) if b_field in names else int(b_field)
        raw.seek(max(byte_start, raw.tell()))
        def lines():
            while raw.tell() < byte_stop:
                line = raw.readline()
                if not line:
                    return
                yield line.decode("utf-8", errors="replace")
        for record in csv.reader(lines(), delimiter=delimiter):
            if len(record) > max(ia, ib):
                yield record[ia], record[ib]


def csv_byte_ranges(path: str, target_bytes: int) -> list[tuple[int, int]]:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.readline()
        cuts = [fh.tell()]
        while cuts[-1] + target_bytes < size:
            fh.seek(cuts[-1] + target_bytes)
            fh.readline()  # align to the next line start
            if fh.tell() >= size:
                break
            cuts.append(fh.tell())
    cuts.append(size)
    return list(zip(cuts, cuts[1:]))


def count_rows(path: str, kind: str) -> int:
    if kind == "pairs_parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    with open(path, "rb") as fh:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 24), b""))
    return max(0, lines - (1 if kind in ("pairs_csv", "pairs_tsv") else 0))


def expand_paths(path: str, suffix: str) -> list[str]:
    if os.path.isdir(path):
        out = []
        for root, _dirs, files in os.walk(path):
            out.extend(os.path.join(root, f) for f in files if f.endswith(suffix))
        return sorted(out)
    return [path]
