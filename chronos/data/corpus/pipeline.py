"""Parallel corpus build: clean → dedup → leakage filter → split → manifest.

Stage 1 (parallel): every work unit (source × row range) is read, cleaned
and written as a zstd parquet part with 64-bit keys.
Stage 2 (single process, numpy): global exact dedup on the pair key in source
priority order, cross-split / eval leakage removal on either side's key, and
a deterministic validation split by Chinese-side hash.
Outputs are append-only parquet parts plus ``corpus_manifest.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

import numpy as np

from .clean import CleanConfig, PairCleaner, pair_key, side_key
from .readers import count_rows, csv_byte_ranges, iter_csv_byte_range, iter_pairs


@dataclass
class PairSource:
    name: str
    path: str
    kind: str
    fields: tuple[str, str]          # (zh_field, en_field)
    clean: CleanConfig = field(default_factory=CleanConfig)
    priority: int = 0                 # lower wins on duplicates
    weight: float = 1.0               # mixture weight (sampling multiplier)
    rows_per_unit: int = 1_000_000


def h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def _clean_unit(args) -> dict:
    source_dict, start, stop, out_path = args
    import pyarrow as pa
    import pyarrow.parquet as pq
    src = PairSource(**{**source_dict, "clean": CleanConfig(**source_dict["clean"])})
    cleaner = PairCleaner(src.clean)
    zh_out, en_out, pk, zk, ek = [], [], [], [], []
    reasons = Counter()
    if src.kind in ("pairs_csv", "pairs_tsv"):
        stream = iter_csv_byte_range(src.path, src.fields, start, stop,
                                     "\t" if src.kind == "pairs_tsv" else ",")
    else:
        stream = iter_pairs(src.path, src.kind, src.fields, start=start, stop=stop)
    for zh, en in stream:
        result = cleaner.clean(zh, en)
        if isinstance(result, str):
            reasons[result] += 1
            continue
        zh, en = result
        zh_out.append(zh)
        en_out.append(en)
        pk.append(h64(pair_key(zh, en)))
        zk.append(h64(side_key(zh)))
        ek.append(h64(side_key(en)))
    reasons["kept"] = len(zh_out)
    table = pa.table({"zh": zh_out, "en": en_out,
                      "pair_key": pa.array(pk, pa.int64()), "zh_key": pa.array(zk, pa.int64()),
                      "en_key": pa.array(ek, pa.int64())})
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    info = {"source": src.name, "start": start, "stop": stop, "path": out_path, "reasons": dict(reasons)}
    with open(out_path + ".json", "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    os.replace(tmp, out_path)  # the part exists only once complete (resume-safe)
    return info


def _units(sources: list[PairSource], parts_dir: str):
    units = []
    for src in sources:
        if src.kind in ("pairs_csv", "pairs_tsv"):
            # ~150 bytes per pair row: byte ranges avoid rescanning from row 0.
            ranges = csv_byte_ranges(src.path, src.rows_per_unit * 150)
        else:
            n = count_rows(src.path, src.kind)
            ranges = [(s, min(n, s + src.rows_per_unit)) for s in range(0, n, src.rows_per_unit)]
        for i, (start, stop) in enumerate(ranges):
            out = os.path.join(parts_dir, f"{src.name}.{i:05d}.parquet")
            units.append(((asdict(src) | {"clean": asdict(src.clean)}), start, stop, out))
    return units


def build_corpus(sources: list[PairSource], eval_pairs: dict[str, list[tuple[str, str]]],
                 out_dir: str, *, workers: int = 16, valid_per_mille: int = 2,
                 min_free_gb: float = 8.0, keep_parts: bool = False, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    parts_dir = os.path.join(out_dir, "clean_parts")
    os.makedirs(parts_dir, exist_ok=True)
    if shutil.disk_usage(out_dir).free < min_free_gb * (1 << 30):
        raise OSError(f"less than {min_free_gb} GiB free at {out_dir}")
    t0 = time.time()
    units = [u for u in _units(sources, parts_dir) if not os.path.exists(u[3])]
    done_units = []
    log(f"[corpus] stage1: {len(units)} pending units, {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_clean_unit, u) for u in units]
        for i, fut in enumerate(as_completed(futures), 1):
            info = fut.result()
            done_units.append(info)
            log(f"[corpus] {i}/{len(units)} {info['source']}[{info['start']}:{info['stop']}] "
                f"kept={info['reasons'].get('kept', 0)} ({time.time() - t0:.0f}s)")
    manifest = _finalize(sources, eval_pairs, out_dir, parts_dir, done_units, valid_per_mille, t0, log)
    if not keep_parts:
        # The final split files hold every kept row; drop the intermediate
        # text but keep the per-unit reason sidecars for auditing.
        for name in os.listdir(parts_dir):
            if name.endswith(".parquet"):
                os.remove(os.path.join(parts_dir, name))
    return manifest


def _split_of(zh_key: np.ndarray, valid_per_mille: int) -> np.ndarray:
    """0 = train, 1 = valid; deterministic in the Chinese-side hash."""
    mixed = (zh_key.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return (mixed % np.uint64(1000) < np.uint64(valid_per_mille)).astype(np.int8)


def _finalize(sources, eval_pairs, out_dir, parts_dir, done_units, valid_per_mille, t0, log):
    import pyarrow as pa
    import pyarrow.parquet as pq
    by_source = {s.name: s for s in sources}
    parts = sorted((p for p in os.listdir(parts_dir) if p.endswith(".parquet")),
                   key=lambda p: (by_source[p.split(".")[0]].priority, p))
    # ---- gather keys (≈ 24 bytes/row) ----
    pair_keys, zh_keys, en_keys, part_id, row_id = [], [], [], [], []
    for pi, name in enumerate(parts):
        t = pq.read_table(os.path.join(parts_dir, name), columns=["pair_key", "zh_key", "en_key"])
        n = t.num_rows
        pair_keys.append(t["pair_key"].to_numpy()); zh_keys.append(t["zh_key"].to_numpy())
        en_keys.append(t["en_key"].to_numpy())
        part_id.append(np.full(n, pi, np.int32)); row_id.append(np.arange(n, dtype=np.int64))
    pair_keys = np.concatenate(pair_keys); zh_keys = np.concatenate(zh_keys); en_keys = np.concatenate(en_keys)
    part_id = np.concatenate(part_id); row_id = np.concatenate(row_id)
    total = len(pair_keys)
    # ---- exact dedup: first occurrence in priority order wins ----
    _, first = np.unique(pair_keys, return_index=True)
    keep = np.zeros(total, bool); keep[first] = True
    dup = int(total - keep.sum())
    # ---- evaluation leakage on either side ----
    eval_zh = np.array([h64(side_key(zh)) for pairs in eval_pairs.values() for _en, zh in pairs], np.int64)
    eval_en = np.array([h64(side_key(en)) for pairs in eval_pairs.values() for en, _zh in pairs], np.int64)
    leak = np.isin(zh_keys, eval_zh) | np.isin(en_keys, eval_en)
    leaked = int((leak & keep).sum())
    keep &= ~leak
    # ---- split by zh hash; drop en-side keys shared across splits ----
    split = _split_of(zh_keys, valid_per_mille)
    valid_en = np.unique(en_keys[keep & (split == 1)])
    conflict = keep & (split == 0) & np.isin(en_keys, valid_en)
    keep &= ~conflict
    log(f"[corpus] rows={total} duplicates={dup} eval_leak={leaked} split_conflicts={int(conflict.sum())}")
    # ---- write final splits, part by part (bounded memory) ----
    final_dir = os.path.join(out_dir, "pairs")
    os.makedirs(final_dir, exist_ok=True)
    counts = Counter()
    for pi, name in enumerate(parts):
        mask = part_id == pi
        rows, k, sp = row_id[mask], keep[mask], split[mask]
        if not k.any():
            continue
        t = pq.read_table(os.path.join(parts_dir, name), columns=["zh", "en", "pair_key"])
        src = name.split(".")[0]
        for split_name, code in (("train", 0), ("valid", 1)):
            sel = rows[k & (sp == code)]
            if len(sel) == 0:
                continue
            sub = t.take(pa.array(sel)).append_column("source", pa.array([src] * len(sel)))
            pq.write_table(sub, os.path.join(final_dir, f"{split_name}.{name}"), compression="zstd")
            counts[(split_name, src)] += len(sel)
    reasons = Counter()
    for name in parts:
        meta_path = os.path.join(parts_dir, name + ".json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                info = json.load(fh)
            reasons.update({f"{info['source']}:{k}": v for k, v in info["reasons"].items()})
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "sources": [asdict(s) | {"clean": asdict(s.clean)} for s in sources],
        "eval_sets": {k: len(v) for k, v in eval_pairs.items()},
        "rows_after_clean": total, "duplicates": dup, "eval_leak_removed": leaked,
        "split_conflicts_removed": int(conflict.sum()),
        "final": {f"{s}:{src}": n for (s, src), n in sorted(counts.items())},
        "final_train": sum(n for (s, _), n in counts.items() if s == "train"),
        "final_valid": sum(n for (s, _), n in counts.items() if s == "valid"),
        "clean_reasons": dict(sorted(reasons.items())),
        "valid_per_mille": valid_per_mille,
        "split_policy": "zh-side hash; en-side keys shared with valid removed from train; "
                        "eval sets removed on either side",
    }
    with open(os.path.join(out_dir, "corpus_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest
