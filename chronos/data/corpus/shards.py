"""Tokenized, memory-mapped training shards for pretraining and SFT.

Pretrain stream: one flat uint16/uint32 token array per shard (``.bin``),
documents separated by ``<|endoftext|>``; the loader packs fixed windows.
Document mix (per pair, seeded):
  * monolingual zh, monolingual en       — language modelling on both sides
  * ``<|en|>{en}<|zh|>{zh}`` and reverse  — cross-lingual alignment
SFT stream: chat-template records ``[ids, loss_mask]`` (only the assistant
reply is supervised) in a ``.bin`` + ``.idx`` pair for O(1) random access.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SFT_PROMPT_EN_ZH = "翻译成中文，只输出译文：\n"


def _dtype_for(vocab_size: int):
    return np.uint16 if vocab_size <= 65535 else np.uint32


def _encode_batch(tok, texts: list[str]) -> list[list[int]]:
    return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def _pretrain_part(args) -> dict:
    tokenizer_dir, part_path, out_path, seed, mix = args
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    eot, en_tag, zh_tag = (tok.token_to_id(t) for t in ("<|endoftext|>", "<|en|>", "<|zh|>"))
    table = pq.read_table(part_path, columns=["zh", "en"])
    zh, en = table["zh"].to_pylist(), table["en"].to_pylist()
    rng = random.Random(seed)
    zh_ids, en_ids = _encode_batch(tok, zh), _encode_batch(tok, en)
    out, docs = [], 0
    for z, e in zip(zh_ids, en_ids):
        r = rng.random()
        if r < mix["mono"]:
            out.extend(z); out.append(eot); out.extend(e); out.append(eot); docs += 2
        elif r < mix["mono"] + mix["en_zh"]:
            out.append(en_tag); out.extend(e); out.append(zh_tag); out.extend(z); out.append(eot); docs += 1
        else:
            out.append(zh_tag); out.extend(z); out.append(en_tag); out.extend(e); out.append(eot); docs += 1
    arr = np.asarray(out, dtype=_dtype_for(tok.get_vocab_size()))
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        arr.tofile(fh)
    os.replace(tmp, out_path)
    return {"part": os.path.basename(part_path), "tokens": int(arr.size), "docs": docs,
            "zh_tokens": int(sum(map(len, zh_ids))), "en_tokens": int(sum(map(len, en_ids)))}


def _sft_part(args) -> dict:
    tokenizer_dir, part_path, out_prefix, max_len = args
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    table = pq.read_table(part_path, columns=["zh", "en"])
    prompts = [f"<|im_start|>user\n{SFT_PROMPT_EN_ZH}{e}<|im_end|>\n<|im_start|>assistant\n"
               for e in table["en"].to_pylist()]
    replies = [f"{z}<|im_end|>\n" for z in table["zh"].to_pylist()]
    p_ids = [e.ids for e in tok.encode_batch(prompts, add_special_tokens=False)]
    r_ids = [e.ids for e in tok.encode_batch(replies, add_special_tokens=False)]
    dtype = _dtype_for(tok.get_vocab_size())
    tokens, offsets, prompt_lens, skipped = [], [0], [], 0
    for p, r in zip(p_ids, r_ids):
        if len(p) + len(r) > max_len:
            skipped += 1
            continue
        tokens.extend(p); tokens.extend(r)
        offsets.append(offsets[-1] + len(p) + len(r))
        prompt_lens.append(len(p))
    for suffix, arr in ((".bin", np.asarray(tokens, dtype)),
                        (".idx", np.asarray(offsets, np.int64)),
                        (".plen", np.asarray(prompt_lens, np.int32))):
        tmp = out_prefix + suffix + ".tmp"
        with open(tmp, "wb") as fh:
            arr.tofile(fh)
        os.replace(tmp, out_prefix + suffix)
    return {"part": os.path.basename(part_path), "records": len(prompt_lens), "skipped_too_long": skipped,
            "tokens": len(tokens)}


def build_shards(pairs_dir: str, tokenizer_dir: str, out_dir: str, *, workers: int = 16,
                 seed: int = 1234, mix: dict | None = None, sft_max_len: int = 512,
                 log=print) -> dict:
    """Tokenize every ``pairs/{train,valid}.*.parquet`` into pretrain + SFT shards."""
    mix = mix or {"mono": 0.5, "en_zh": 0.25, "zh_en": 0.25}
    t0 = time.time()
    with open(os.path.join(tokenizer_dir, "tokenizer_report.json"), encoding="utf-8") as fh:
        tok_meta = json.load(fh)
    jobs = []
    for split in ("train", "valid"):
        parts = sorted(p for p in os.listdir(pairs_dir) if p.startswith(split + ".") and p.endswith(".parquet"))
        os.makedirs(os.path.join(out_dir, "pretrain", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "sft", split), exist_ok=True)
        for i, name in enumerate(parts):
            stem = name[:-len(".parquet")]
            pre = os.path.join(out_dir, "pretrain", split, stem + ".bin")
            sft = os.path.join(out_dir, "sft", split, stem)
            if not os.path.exists(pre):
                jobs.append(("pretrain", split, _pretrain_part,
                             (tokenizer_dir, os.path.join(pairs_dir, name), pre, seed + i, mix)))
            if not os.path.exists(sft + ".plen"):
                jobs.append(("sft", split, _sft_part,
                             (tokenizer_dir, os.path.join(pairs_dir, name), sft, sft_max_len)))
    log(f"[shards] {len(jobs)} pending jobs, {workers} workers")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, args): (kind, split) for kind, split, fn, args in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            kind, split = futures[fut]
            info = fut.result() | {"kind": kind, "split": split}
            results.append(info)
            if i % 10 == 0 or i == len(futures):
                log(f"[shards] {i}/{len(futures)} ({time.time() - t0:.0f}s)")
    summary = {}
    for kind in ("pretrain", "sft"):
        for split in ("train", "valid"):
            folder = os.path.join(out_dir, kind, split)
            if kind == "pretrain":
                files = [f for f in os.listdir(folder) if f.endswith(".bin")]
                summary[f"{kind}_{split}_tokens"] = sum(os.path.getsize(os.path.join(folder, f)) for f in files) // 2
            else:
                files = [f for f in os.listdir(folder) if f.endswith(".plen")]
                summary[f"{kind}_{split}_records"] = sum(os.path.getsize(os.path.join(folder, f)) for f in files) // 4
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_s": round(time.time() - t0, 1), "tokenizer_fingerprint": tok_meta.get("fingerprint"),
                "tokenizer_dir": os.path.abspath(tokenizer_dir), "pairs_dir": os.path.abspath(pairs_dir),
                "mix": mix, "seed": seed, "sft_max_len": sft_max_len, "sft_prompt_en_zh": SFT_PROMPT_EN_ZH,
                "dtype": "uint16" if tok_meta.get("vocab_size", 0) <= 65535 else "uint32",
                **summary, "jobs_this_run": len(results)}
    with open(os.path.join(out_dir, "shards_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


class PackedTokenStream:
    """Random fixed-length windows over memory-mapped pretrain shards."""

    def __init__(self, folder: str, seq_len: int, dtype=np.uint16, seed: int = 0):
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".bin"))
        self.arrays = [np.memmap(f, dtype=dtype, mode="r") for f in files]
        self.arrays = [a for a in self.arrays if a.size > seq_len + 1]
        sizes = np.array([a.size for a in self.arrays], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.total_tokens = int(sizes.sum())
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int) -> np.ndarray:
        which = self.rng.choice(len(self.arrays), size=batch_size, p=self.weights)
        out = np.empty((batch_size, self.seq_len + 1), dtype=np.int32)
        for row, i in enumerate(which):
            arr = self.arrays[i]
            start = int(self.rng.integers(0, arr.size - self.seq_len - 1))
            out[row] = arr[start:start + self.seq_len + 1]
        return out


class SFTRecords:
    """Length-bucketed SFT batches with assistant-only loss masks."""

    def __init__(self, folder: str, dtype=np.uint16, seed: int = 0, pad_id: int = 0):
        stems = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".plen"))
        self.tokens, self.offsets, self.plens, self.index = [], [], [], []
        for si, stem in enumerate(stems):
            base = os.path.join(folder, stem)
            tok = np.memmap(base + ".bin", dtype=dtype, mode="r")
            off = np.fromfile(base + ".idx", dtype=np.int64)
            plen = np.fromfile(base + ".plen", dtype=np.int32)
            self.tokens.append(tok); self.offsets.append(off); self.plens.append(plen)
            self.index.append(np.stack([np.full(len(plen), si), np.arange(len(plen)),
                                        np.diff(off)], axis=1))
        self.index = np.concatenate(self.index) if self.index else np.zeros((0, 3), np.int64)
        # Sorted by length once: each batch is a contiguous similar-length
        # window found by binary search (O(log n), no full scan per batch).
        self.index = self.index[np.argsort(self.index[:, 2], kind="stable")]
        self.lengths = self.index[:, 2]
        self.rng = np.random.default_rng(seed)
        self.pad_id = pad_id

    def __len__(self):
        return len(self.index)

    def batch(self, token_budget: int, max_rows: int = 256, width_multiple: int = 64):
        """Sample a bucket of similar-length records filling ~token_budget tokens.

        The padded width is rounded up to ``width_multiple`` so the training
        graph sees a handful of shapes (Metal kernels are shape-specialized).
        """
        target = int(self.lengths[int(self.rng.integers(0, len(self.index)))])
        tol = max(4, target // 8)
        lo = int(np.searchsorted(self.lengths, target - tol, side="left"))
        hi = int(np.searchsorted(self.lengths, target + tol, side="right"))
        bucket = -(-(target + tol) // width_multiple) * width_multiple
        rows = max(1, min(max_rows, token_budget // max(bucket, 1), hi - lo))
        pick = self.index[lo + self.rng.choice(hi - lo, size=rows, replace=False)]
        width = int(pick[:, 2].max())
        width = -(-width // width_multiple) * width_multiple + 1   # +1: inputs/labels are shifted
        ids = np.full((len(pick), width), self.pad_id, dtype=np.int32)
        labels = np.full((len(pick), width), -100, dtype=np.int32)
        for r, (si, ri, n) in enumerate(pick):
            off = self.offsets[si][ri]
            seq = self.tokens[si][off:off + n]
            ids[r, :n] = seq
            p = int(self.plens[si][ri])
            labels[r, p:n] = seq[p:]
        return ids, labels


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Tokenize the pair corpus into training shards")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--sft-max-len", type=int, default=512)
    args = parser.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    manifest = build_shards(args.pairs, args.tokenizer, args.out, workers=args.workers,
                            sft_max_len=args.sft_max_len, log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in manifest.items() if k.endswith(("tokens", "records", "elapsed_s"))}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
