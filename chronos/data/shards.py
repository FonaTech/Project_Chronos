"""Readers for tokenized, memory-mapped training shards (any task, any language).

Layout under a shard folder (written by any task's own builder):

* ``pretrain/{train,valid}/*.bin``: one flat uint16/uint32 token array per
  file, documents separated by the EOS token; the reader samples fixed windows.
* ``sft/{train,valid}/<stem>.{bin,idx,plen}``: chat-template records in a
  token array (``.bin``), int64 record offsets (``.idx``, n+1 entries) and
  int32 prompt lengths (``.plen``).  Only tokens after the prompt are
  supervised, so the loss covers the assistant reply alone.
* optional ``sft/.../<stem>.lmask``: one uint8 per token of ``.bin`` (1 =
  supervised).  Multi-turn records (tool calls, tool results, several
  assistant turns) need it: the assistant's text and ``tool_calls`` are
  learned, user/system/tool-result tokens are context.  When present it
  replaces the ``.plen`` rule for that stem; ``.plen`` stays the first
  supervised offset so older readers keep working.

An optional ``shards_manifest.json`` may record ``"dtype": "uint16"|"uint32"``.
"""
from __future__ import annotations

import json
import os

import numpy as np


def dtype_for(vocab_size: int):
    """Smallest unsigned dtype that holds every token id of ``vocab_size``."""
    return np.uint16 if vocab_size <= 65535 else np.uint32


_dtype_for = dtype_for       # pre-0.3.1a2 name


def manifest_dtype(shards_dir: str, default=np.uint16):
    """Token dtype recorded in ``<shards_dir>/shards_manifest.json`` (``default`` if absent)."""
    path = os.path.join(shards_dir, "shards_manifest.json")
    try:
        with open(path, encoding="utf-8") as fh:
            name = json.load(fh).get("dtype")
    except (OSError, ValueError):
        return default
    return {"uint16": np.uint16, "uint32": np.uint32}.get(name, default)


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
        self.lmasks = []
        for si, stem in enumerate(stems):
            base = os.path.join(folder, stem)
            tok = np.memmap(base + ".bin", dtype=dtype, mode="r")
            off = np.fromfile(base + ".idx", dtype=np.int64)
            plen = np.fromfile(base + ".plen", dtype=np.int32)
            lmask = None
            if os.path.exists(base + ".lmask"):
                lmask = np.memmap(base + ".lmask", dtype=np.uint8, mode="r")
                if lmask.size != tok.size:
                    raise ValueError(f"{base}.lmask has {lmask.size} entries for {tok.size} tokens")
            self.tokens.append(tok); self.offsets.append(off); self.plens.append(plen)
            self.lmasks.append(lmask)
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
            lm = self.lmasks[si]
            if lm is not None:
                keep = np.asarray(lm[off:off + n]).astype(bool)
                # int32 first: under NumPy 2 (NEP 50) np.where keeps uint16 and wraps -100 → 65436.
                labels[r, :n] = np.where(keep, np.asarray(seq, dtype=np.int32), -100)
            else:
                p = int(self.plens[si][ri])
                labels[r, p:n] = seq[p:]
        return ids, labels

    def packed_batch(self, token_budget: int, seq_len: int, sep_id: int = 0):
        """Rows of exactly ``seq_len + 1`` tokens, each filled with whole records.

        Records are taken in random order and joined by one ``sep_id`` token
        (the document separator that ``segment_ids`` cuts on); a record that
        does not fit is cut at the row end, a record longer than a row is
        truncated.  Labels keep the loss mask of each record; separators and
        the row's unused tail carry no loss.  Compared with ``batch`` there is no
        length bucketing and almost no padding, so short SFT data wastes no compute."""
        rows = max(1, int(token_budget) // int(seq_len))
        width = int(seq_len) + 1
        ids = np.full((rows, width), self.pad_id, dtype=np.int32)
        labels = np.full((rows, width), -100, dtype=np.int32)
        for r in range(rows):
            pos = 0
            while pos < width:
                si, ri, n = self.index[int(self.rng.integers(0, len(self.index)))]
                off = self.offsets[si][ri]
                take = min(int(n), width - pos)
                seq = np.asarray(self.tokens[si][off:off + take], dtype=np.int32)
                ids[r, pos:pos + take] = seq
                lm = self.lmasks[si]
                if lm is not None:
                    keep = np.asarray(lm[off:off + take]).astype(bool)
                else:
                    keep = np.arange(take) >= int(self.plens[si][ri])
                labels[r, pos:pos + take] = np.where(keep, seq, -100)
                pos += take
                if pos < width:
                    ids[r, pos] = sep_id   # closes the document; carries no loss
                    pos += 1
        return ids, labels

    def supervised_tokens(self) -> int:
        """Tokens that carry loss over every record (``.lmask`` or ``.plen`` rule)."""
        total = 0
        for off, plen, lm in zip(self.offsets, self.plens, self.lmasks):
            total += int(np.asarray(lm).sum()) if lm is not None else int(off[-1]) - int(plen.sum())
        return total
