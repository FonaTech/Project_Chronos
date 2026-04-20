"""
chronos/data/flexible_dataset.py

Streaming JSONL loader with byte-offset indexing. Only the offsets list
(8 bytes/sample) lives in RAM — records are parsed on demand via
seek()+readline()+json.loads() inside __getitem__. This keeps memory
flat for multi-GB pretrain corpora where the prior "load all into a
Python list" path would OOM.

Supported formats (auto-detected from the first record):
  {"text": "..."}                    ← minimind pretrain
  {"content": "..."}                 ← common HF datasets
  {"instruction": "...", "output": "..."}  ← Alpaca-style
  {"conversations": [...]}           ← ShareGPT-style (SFT)
  {"prompt": "...", "response": "..."}
  {"input": "...", "output": "..."}
  {"question": "...", "answer": "..."}
  any JSON with string values        ← last-resort: join all strings
"""
import json
import os
import threading

import torch
from torch.utils.data import Dataset

_TEXT_KEYS = ("text", "content", "story", "passage", "document", "article")
_PAIR_KEYS = (
    ("instruction", "output"),
    ("instruction", "response"),
    ("prompt", "response"),
    ("prompt", "completion"),
    ("input", "output"),
    ("question", "answer"),
    ("query", "answer"),
)


def _extract_text(record: dict) -> str:
    for k in _TEXT_KEYS:
        if k in record and record[k]:
            return str(record[k])

    for k1, k2 in _PAIR_KEYS:
        if k1 in record and k2 in record:
            parts = [str(record[k1]).strip(), str(record[k2]).strip()]
            return "\n".join(p for p in parts if p)

    if "conversations" in record:
        convs = record["conversations"]
        if isinstance(convs, list):
            return " ".join(
                str(c.get("value", c.get("content", "")))
                for c in convs
            )

    if "messages" in record:
        msgs = record["messages"]
        if isinstance(msgs, list):
            return " ".join(str(m.get("content", "")) for m in msgs)

    parts = [str(v) for v in record.values()
             if isinstance(v, (str, int, float)) and str(v).strip()]
    return " ".join(parts)


class FlexibleDataset(Dataset):
    """Streaming JSONL dataset. Memory is O(N · 8B) via offset index."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = os.path.abspath(data_path)

        # Build the offset index by streaming the file once. Each entry
        # points to the first byte of a non-empty line. We skip blank
        # lines but do NOT parse JSON yet — that happens lazily.
        self.offsets: list[int] = []
        with open(self.data_path, "rb") as f:
            pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                # Record offset only if the line has non-whitespace bytes.
                # Empty lines and pure-whitespace lines are skipped so
                # __getitem__ never has to handle them.
                if line.strip():
                    self.offsets.append(pos)
                pos = f.tell()

        if not self.offsets:
            raise ValueError(f"No valid JSON records found in {data_path}")

        # File handle is opened per-worker in __getitem__; a shared
        # handle would break under DataLoader num_workers>0 via fork.
        self._fh = None
        self._fh_lock = threading.Lock()

        # Peek at the first record to report detected format. This reuses
        # the same lazy path we use at runtime, so any parse failures
        # surface here instead of mid-training.
        first = self._read_record(0)
        detected = _extract_text(first)
        field_hint = (
            next((k for k in _TEXT_KEYS if k in first), None)
            or next((f"{k1}+{k2}" for k1, k2 in _PAIR_KEYS
                     if k1 in first and k2 in first), None)
            or ("conversations" if "conversations" in first else "auto")
        )
        print(f"[FlexibleDataset] {len(self.offsets)} records (streaming), "
              f"detected format: '{field_hint}', "
              f"sample preview: {detected[:80]!r}")

    def _get_fh(self):
        """Lazy per-process file handle. Reopened after fork (pid changes)."""
        pid = os.getpid()
        if self._fh is None or getattr(self._fh, "_chronos_pid", None) != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.data_path, "rb")
            self._fh._chronos_pid = pid
        return self._fh

    def _read_record(self, index: int) -> dict:
        offset = self.offsets[index]
        with self._fh_lock:
            fh = self._get_fh()
            fh.seek(offset)
            raw = fh.readline()
        return json.loads(raw.decode("utf-8"))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        record = self._read_record(index)
        text = _extract_text(record)
        tok = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        )
        tokens = (
            [self.tokenizer.bos_token_id]
            + tok.input_ids
            + [self.tokenizer.eos_token_id]
        )
        pad_id = self.tokenizer.pad_token_id or 0
        tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        return input_ids, labels

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
