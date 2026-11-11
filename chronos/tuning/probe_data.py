"""Real-token batches for the auto-tune probe.

The probe used to train on random token ids, so no hyperparameter could change its loss in a way that
carries over to real data.  ``load_probe_tokens`` encodes a bounded sample of the user's dataset with the
model's tokenizer; the tuner falls back to random ids only when the data cannot be read.
"""
from __future__ import annotations

import os

import numpy as np


def load_probe_tokens(dataset_path: str, tokenizer_dir: str | None, *, max_tokens: int = 400_000,
        max_records: int = 20_000) -> tuple[np.ndarray, int] | None:
    """``(token ids int64, tokenizer vocab size)`` or ``None`` when nothing could be read or encoded."""
    try:
        from chronos.deps import get_general_tokenizer_path
        from chronos.tokenizer.extract import _texts_from_record, expand, iter_records
        from chronos.tokenizer.model import ChronosTokenizer
        directory = tokenizer_dir if tokenizer_dir and os.path.isfile(os.path.join(tokenizer_dir, 'chronos_tokenizer.json')) \
            else get_general_tokenizer_path()
        tokenizer = ChronosTokenizer.load(directory)
        ids: list[int] = []
        seen = 0
        for path in expand([dataset_path]):
            for record in iter_records(path):
                seen += 1
                for text in _texts_from_record(record, None):
                    ids.extend(tokenizer.encode(text, allow_special=False))
                    ids.append(tokenizer.vocab['<|endoftext|>'])
                if len(ids) >= max_tokens or seen >= max_records:
                    break
            if len(ids) >= max_tokens or seen >= max_records:
                break
        if len(ids) < 512:
            return None
        return np.asarray(ids[:max_tokens], dtype=np.int64), int(tokenizer.vocab_size)
    except Exception:
        return None


def probe_batch(tokens: np.ndarray, step: int, batch: int, seq_len: int) -> np.ndarray:
    """``[batch, seq_len]`` windows at deterministic offsets that advance with ``step`` (same data in every trial)."""
    span = max(1, tokens.size - seq_len - 1)
    starts = [((step * batch + row) * seq_len * 7919) % span for row in range(batch)]
    return np.stack([tokens[s:s + seq_len] for s in starts])
