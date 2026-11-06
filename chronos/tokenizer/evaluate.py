"""Tokenizer quality metrics: compression, fertility, fallback, round-trip."""
from __future__ import annotations

import re

_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def evaluate_tokenizer(tokenizer, texts_by_bucket: dict[str, list[str]], limit: int = 5000) -> dict:
    from .model import tokenizer_normalize
    byte_ids = set(tokenizer._byte_ids)
    used = set()
    report = {}
    for bucket, texts in texts_by_bucket.items():
        sample = texts[:limit]
        if not sample:
            continue
        tokens = byte_tokens = chars = bytes_ = words = han = roundtrip_fail = 0
        for text in sample:
            ids = tokenizer.encode(text, allow_special=False)
            used.update(ids)
            tokens += len(ids)
            byte_tokens += sum(1 for i in ids if i in byte_ids)
            chars += len(text)
            bytes_ += len(text.encode("utf-8"))
            words += len(text.split())
            han += len(_HAN.findall(text))
            if tokenizer.decode(ids) != tokenizer_normalize(text):
                roundtrip_fail += 1
        report[bucket] = {
            "texts": len(sample),
            "bytes_per_token": round(bytes_ / max(tokens, 1), 3),
            "chars_per_token": round(chars / max(tokens, 1), 3),
            "tokens_per_word": round(tokens / max(words, 1), 3),
            "tokens_per_han": round(tokens / han, 3) if han else None,
            "byte_fallback_rate": round(byte_tokens / max(tokens, 1), 5),
            "roundtrip_failures": roundtrip_fail,
        }
    report["vocab_utilization"] = round(len(used) / max(tokenizer.vocab_size, 1), 4)
    return report
