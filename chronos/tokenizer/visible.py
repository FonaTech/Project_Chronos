"""Decode generated ids for display without losing structural tags.

``skip_special_tokens=True`` strips every *special* token.  A custom HF
tokenizer may register ``<tool_call>`` / ``</tool_call>`` (and ``<tools>``)
as special or added tokens; plain skipping then deletes the tags and the
serving tool-call parser sees bare JSON.  :func:`decode_visible` drops only
control tokens (``<|im_end|>``, padding, BOS/EOS, ...) and keeps the tags as
text, for HF tokenizers and the native ``ChronosTokenizer`` alike.
"""
from __future__ import annotations

from typing import Iterable

# Tags the chat template writes as text and the server parses back.
STRUCTURAL_TAGS: tuple[str, ...] = ("<tool_call>", "</tool_call>", "<tools>", "</tools>")


def _token_id(tokenizer, token: str) -> int | None:
    vocab_lookup = getattr(tokenizer, "convert_tokens_to_ids", None)
    if vocab_lookup is not None:
        tid = vocab_lookup(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and (unk is None or tid != unk or token == getattr(tokenizer, "unk_token", None)):
            return tid
        return None
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, dict):
        tid = vocab.get(token)
        return int(tid) if tid is not None else None
    return None


def control_token_ids(tokenizer, keep: Iterable[str] = STRUCTURAL_TAGS) -> frozenset[int]:
    """Ids of special/added tokens to hide, i.e. all of them except ``keep``.

    Cached on the tokenizer object; the set only changes if tokens are added
    later, which the cache key (vocab size) detects.
    """
    keep = tuple(keep)
    size = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    cache = getattr(tokenizer, "_chronos_control_ids", None)
    if cache is not None and cache[0] == (size, keep):
        return cache[1]
    keep_ids = {i for i in (_token_id(tokenizer, t) for t in keep) if i is not None}
    hidden: set[int] = set(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or ()))
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(decoder, dict):          # HF: added tokens flagged ``special``
        hidden.update(int(i) for i, tok in decoder.items() if getattr(tok, "special", False))
    special = getattr(tokenizer, "special_tokens", None)
    if isinstance(special, (list, tuple)):  # native ChronosTokenizer
        hidden.update(i for i in (_token_id(tokenizer, t) for t in special) if i is not None)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)   # minimal tokenizers expose only these
        if isinstance(tid, int):
            hidden.add(tid)
    ids = frozenset(hidden - keep_ids)
    try:
        tokenizer._chronos_control_ids = ((size, keep), ids)
    except Exception:                       # slotted / frozen tokenizer objects
        pass
    return ids


def decode_visible(tokenizer, ids, keep: Iterable[str] = STRUCTURAL_TAGS) -> str:
    """Decode ``ids`` hiding control tokens but keeping structural tags."""
    hidden = control_token_ids(tokenizer, keep)
    kept = [int(i) for i in ids if int(i) not in hidden]
    if not kept:
        return ""
    if hasattr(tokenizer, "special_tokens") and not hasattr(tokenizer, "all_special_ids"):
        return tokenizer.decode(kept, skip_special=False)       # ChronosTokenizer
    return tokenizer.decode(kept, skip_special_tokens=False)
