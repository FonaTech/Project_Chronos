"""Multi-turn chat records (tool calls, tool results) for SFT shards.

``render_chat`` reproduces ``chronos.tokenizer.model.CHAT_TEMPLATE`` in
Python and returns the text as ``(segment, supervised)`` pieces, so each
token's loss flag is known without guessing from offsets:

* learned: an assistant turn's content, its ``<tool_call>{json}</tool_call>``
  lines and the closing ``<|im_end|>\\n``;
* context: system/tool definitions, user turns, ``tool`` results and every
  ``<|im_start|>role\\n`` header (the generation prompt the model is given).

``encode_chat`` tokenizes the pieces one by one and concatenates the ids, the
same way the translation shards join prompt and reply encodings, so every
piece boundary is a token boundary.  ``write_sft_records`` writes the
``.bin/.idx/.plen/.lmask`` layout read by ``chronos.data.shards.SFTRecords``.

A record whose last turn is not an assistant turn ends with context tokens;
the trailing context is dropped (it carries no loss and would read as
padding to the router's real-token mask, which ends at the last supervised
token).
"""
from __future__ import annotations

import json
import os

import numpy as np

ROLES = ("system", "user", "assistant", "tool")


def _tool_json(value) -> str:
    # ``tojson`` as transformers overrides it for chat templates: key order
    # kept, no ASCII/HTML escaping, default ", " / ": " separators.
    return json.dumps(value, ensure_ascii=False)


def _call_parts(call: dict) -> dict:
    fn = call.get("function") if isinstance(call, dict) and call.get("function") else call
    if not isinstance(fn, dict) or "name" not in fn:
        raise ValueError(f"tool call needs a name: {call!r}")
    args = fn.get("arguments", {})
    return {"name": fn["name"], "arguments": args}


def render_chat(messages, tools=None, *, add_generation_prompt: bool = False) -> list[tuple[str, bool]]:
    """``[(text, supervised)]`` pieces whose concatenation is the template text."""
    pieces: list[tuple[str, bool]] = []
    messages = list(messages or [])
    for m in messages:
        if m.get("role") not in ROLES:
            raise ValueError(f"unknown chat role {m.get('role')!r}; expected one of {ROLES}")
    skip_first = bool(tools) and bool(messages) and messages[0]["role"] == "system"
    if tools:
        head = "<|im_start|>system\n"
        if skip_first:
            head += (messages[0].get("content") or "") + "\n\n"
        head += "<tools>\n" + "".join(_tool_json(t) + "\n" for t in tools) + "</tools><|im_end|>\n"
        pieces.append((head, False))
    for i, m in enumerate(messages):
        if skip_first and i == 0:
            continue
        role, text = m["role"], m.get("content") or ""
        pieces.append((f"<|im_start|>{role}\n", False))
        if role != "assistant":
            pieces.append((text + "<|im_end|>\n", False))
            continue
        body = text
        for j, call in enumerate(m.get("tool_calls") or []):
            if text or j:
                body += "\n"
            body += "<tool_call>" + _tool_json(_call_parts(call)) + "</tool_call>"
        pieces.append((body + "<|im_end|>\n", True))
    if add_generation_prompt:
        pieces.append(("<|im_start|>assistant\n", False))
    return [(t, s) for t, s in pieces if t]


def _ids(tokenizer, text: str) -> list[int]:
    """Token ids from a ``ChronosTokenizer``, HF ``tokenizers.Tokenizer`` or
    ``transformers`` tokenizer (no BOS/EOS added)."""
    try:
        out = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        out = tokenizer.encode(text)
    return list(getattr(out, "ids", out))


def encode_chat(tokenizer, messages, tools=None) -> tuple[np.ndarray, np.ndarray]:
    """``(ids int64, supervised uint8)`` for one conversation; trailing context
    after the last assistant turn is dropped.  Raises if nothing is learned."""
    ids: list[int] = []
    mask: list[int] = []
    for text, supervised in render_chat(messages, tools):
        piece = _ids(tokenizer, text)
        ids.extend(piece)
        mask.extend([int(supervised)] * len(piece))
    if 1 not in mask:
        raise ValueError("conversation has no assistant turn to learn")
    end = len(mask) - mask[::-1].index(1)
    return np.asarray(ids[:end], np.int64), np.asarray(mask[:end], np.uint8)


def write_sft_records(prefix: str, records, *, dtype=np.uint16, max_len: int | None = None) -> dict:
    """Write ``(ids, supervised)`` records to ``<prefix>.{bin,idx,plen,lmask}``.

    ``.plen`` is each record's first supervised offset (older readers apply
    the prompt rule); ``.lmask`` carries the exact per-token flags.  Records
    longer than ``max_len`` are skipped and counted.  Files are written to
    ``.tmp`` names and renamed, so a reader never sees a partial set.
    """
    limit = np.iinfo(dtype).max
    tokens, masks, offsets, plens, skipped = [], [], [0], [], 0
    for ids, mask in records:
        ids, mask = np.asarray(ids), np.asarray(mask, np.uint8)
        if ids.shape != mask.shape or ids.ndim != 1:
            raise ValueError("each record needs 1-D ids and a mask of the same length")
        if max_len is not None and ids.size > max_len:
            skipped += 1
            continue
        if ids.size and (int(ids.max()) > limit or int(ids.min()) < 0):
            raise ValueError(f"token id outside {np.dtype(dtype).name}")
        hits = np.flatnonzero(mask)
        if not hits.size:
            raise ValueError("record has no supervised token")
        tokens.append(ids.astype(dtype)); masks.append(mask)
        offsets.append(offsets[-1] + ids.size); plens.append(int(hits[0]))
    arrays = {
        ".bin": np.concatenate(tokens) if tokens else np.zeros(0, dtype),
        ".idx": np.asarray(offsets, np.int64),
        ".plen": np.asarray(plens, np.int32),
        ".lmask": np.concatenate(masks) if masks else np.zeros(0, np.uint8),
    }
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    for suffix, arr in arrays.items():
        tmp = prefix + suffix + ".tmp"
        with open(tmp, "wb") as fh:
            arr.tofile(fh)
    for suffix in (".bin", ".idx", ".lmask", ".plen"):   # .plen last: readers key on it
        os.replace(prefix + suffix + ".tmp", prefix + suffix)
    return {"records": len(plens), "skipped_too_long": skipped, "tokens": int(offsets[-1]),
            "supervised_tokens": int(arrays[".lmask"].sum())}
