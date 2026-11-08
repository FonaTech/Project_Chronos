"""Parse ``<tool_call>{json}</tool_call>`` blocks out of generated text.

The chat template (``chronos.tokenizer.model.CHAT_TEMPLATE``) teaches the
model to emit tool calls as ``<tool_call>{"name": ..., "arguments": {...}}
</tool_call>`` lines after (optional) assistant text.  ``ToolCallStream``
consumes decoded text pieces as they stream and splits them into visible
content and parsed calls, holding back only what could still be the start
of a tag, so clients never see half a tag.

A block whose JSON does not parse, or that has no ``name``, is returned as
ordinary content: nothing the model wrote is dropped.
"""
from __future__ import annotations

import json
import uuid

OPEN, CLOSE = "<tool_call>", "</tool_call>"


def _prefix_hold(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def parse_call(body: str) -> dict | None:
    """``{"name": str, "arguments": dict}`` or ``None`` for an invalid block."""
    try:
        obj = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = fn.get("arguments", fn.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


class ToolCallStream:
    """Incremental splitter: ``feed(piece) -> (content, calls)``, then ``finish()``.

    Whitespace between consecutive calls (the template's ``"\\n"`` joiner) is
    not reported as content once a call has been parsed."""

    def __init__(self):
        self._buf = ""          # undecided text (outside a block)
        self._block = None      # text inside an open block, or None
        self._pending_ws = ""   # whitespace after a call, emitted only if real text follows
        self.calls: list[dict] = []

    def _emit_text(self, text: str) -> str:
        if not text:
            return ""
        if self.calls and not text.strip():
            self._pending_ws += text
            return ""
        out, self._pending_ws = self._pending_ws + text, ""
        return out

    def feed(self, piece: str) -> tuple[str, list[dict]]:
        content, calls = [], []
        self._buf += piece or ""
        while True:
            if self._block is None:
                i = self._buf.find(OPEN)
                if i < 0:
                    keep = _prefix_hold(self._buf, OPEN)
                    content.append(self._emit_text(self._buf[: len(self._buf) - keep]))
                    self._buf = self._buf[len(self._buf) - keep:]
                    break
                content.append(self._emit_text(self._buf[:i]))
                self._block, self._buf = "", self._buf[i + len(OPEN):]
            else:
                j = self._buf.find(CLOSE)
                if j < 0:
                    keep = _prefix_hold(self._buf, CLOSE)
                    self._block += self._buf[: len(self._buf) - keep]
                    self._buf = self._buf[len(self._buf) - keep:]
                    break
                body, self._buf = self._block + self._buf[:j], self._buf[j + len(CLOSE):]
                self._block = None
                call = parse_call(body.strip())
                if call is None:
                    content.append(self._emit_text(OPEN + body + CLOSE))
                else:
                    self._pending_ws = ""
                    self.calls.append(call)
                    calls.append(call)
        return "".join(content), calls

    def finish(self) -> tuple[str, list[dict]]:
        """Flush held text; an unterminated block is returned as content."""
        tail = self._buf if self._block is None else OPEN + self._block + self._buf
        self._buf, self._block = "", None
        text = tail if not (self.calls and not tail.strip()) else ""
        self._pending_ws = ""
        return text, []


def split_tool_calls(text: str) -> tuple[str, list[dict]]:
    """One-shot form of ``ToolCallStream``."""
    stream = ToolCallStream()
    content, calls = stream.feed(text)
    tail, _ = stream.finish()
    return (content + tail).strip() if calls else content + tail, calls


def ollama_tool_calls(calls: list[dict]) -> list[dict]:
    """Ollama ``message.tool_calls`` (arguments stay an object)."""
    return [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]


def openai_tool_calls(calls: list[dict], *, start: int = 0) -> list[dict]:
    """OpenAI ``tool_calls`` (arguments are a JSON string, each call has an id)."""
    return [{"index": start + k, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
            for k, c in enumerate(calls)]
