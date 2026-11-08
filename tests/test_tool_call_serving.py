"""Tool-call inference: <tool_call> parsing and Ollama/OpenAI response shapes."""
from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.serving.tool_calls import ToolCallStream, parse_call, split_tool_calls  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
CALL_TEXT = ('Let me check.\n<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>\n'
             '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>')


def _pieces(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_stream_splits_any_chunking_identically(size):
    s = ToolCallStream()
    content, calls = "", []
    for piece in _pieces(CALL_TEXT, size):
        c, k = s.feed(piece)
        assert "<tool" not in c and "</tool" not in c        # no partial tag ever leaks
        content += c
        calls += k
    tail, _ = s.finish()
    assert (content + tail).strip() == "Let me check."
    assert calls == [{"name": "get_weather", "arguments": {"city": "北京"}}, {"name": "get_time", "arguments": {}}]
    assert split_tool_calls(CALL_TEXT) == ("Let me check.", calls)


def test_plain_text_and_invalid_blocks_are_kept():
    assert split_tool_calls("no tools here <tool") == ("no tools here <tool", [])
    bad = "x <tool_call>{not json}</tool_call> y"
    assert split_tool_calls(bad) == (bad, [])
    assert split_tool_calls('<tool_call>{"arguments": {}}</tool_call>')[1] == []          # no name
    assert split_tool_calls("a <tool_call>{\"name\": \"f\"")[0] == "a <tool_call>{\"name\": \"f\""  # unterminated
    assert parse_call('{"function": {"name": "f", "arguments": "{\\"a\\": 1}"}}') == {"name": "f", "arguments": {"a": 1}}
    assert parse_call('{"name": "f", "arguments": [1]}') is None
    assert parse_call('{"name": "f"}') == {"name": "f", "arguments": {}}


def test_round_trip_with_training_template():
    """What the template teaches the model to write is exactly what the server parses."""
    from chronos.data.chat_records import render_chat
    calls = [{"function": {"name": "get_weather", "arguments": {"city": "北京"}}}, {"name": "get_time", "arguments": {}}]
    msgs = [{"role": "user", "content": "天气?"}, {"role": "assistant", "content": "Let me check.", "tool_calls": calls}]
    learned = "".join(t for t, sup in render_chat(msgs, TOOLS) if sup).removesuffix("<|im_end|>\n")
    assert learned == CALL_TEXT
    assert split_tool_calls(learned) == ("Let me check.", [{"name": "get_weather", "arguments": {"city": "北京"}},
                                                           {"name": "get_time", "arguments": {}}])


class _Runtime:
    """Stub runtime: streams fixed text in small pieces."""

    def __init__(self, text, piece=3):
        self.text, self.piece, self.tools_seen = text, piece, []

    def available(self):
        return {"m": __file__}

    def get(self, name):
        return type("LM", (), {"name": name})()

    def encode_chat(self, lm, messages, tools=None):
        self.tools_seen.append(tools)
        return [1, 2, 3]

    def generate_stream(self, name, ids, **_opts):
        for p in _pieces(self.text, self.piece):
            yield {"text": p}
        yield {"done": True, "done_reason": "stop", "prompt_tokens": 3, "completion_tokens": 9}


def _client(text, piece=3):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from chronos.serving.ollama_server import create_app
    rt = _Runtime(text, piece)
    return TestClient(create_app(rt)), rt


def test_ollama_chat_returns_tool_calls_stream_and_non_stream():
    body = {"model": "m", "messages": [{"role": "user", "content": "天气?"}], "tools": TOOLS}
    client, rt = _client(CALL_TEXT)
    msg = client.post("/api/chat", json={**body, "stream": False}).json()["message"]
    assert msg["content"] == "Let me check."
    assert msg["tool_calls"] == [{"function": {"name": "get_weather", "arguments": {"city": "北京"}}},
                                 {"function": {"name": "get_time", "arguments": {}}}]
    assert rt.tools_seen[-1] == TOOLS
    lines = [json.loads(x) for x in client.post("/api/chat", json=body).text.splitlines() if x.strip()]
    assert lines[-1]["done"] is True
    streamed = [c for line in lines for c in line["message"].get("tool_calls", [])]
    assert streamed == msg["tool_calls"]
    assert "".join(line["message"]["content"] for line in lines).strip() == "Let me check."
    assert not any("<tool" in line["message"]["content"] for line in lines)


def test_ollama_chat_without_tools_is_untouched():
    client, _ = _client(CALL_TEXT)
    msg = client.post("/api/chat", json={"model": "m", "stream": False,
                                         "messages": [{"role": "user", "content": "hi"}]}).json()["message"]
    assert msg == {"role": "assistant", "content": CALL_TEXT}


def test_openai_chat_tool_calls_stream_and_non_stream():
    body = {"model": "m", "messages": [{"role": "user", "content": "天气?"}], "tools": TOOLS}
    client, _ = _client(CALL_TEXT, piece=4)
    out = client.post("/v1/chat/completions", json=body).json()["choices"][0]
    assert out["finish_reason"] == "tool_calls"
    assert out["message"]["content"] == "Let me check."
    calls = out["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "北京"}
    assert all(c["type"] == "function" and c["id"].startswith("call_") and "index" not in c for c in calls)
    events = [json.loads(x[6:]) for x in client.post("/v1/chat/completions", json={**body, "stream": True}).text.split("\n\n")
              if x.startswith("data: {")]
    deltas = [e["choices"][0]["delta"] for e in events]
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    streamed = [c for d in deltas for c in d.get("tool_calls", [])]
    assert [c["index"] for c in streamed] == [0, 1]
    assert [c["function"]["name"] for c in streamed] == ["get_weather", "get_time"]
    assert "".join(d.get("content", "") for d in deltas).strip() == "Let me check."


def test_openai_plain_answer_with_tools_offered():
    client, _ = _client("It is sunny.")
    out = client.post("/v1/chat/completions", json={"model": "m", "tools": TOOLS,
                                                     "messages": [{"role": "user", "content": "hi"}]}).json()
    choice = out["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["message"] == {"role": "assistant", "content": "It is sunny."}
