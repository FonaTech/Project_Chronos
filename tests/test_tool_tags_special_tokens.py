"""Tool-call tags survive decoding when a custom HF tokenizer marks them special."""
from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.tokenizer.visible import control_token_ids, decode_visible  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
ANSWER = 'Let me check.\n<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'


def _hf_tokenizer(tags_special: bool = True):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, show_progress=False,
                                  special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator([ANSWER, "Let me check the weather."], trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tags = ["<tool_call>", "</tool_call>", "<tools>", "</tools>"]
    if tags_special:
        tok.add_special_tokens({"additional_special_tokens": tags})
    else:
        tok.add_tokens(tags)
    return tok


@pytest.mark.parametrize("tags_special", [True, False])
def test_decode_visible_keeps_tags_drops_control(tags_special):
    tok = _hf_tokenizer(tags_special)
    ids = tok(ANSWER + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert tok.convert_tokens_to_ids("<tool_call>") in ids          # registered as one token
    if tags_special:
        # The bug this guards against: plain skipping deletes the tags.
        assert "<tool_call>" not in tok.decode(ids, skip_special_tokens=True)
    assert decode_visible(tok, ids) == ANSWER
    hidden = control_token_ids(tok)
    assert tok.convert_tokens_to_ids("<|im_end|>") in hidden
    assert tok.convert_tokens_to_ids("<tool_call>") not in hidden


def test_decode_visible_native_tokenizer():
    from chronos.tokenizer.model import ChronosTokenizer, special_tokens_with
    from chronos.tokenizer.trainer import BPETrainerConfig, train_bpe
    special = special_tokens_with(("<tool_call>", "</tool_call>"))
    trained = train_bpe({"en": [ANSWER] * 4}, BPETrainerConfig(vocab_size=300 + len(special), special_tokens=special))
    tok = ChronosTokenizer.from_trained(trained, special)
    ids = tok.encode(ANSWER + "<|im_end|>")
    assert tok.vocab["<tool_call>"] in ids
    assert decode_visible(tok, ids) == ANSWER


def test_serving_stream_parses_special_tool_tags(monkeypatch):
    """generate_stream → /api/chat and /v1/chat/completions with special tags."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import chronos.runtime.speculative as spec
    from chronos.serving.ollama_server import create_app
    from chronos.serving.runtime import ChronosRuntime, LoadedModel

    tok = _hf_tokenizer(True)
    script = tok(ANSWER, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]

    def fake_generate(model, input_ids, *, on_token=None, **_kw):
        for t in script:                       # one token at a time, as decoding does
            on_token(int(t))
        return input_ids, spec.SpecStats()

    monkeypatch.setattr(spec, "speculative_generate", fake_generate)
    cfg = type("Cfg", (), {"recurrent_depth": 1, "max_inference_depth": None})()
    rt = ChronosRuntime(None)
    rt.loaded["m"] = LoadedModel(name="m", path="m.pth", config=cfg, model=None, tokenizer=tok,
                                 engine=None, device="cpu", lazy_experts=False)
    rt.encode_chat = lambda lm, messages, tools=None: [1, 2, 3]
    client = TestClient(create_app(rt))
    body = {"model": "m", "messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS, "stream": False}

    msg = client.post("/api/chat", json=body).json()["message"]
    assert msg["content"] == "Let me check."
    assert msg["tool_calls"] == [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]

    out = client.post("/v1/chat/completions", json=body).json()["choices"][0]
    assert out["finish_reason"] == "tool_calls"
    assert json.loads(out["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}

    events = [json.loads(x[6:]) for x in client.post("/v1/chat/completions", json={**body, "stream": True}).text.split("\n\n")
              if x.startswith("data: {")]
    streamed = [c for e in events for c in e["choices"][0]["delta"].get("tool_calls", [])]
    assert [c["function"]["name"] for c in streamed] == ["get_weather"]
    assert not any("<tool" in e["choices"][0]["delta"].get("content", "") for e in events)
