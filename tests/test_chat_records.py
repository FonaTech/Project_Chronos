"""Tool-call chat template, multi-turn loss masks and ``.lmask`` SFT shards."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.data.chat_records import encode_chat, render_chat, write_sft_records
from chronos.tokenizer.model import CHAT_TEMPLATE

OLD_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

PLAIN = [{"role": "system", "content": "你是翻译助手。"},
         {"role": "user", "content": "翻译成中文：Hello"},
         {"role": "assistant", "content": "你好"}]

TOOL_CHAT = [
    {"role": "system", "content": "You can call tools."},
    {"role": "user", "content": "北京和上海今天天气？"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "北京"}}},
        {"name": "get_weather", "arguments": {"city": "上海"}}]},
    {"role": "tool", "content": "{\"temp\": 12}"},
    {"role": "tool", "content": "{\"temp\": 18}"},
    {"role": "assistant", "content": "北京 12°C，上海 18°C。"},
]


def _jinja(template, messages, tools=None, add_generation_prompt=False):
    pytest.importorskip("transformers")
    from transformers.utils.chat_template_utils import render_jinja_template
    rendered, _ = render_jinja_template(conversations=[messages], tools=tools, chat_template=template,
                                        add_generation_prompt=add_generation_prompt)
    return rendered[0]


@pytest.mark.parametrize("gen", [False, True])
def test_template_without_tools_is_unchanged(gen):
    for msgs in (PLAIN, PLAIN[1:], PLAIN[1:2]):
        assert _jinja(CHAT_TEMPLATE, msgs, add_generation_prompt=gen) == _jinja(OLD_TEMPLATE, msgs, add_generation_prompt=gen)


@pytest.mark.parametrize("messages,tools", [
    (PLAIN, None), (TOOL_CHAT, TOOLS), (TOOL_CHAT[1:], TOOLS), (PLAIN, TOOLS),
    ([{"role": "user", "content": "x"},
      {"role": "assistant", "content": "先查一下。", "tool_calls": [{"name": "f", "arguments": "{\"a\": 1}"}]}], TOOLS),
])
def test_python_render_matches_jinja_template(messages, tools):
    text = "".join(t for t, _ in render_chat(messages, tools))
    assert text == _jinja(CHAT_TEMPLATE, messages, tools)
    gen = "".join(t for t, _ in render_chat(messages, tools, add_generation_prompt=True))
    assert gen == _jinja(CHAT_TEMPLATE, messages, tools, add_generation_prompt=True)


def test_tool_call_text_layout():
    text = "".join(t for t, _ in render_chat(TOOL_CHAT, TOOLS))
    assert text.startswith("<|im_start|>system\nYou can call tools.\n\n<tools>\n{\"type\": \"function\"")
    assert ('<|im_start|>assistant\n<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>\n'
            '<tool_call>{"name": "get_weather", "arguments": {"city": "上海"}}</tool_call><|im_end|>\n') in text
    assert text.count("<|im_start|>tool\n") == 2 and text.count("<|im_start|>system") == 1


def test_supervision_covers_assistant_turns_only():
    pieces = render_chat(TOOL_CHAT, TOOLS)
    learned = [t for t, s in pieces if s]
    assert len(learned) == 2 and all(t.endswith("<|im_end|>\n") for t in learned)
    assert learned[0].startswith("<tool_call>") and learned[1] == "北京 12°C，上海 18°C。<|im_end|>\n"
    context = "".join(t for t, s in pieces if not s)
    assert "<tools>" in context and "<|im_start|>tool\n{\"temp\": 12}<|im_end|>" in context
    assert "<|im_start|>assistant\n" in context            # role header is the prompt, not a target
    with pytest.raises(ValueError):
        render_chat([{"role": "robot", "content": "x"}])


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from chronos.tokenizer import build_tokenizer
    import json
    root = tmp_path_factory.mktemp("chat_tok")
    with open(root / "d.jsonl", "w", encoding="utf-8") as fh:
        for i in range(200):
            fh.write(json.dumps({"text": f"天气 weather 北京 上海 call {i} tool_call get_weather city"},
                                ensure_ascii=False) + "\n")
    return build_tokenizer([str(root / "d.jsonl")], str(root / "out"), vocab_size=700, log=lambda *_: None)


def test_encode_matches_full_text_and_masks_exactly(tok):
    ids, mask = encode_chat(tok, TOOL_CHAT, TOOLS)
    text = "".join(t for t, _ in render_chat(TOOL_CHAT, TOOLS))
    assert tok.decode(ids.tolist()) == text                       # pieces join to the template text
    learned = tok.decode(ids[mask == 1].tolist())
    assert learned.startswith("<tool_call>") and learned.endswith("。<|im_end|>\n")
    assert "<|im_start|>" not in learned and "temp" not in learned
    im_end = tok.token_id("<|im_end|>")
    assert mask[-1] == 1 and int((ids[mask == 1] == im_end).sum()) == 2
    # Trailing context (a tool result with no reply) is dropped from the record.
    ids2, mask2 = encode_chat(tok, TOOL_CHAT[:4], TOOLS)
    assert mask2[-1] == 1 and tok.decode(ids2.tolist()).endswith("</tool_call><|im_end|>\n")
    with pytest.raises(ValueError):
        encode_chat(tok, TOOL_CHAT[:2], TOOLS)


def test_lmask_shards_batch_and_router_mask(tok, tmp_path):
    import importlib.util
    import torch
    from chronos.data.shards import SFTRecords
    from chronos.trainer.loss_mixin import real_token_mask_torch
    token_mask_for = None
    if importlib.util.find_spec("mlx") is not None:
        from chronos.mlx.training.longrun import token_mask_for
    recs = [encode_chat(tok, TOOL_CHAT, TOOLS), encode_chat(tok, PLAIN, None), encode_chat(tok, TOOL_CHAT[1:4], TOOLS)]
    stats = write_sft_records(str(tmp_path / "sft" / "part0"), recs, max_len=10_000)
    assert stats["records"] == 3 and stats["supervised_tokens"] == sum(int(m.sum()) for _, m in recs)
    folder = str(tmp_path / "sft")
    assert sorted(os.listdir(folder)) == ["part0.bin", "part0.idx", "part0.lmask", "part0.plen"]
    plen = np.fromfile(os.path.join(folder, "part0.plen"), np.int32)
    assert plen.tolist() == [int(np.flatnonzero(m)[0]) for _, m in recs]
    sft = SFTRecords(folder, seed=0)
    assert sft.supervised_tokens() == stats["supervised_tokens"]
    by_len = {len(i): (i, m) for i, m in recs}
    seen = set()
    for _ in range(30):
        ids, labels = sft.batch(token_budget=1 << 16, width_multiple=16)
        for r in range(ids.shape[0]):
            n = int((ids[r] != 0).sum())
            ref_ids, ref_mask = by_len[n]
            np.testing.assert_array_equal(ids[r, :n], ref_ids)
            np.testing.assert_array_equal(labels[r, :n], np.where(ref_mask == 1, ref_ids, -100))
            assert (labels[r, n:] == -100).all()
            seen.add(n)
        # The router's real-token mask covers every record token (records end supervised).
        np.testing.assert_array_equal(real_token_mask_torch(torch.as_tensor(labels)).numpy(), ids != 0)
        if token_mask_for is not None:                     # MLX SFTSource shift: labels[t] = token t+1
            mask = token_mask_for(labels[:, 1:])
            real = ids[:, :-1] != 0
            np.testing.assert_array_equal(real if mask is None else mask, real)
    assert len(seen) >= 2


def test_lmask_size_mismatch_and_bad_records(tok, tmp_path):
    from chronos.data.shards import SFTRecords
    ids, mask = encode_chat(tok, PLAIN, None)
    write_sft_records(str(tmp_path / "s" / "p"), [(ids, mask)])
    with open(tmp_path / "s" / "p.lmask", "ab") as fh:
        fh.write(b"\x01")
    with pytest.raises(ValueError, match="lmask"):
        SFTRecords(str(tmp_path / "s"))
    with pytest.raises(ValueError):
        write_sft_records(str(tmp_path / "t" / "p"), [(ids, np.zeros_like(mask))])
    stats = write_sft_records(str(tmp_path / "u" / "p"), [(ids, mask)], max_len=len(ids) - 1)
    assert stats["records"] == 0 and stats["skipped_too_long"] == 1


def test_server_messages_keep_tool_calls_and_template_renders_them():
    from chronos.serving.ollama_server import _chat_messages
    body = {"messages": [
        {"role": "user", "content": "北京天气？"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}]},
        {"role": "tool", "content": "{\"temp\": 12}"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]}]}
    msgs = _chat_messages(body)
    assert msgs[1]["tool_calls"] == [{"function": {"name": "get_weather", "arguments": {"city": "北京"}}}]
    assert msgs[3]["content"] == "" and "tool_calls" not in msgs[0] and "tool_calls" not in msgs[2]
    served = _jinja(CHAT_TEMPLATE, msgs, TOOLS, add_generation_prompt=True)
    trained = "".join(t for t, _ in render_chat(msgs, TOOLS, add_generation_prompt=True))
    assert served == trained and '<tool_call>{"name": "get_weather", "arguments": {"city": "北京"}}</tool_call>' in served
