"""Chronos tokenizer builder: round-trip, HF parity, determinism, balance."""
from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.tokenizer import build_tokenizer
from chronos.tokenizer.model import ChronosTokenizer, tokenizer_normalize
from chronos.tokenizer.normalize import looks_word_segmented_cjk, normalize_text
from chronos.tokenizer.trainer import BPETrainerConfig, train_bpe

ZH = ["今天天气很好，我们去公园散步吧。", "经济增长放缓，央行维持利率不变。", "他说：“我们明天再讨论这个问题。”",
      "人工智能正在改变世界。", "请把这份文件翻译成英文。"] * 40
EN = ["The weather is nice today, let's take a walk in the park.",
      "Economic growth slowed and the central bank kept rates unchanged.",
      "He said: \"We will discuss this question again tomorrow.\"",
      "Artificial intelligence is changing the world.", "Please translate this document into English."] * 40
EDGE = ["Hello  world\t🙂 2026年11月5日", "<|im_start|>user\n翻译：naïve café ①②\n<|im_end|>\n",
        "  leading and trailing  ", "a" * 80, "ǅ𝔘𐍈 ünïcödé", "", "\n\n", "1,000.50 USD — 100%"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("tok_data")
    path = root / "pairs.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for zh, en in zip(ZH, EN):
            fh.write(json.dumps({"translation": {"zh": zh, "en": en}}, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def tokenizer(corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("tok_out")
    return build_tokenizer([corpus], str(out), vocab_size=900, log=lambda *_: None), str(out)


def test_roundtrip_is_lossless(tokenizer):
    tok, _ = tokenizer
    for text in ZH[:5] + EN[:5] + EDGE:
        assert tok.decode(tok.encode(text)) == tokenizer_normalize(text)


def test_special_tokens_are_atomic(tokenizer):
    tok, _ = tokenizer
    ids = tok.encode("<|im_start|>user\nhi<|im_end|>")
    assert ids[0] == tok.token_id("<|im_start|>") and ids[-1] == tok.token_id("<|im_end|>")
    assert tok.token_id("<|endoftext|>") == 0 and tok.token_id("<|im_start|>") == 1 and tok.token_id("<|im_end|>") == 2
    assert tok.encode("<|im_end|>", allow_special=False) != [tok.token_id("<|im_end|>")]


def test_hf_tokenizer_json_matches_native_encoder(tokenizer):
    tokenizers = pytest.importorskip("tokenizers")
    tok, out = tokenizer
    hf = tokenizers.Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    for text in ZH[:5] + EN[:5] + EDGE:
        assert hf.encode(text).ids == tok.encode(text), text
        assert hf.decode(tok.encode(text), skip_special_tokens=False) == tokenizer_normalize(text)


def test_transformers_autotokenizer_loads_and_applies_chat_template(tokenizer):
    transformers = pytest.importorskip("transformers")
    tok, out = tokenizer
    auto = transformers.AutoTokenizer.from_pretrained(out)
    text = auto.apply_chat_template([{"role": "user", "content": "翻译成中文：Hello"}],
                                    tokenize=False, add_generation_prompt=True)
    assert text.startswith("<|im_start|>user\n") and text.endswith("<|im_start|>assistant\n")
    assert auto(text, add_special_tokens=False)["input_ids"] == tok.encode(text)


def test_training_is_deterministic_and_reload_identical(tokenizer, corpus, tmp_path):
    tok, out = tokenizer
    again = build_tokenizer([corpus], str(tmp_path), vocab_size=900, log=lambda *_: None)
    assert again.fingerprint() == tok.fingerprint()
    assert ChronosTokenizer.load(out).fingerprint() == tok.fingerprint()


def test_language_balance_gives_both_scripts_merges():
    trained = train_bpe({"zh": ZH, "en": EN * 20}, BPETrainerConfig(vocab_size=700))
    merged = [a + b for a, b in trained.merges]
    assert any(any("一" <= ch <= "鿿" for ch in m) and len(m) >= 2 for m in merged)
    assert any(m.isascii() and m.strip("▁").isalpha() and len(m) >= 3 for m in merged)
    assert trained.stats["bucket_weights"]["zh"] > trained.stats["bucket_weights"]["en"]


def test_segmented_chinese_detection_and_unsegmentation():
    segmented = ["表演 的 明星 是 X 女孩 团队 , 其中 有些 人 受过 训练 。"] * 5
    assert looks_word_segmented_cjk(segmented)
    assert not looks_word_segmented_cjk(ZH[:5])
    assert normalize_text(segmented[0], unsegment_cjk=True) == "表演的明星是X女孩团队,其中有些人受过训练。"


def test_stray_script_bucket_cannot_dominate_merges():
    trained = train_bpe({"zh": ZH, "latin": EN, "ja": ["カタカナ"]}, BPETrainerConfig(vocab_size=700))
    weights = trained.stats["bucket_weights"]
    assert "ja" not in weights and max(weights.values()) <= 4.0
