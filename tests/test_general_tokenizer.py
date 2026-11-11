"""Bundled general 32k tokenizer: loads, task-neutral control ids, HF == native, planner default for new models."""
from __future__ import annotations

import json
import os

import pytest

from chronos.deps import GENERAL_TOKENIZER_DIR, GENERAL_TOKENIZER_VOCAB, get_general_tokenizer_path, get_tokenizer_path
from chronos.model.planner import PlanRequest, plan_training
from chronos.tokenizer.model import ChronosTokenizer, special_tokens_with

SAMPLES = ["Hello, world!", "研究人员发现了一种新的抗生素。", "x = f(y) + 3.14", "def f(a, b):\n    return a + b\n",
    "The quick brown fox jumps over the lazy dog.", "<|im_start|>user\nhi<|im_end|>\n"]


def test_asset_files_are_present_and_licensed():
    for name in ("chronos_tokenizer.json", "tokenizer.json", "tokenizer_config.json", "tokenizer_report.json", "NOTICE", "LICENSE"):
        assert (GENERAL_TOKENIZER_DIR / name).is_file(), name
    assert get_general_tokenizer_path() == str(GENERAL_TOKENIZER_DIR)


def test_vocab_size_and_task_neutral_control_ids():
    tok = ChronosTokenizer.load(str(GENERAL_TOKENIZER_DIR))
    assert len(tok.vocab) == GENERAL_TOKENIZER_VOCAB == 32768
    specials = special_tokens_with(slots=64)
    for i, name in enumerate(specials):
        assert tok.vocab[name] == i, name
    assert [tok.vocab[t] for t in ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>")] == [0, 1, 2, 3]
    assert not [t for t in ("<|en|>", "<|zh|>", "<|translate|>") if t in tok.vocab]


def test_hf_and_native_encoders_agree_and_round_trip():
    from tokenizers import Tokenizer
    tok = ChronosTokenizer.load(str(GENERAL_TOKENIZER_DIR))
    hf = Tokenizer.from_file(str(GENERAL_TOKENIZER_DIR / "tokenizer.json"))
    for text in SAMPLES:
        ids = tok.encode(text)
        assert hf.encode(text).ids == ids, text
        assert tok.decode(ids) == text, text


def test_report_names_no_local_path_and_records_derivation():
    report = json.load(open(GENERAL_TOKENIZER_DIR / "tokenizer_report.json", encoding="utf-8"))
    assert "/Users/" not in json.dumps(report) and "/Volumes/" not in json.dumps(report)
    assert report["vocab_size"] == 32768 and "derived_from" in report


def test_planner_defaults_new_models_to_the_bundled_tokenizer(tmp_path):
    plan = plan_training(PlanRequest(ram_gb=32, target_total=30e6, num_experts=8, active_experts=2, seq_len=512))
    assert plan["config"]["vocab_size"] == 32768
    assert plan["config"]["tokenizer_path"] == get_general_tokenizer_path()
    assert "bundled general 32k" in plan["report"]["sources"]["vocab_size"]
    assert "bundled general 32k" in plan["report"]["sources"]["tokenizer_path"]


def test_explicit_vocab_keeps_the_user_choice_and_adds_no_tokenizer_path():
    plan = plan_training(PlanRequest(ram_gb=32, target_total=30e6, num_experts=8, active_experts=2, vocab_size=6400, seq_len=512))
    assert plan["config"]["vocab_size"] == 6400
    assert "tokenizer_path" not in plan["config"]


def test_checkpoints_without_a_recorded_tokenizer_keep_the_legacy_default(monkeypatch):
    monkeypatch.delenv("CHRONOS_DEFAULT_TOKENIZER", raising=False)
    assert os.path.basename(get_tokenizer_path()) == "legacy_6400"
