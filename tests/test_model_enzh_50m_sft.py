"""chronos-enzh-50m-sft driver: input boundary, sentence split, loop guard (no weights needed)."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "chronos-enzh-50m-sft")


def _driver():
    spec = importlib.util.spec_from_file_location("chronos_enzh_translate", os.path.join(ROOT, "translate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _driver()


@pytest.mark.parametrize("text", ["我今天很高兴。", "我喜欢Python编程", "今日はいい天気です", "Привет, как дела?", "12345", ""])
def test_non_english_input_is_refused(text):
    with pytest.raises(T.BoundaryError):
        T.check_english(text)


@pytest.mark.parametrize("text", ["Hello.", "I visited 北京 last year.", "2. Use template with slots"])
def test_english_input_passes(text):
    T.check_english(text)


def test_sentence_split_keeps_abbreviations_and_list_numbers():
    text = "Dr. Smith arrived. He was late! Was it the U.S. team? Yes.\n2. HTML for section 7"
    assert T.split_sentences(text) == ["Dr. Smith arrived.", "He was late!", "Was it the U.S. team?", "Yes.",
        "2. HTML for section 7"]


def test_loop_guard():
    assert T.loop_start([1, 2, 3]) == -1
    assert T.loop_start([5] * 7) == -1    # seven equal digits are not a loop
    assert T.loop_start([5] * 8) == 1
    assert T.loop_start([9, 1, 2] * 4) == 3


def test_zh_en_direction_is_refused(capsys):
    assert T.main(["--direction", "zh_en", "你好"]) == 2
    assert "zh_en is disabled" in capsys.readouterr().err


def test_shipped_settings_agree():
    prompts = json.load(open(os.path.join(ROOT, "prompts.json"), encoding="utf-8"))
    generation = json.load(open(os.path.join(ROOT, "generation_config.json"), encoding="utf-8"))
    assert prompts["supported_directions"] == ["en_zh"]
    assert prompts["template"].count("{source}") == 1 and prompts["template"].endswith("<|im_start|>assistant\n")
    assert generation["do_sample"] is False and generation["max_source_tokens"] == 128
    config = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
    assert config["vocab_size"] == 16384 and not any("/Users/" in str(v) for v in config.values())
