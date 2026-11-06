"""Task-agnostic UI (monitor evals, data inspector) and the Inference-tab
self-speculative controls, including exactness on the real runner paths."""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---- evaluation discovery ------------------------------------------------------
def test_split_key_groups_by_metric_name_not_task():
    from chronos.monitor.evals import split_key
    assert split_key("probe_en_zh_bleu") == ("bleu", "probe en_zh")
    assert split_key("probe_zh_en_bleu") == ("bleu", "probe zh_en")
    assert split_key("qa_accuracy") == ("accuracy", "qa")
    assert split_key("bleu_zh") == ("bleu", "bleu_zh")
    assert split_key("accuracy") == ("accuracy", "accuracy")
    assert split_key("pass1") == ("pass1", "pass1")
    assert split_key("weird_thing") == ("weird_thing", "weird_thing")


def test_eval_metrics_discovers_any_task_and_separates_context():
    from chronos.monitor.evals import eval_metrics
    events = [
        {"event": "validation", "phase": "p", "step": 1, "valid_ce": 3.0, "qa_accuracy": 0.4,
         "code_pass1": 0.1, "eval_n": 50, "eval_s": 2.0, "flag": True},
        {"event": "validation", "phase": "p", "step": 2, "valid_ce": 2.8, "qa_accuracy": 0.5,
         "math_accuracy": 0.2, "mystery": 7.0},
        {"event": "final", "phase": "p", "step": 3, "hours": 1.0, "chrf": 40.0},
        {"event": "rollback", "phase": "p", "step": 2, "lr_scale": 0.5},       # not an eval event
    ]
    ev = eval_metrics(events)
    fam = {f["metric"]: f for f in ev["families"]}
    assert set(fam) == {"accuracy", "pass1", "mystery", "chrf"}
    assert [s["label"] for s in fam["accuracy"]["series"]] == ["qa", "math"]
    assert fam["accuracy"]["direction"] == "higher" and fam["mystery"]["direction"] is None
    assert "valid_ce" in ev["keys"] and "lr_scale" not in ev["keys"] and "flag" not in ev["keys"]
    assert set(ev["context"]) == {"eval_n", "eval_s", "hours"}


def test_sample_view_maps_roles_for_any_task():
    from chronos.monitor.evals import sample_view
    mt = sample_view({"time": "t", "phase": "pretrain", "step": 5, "direction": "en_zh", "set": "dev",
                      "src": "Hello", "ref": "你好", "hyp": "您好"})
    assert (mt["input"], mt["reference"], mt["output"]) == ("Hello", "你好", "您好")
    assert mt["tags"] == {"direction": "en_zh", "set": "dev"}
    qa = sample_view({"phase": "sft", "step": 9, "question": "2+2?", "answer": "4", "prediction": "4",
                      "task": "math", "score": 1.0, "meta": {"x": 1}})
    assert (qa["input"], qa["reference"], qa["output"]) == ("2+2?", "4", "4")
    assert qa["tags"] == {"task": "math", "score": 1.0}


def _run_with_generic_evals(root):
    path = os.path.join(root, "chat")
    os.makedirs(os.path.join(path, "inspections"))
    with open(os.path.join(path, "run_spec.json"), "w", encoding="utf-8") as fh:
        json.dump({"budget_hours": 1.0, "phases": [{"name": "sft", "budget_fraction": 1.0}],
                   "task": "chat", "seed": 3, "model_config": {"hidden_size": 8}}, fh)
    rows = [{"phase": "sft", "step": i, "total": 10, "loss": 2.0, "time": f"2026-11-06T00:00:0{i}"} for i in range(1, 4)]
    rows += [{"phase": "sft", "event": "validation", "step": 2, "valid_ce": 1.9, "helpful_score": 0.6,
              "safety_accuracy": 0.9, "judge_n": 32}]
    with open(os.path.join(path, "metrics.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(path, "samples.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"phase": "sft", "step": 2, "prompt": "Hi", "response": "Hello!", "task": "chat"}) + "\n")
    with open(os.path.join(path, "inspections", "sft_s2.json"), "w", encoding="utf-8") as fh:
        json.dump({"phase": "sft", "step": 2, "qa": {"accuracy": 0.7, "n": 100}, "run_dir": "/x"}, fh)
    return path


def test_snapshot_exposes_generic_evals_samples_and_inspections(tmp_path):
    from chronos.monitor import RunMonitor
    _run_with_generic_evals(str(tmp_path))
    snap = RunMonitor(str(tmp_path)).snapshot("chat")
    fams = {f["metric"]: f for f in snap["evals"]["families"]}
    assert set(fams) == {"score", "accuracy"} and snap["evals"]["context"] == ["judge_n"]
    assert snap["samples"][0]["input"] == "Hi" and snap["samples"][0]["output"] == "Hello!"
    assert snap["samples"][0]["tags"] == {"task": "chat"}
    insp = snap["inspections"][0]
    assert insp["metrics"] == {"qa.accuracy": 0.7} and insp["step"] == 2


def test_gradio_monitor_eval_dropdown_follows_discovered_metrics(tmp_path):
    pytest.importorskip("gradio")
    from ui.tabs.monitor_tab import refresh
    _run_with_generic_evals(str(tmp_path))
    out = refresh(str(tmp_path), "chat", "sft", "primary_loss", "accuracy")
    assert len(out) == 31
    eval_update, eval_plot, samples_html = out[-7], out[-6], out[-4]
    assert set(eval_update["choices"]) == {"score", "accuracy"} and eval_update["value"] == "accuracy"
    values = eval_plot["value"]
    assert list(values["series"]) == ["safety"] and list(values["value"]) == [0.9]
    assert "Hello!" in samples_html and "<script" not in samples_html
    fallback = refresh(str(tmp_path), "chat", "sft", "primary_loss", "no-such-metric")
    assert fallback[-7]["value"] in {"score", "accuracy"}


def test_monitor_page_hardcodes_no_task():
    with open(os.path.join(_repo_root, "chronos", "monitor", "static", "app.js"), encoding="utf-8") as fh:
        js = fh.read()
    for needle in ("probe_en_zh", "probe_zh_en", "bleu_zh", "en→zh", "flores", "FLORES"):
        assert needle not in js, needle
    with open(os.path.join(_repo_root, "ui", "tabs", "monitor_tab.py"), encoding="utf-8") as fh:
        tab = fh.read()
    for needle in ("probe_en_zh", "bleu_zh", "en→zh", "FLORES"):
        assert needle not in tab, needle


# ---- data inspector -------------------------------------------------------------
def test_record_schema_covers_common_dataset_shapes():
    from ui.tabs.data_tab import record_schema
    assert record_schema({"messages": [{"role": "user", "content": "hi"}]}) == "chat"
    assert record_schema({"conversations": [{"from": "human", "value": "hi"}]}) == "chat"
    assert record_schema({"prompt": "p", "chosen": "a", "rejected": "b"}) == "preference"
    assert record_schema({"instruction": "do", "input": "", "output": "done"}) == "instruction"
    assert record_schema({"question": "q", "answer": "a"}) == "prompt_response"
    assert record_schema({"translation": {"en": "hi", "de": "hallo"}}) == "parallel_pair"
    assert record_schema({"en": "hi", "zh": "你好"}) == "parallel_pair"
    assert record_schema({"text": "a document"}) == "plain_text"
    assert record_schema("raw line") == "plain_text"
    assert record_schema({"id": 3}) == "other"


def test_data_scan_mixed_formats_in_a_directory(tmp_path):
    from ui.tabs.data_tab import _scan
    d = tmp_path / "mix"
    d.mkdir()
    with open(d / "chat.jsonl", "w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"messages": [{"role": "user", "content": f"question {i}"},
                                              {"role": "assistant", "content": "答案"}]}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"messages": [{"role": "user", "content": "question 0"},
                                          {"role": "assistant", "content": "答案"}]}, ensure_ascii=False) + "\n")
    with open(d / "qa.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["question", "answer"])
        w.writeheader()
        w.writerow({"question": "What is 2+2?", "answer": "4"})
    (d / "notes.txt").write_text("free text line one\nfree text line two\n", encoding="utf-8")
    summary, lens, previews = _scan(str(d))
    assert "error" not in summary, summary
    assert summary["files"] == 3 and summary["scanned"] == 7
    assert summary["schemas"] == {"chat": 4, "plain_text": 2, "prompt_response": 1}
    assert summary["dup_ratio"] == round(1 / 7, 4)
    assert "messages" in summary["fields"] and "question" in summary["fields"]
    assert len(lens) == 7 and summary["char_max"] >= summary["char_median"] >= summary["char_min"] > 0
    assert previews and set(summary["schema_examples"]) == set(summary["schemas"])
    missing, _, _ = _scan(str(tmp_path / "nope"))
    assert "error" in missing


# ---- Inference tab: self-speculative controls -----------------------------------
def test_speculative_options_and_row_fields():
    from ui.tabs.inference_tab import _speculative_options, _speculative_row_fields
    assert _speculative_options(False, 2, 4) is None
    assert _speculative_options(True, "2", 12) == {"draft_depth": 2, "gamma": 8, "gamma_max": 8, "adaptive_gamma": True}
    assert _speculative_options(True, None, "x") == {"draft_depth": 1, "gamma": 4, "gamma_max": 8, "adaptive_gamma": True}
    assert _speculative_row_fields({}) == {}
    row = _speculative_row_fields({"speculative": {"enabled": True, "draft_depth": 1, "full_depth": 4,
                                                   "acceptance_rate": 0.5, "tokens_per_verify": 2.5,
                                                   "verify_calls": 4, "draft_calls": 12, "gamma_final": 5}})
    assert row["speculative"] is True and row["spec_full_depth"] == 4 and row["spec_acceptance_rate"] == 0.5


def test_speculative_supported_matches_runtime_fallback():
    from chronos.model.config import ChronosConfig
    from chronos.runtime.speculative import speculative_supported
    base = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
                vocab_size=64, num_experts=2, num_experts_per_tok=1, prelude_layers=1, coda_layers=1)
    assert speculative_supported(ChronosConfig(**base, recurrent_depth=4))
    assert not speculative_supported(ChronosConfig(**base, recurrent_depth=1))
    assert not speculative_supported(ChronosConfig(**base, recurrent_depth=4, recurrent_attention_mode="per_iteration"))


def test_inference_tab_speculative_i18n_present():
    from ui.i18n import LANGUAGE_CHOICES, TRANSLATIONS
    for _, code in LANGUAGE_CHOICES:
        for key in ("infer.speculative", "infer.draft_depth", "infer.spec_gamma", "infer.spec_note",
                    "monitor.eval_metric", "monitor.eval_note", "monitor.sample_input",
                    "monitor.sample_ref", "monitor.sample_output"):
            assert TRANSLATIONS[code].get(key), (code, key)


# ---- speculative on the real Inference-tab runner paths ----------------------------
def _tiny_checkpoint(tmp_path, fallback_mask_prob=0.0):
    import torch
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    torch.manual_seed(5)
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        vocab_size=96, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                        moe_intermediate_size=64, max_position_embeddings=128, recurrent_depth=4,
                        prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=fallback_mask_prob,
                        kv_latent_dim=16, rope_dim=8, sliding_window_size=64, use_moe=True)
    model = ChronosForCausalLM(cfg).eval()
    path = str(tmp_path / "tiny.pth")
    torch.save(model.state_dict(), path)
    return cfg, path


def _clone(cfg):
    from ui.tabs.inference_tab import _clone_model_cfg
    return _clone_model_cfg(cfg)


PROMPT = [5, 17, 33, 2, 61, 9, 40]
OPTS = {"draft_depth": 1, "gamma": 4, "gamma_max": 8, "adaptive_gamma": True}


def test_torch_full_dram_speculative_matches_plain_greedy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from ui.tabs.inference_tab import _run_torch_full_dram
    cfg, path = _tiny_checkpoint(tmp_path)
    mask = [1] * len(PROMPT)
    plain, _ = _run_torch_full_dram("cpu", _clone(cfg), path, PROMPT, mask, 20, 1.0, None, do_sample=False)
    spec, stats = _run_torch_full_dram("cpu", _clone(cfg), path, PROMPT, mask, 20, 1.0, None, do_sample=False,
                                       speculative=dict(OPTS))
    assert spec == plain and len(spec) == 20
    s = stats["speculative"]
    assert s["enabled"] and s["full_depth"] == 4 and s["draft_depth"] == 1
    assert s["verify_calls"] < 20 and stats["tokens_generated"] == 20


def test_torch_lazy_experts_speculative_matches_plain_greedy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from ui.tabs.inference_tab import _run_torch_full_dram, _run_torch_inference
    cfg, path = _tiny_checkpoint(tmp_path)
    mask = [1] * len(PROMPT)
    plain, _ = _run_torch_full_dram("cpu", _clone(cfg), path, PROMPT, mask, 16, 1.0, None, do_sample=False)
    # Half the experts resident; sync_on_demand keeps lazy output exact.
    spec, stats = _run_torch_inference("cpu", _clone(cfg), path, PROMPT, mask, 16, 1.0, None, do_sample=False,
                                       ram_load_ratio=0.5, miss_policy="sync_on_demand", speculative=dict(OPTS))
    assert spec == plain
    s = stats["speculative"]
    assert s["enabled"] and s["verify_calls"] < 16
    assert stats["miss_policy"] == "sync_on_demand" and stats["forwards"] == s["verify_calls"] + s["draft_calls"] + 1


def test_torch_speculative_per_iteration_model_falls_back_to_plain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import torch
    from chronos.model.model_chronos import ChronosForCausalLM
    from ui.tabs.inference_tab import _run_torch_full_dram
    cfg, _ = _tiny_checkpoint(tmp_path)
    cfg.recurrent_attention_mode = "per_iteration"
    torch.manual_seed(5)
    path = str(tmp_path / "per_iter.pth")
    torch.save(ChronosForCausalLM(_clone(cfg)).state_dict(), path)
    mask = [1] * len(PROMPT)
    plain, _ = _run_torch_full_dram("cpu", _clone(cfg), path, PROMPT, mask, 8, 1.0, None, do_sample=False)
    spec, stats = _run_torch_full_dram("cpu", _clone(cfg), path, PROMPT, mask, 8, 1.0, None, do_sample=False,
                                       speculative=dict(OPTS))
    assert spec == plain and stats["speculative"]["enabled"] is False
    assert stats["speculative"]["draft_calls"] == 0


def _mlx_cpu():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    return mx, previous


def test_mlx_full_and_lazy_runners_speculative_match_plain_greedy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mx, previous = _mlx_cpu()
    try:
        from ui.tabs.inference_tab import _run_mlx_full_dram, _run_mlx_lazy_inference
        cfg, path = _tiny_checkpoint(tmp_path)
        plain, _ = _run_mlx_full_dram(_clone(cfg), path, PROMPT, 16, 0.0, do_sample=False)
        spec, stats = _run_mlx_full_dram(_clone(cfg), path, PROMPT, 16, 0.0, do_sample=False,
                                         speculative=dict(OPTS))
        assert spec == plain and len(spec) == 16
        assert stats["speculative"]["enabled"] and stats["speculative"]["verify_calls"] < 16
        lazy, lstats = _run_mlx_lazy_inference(_clone(cfg), path, PROMPT, 16, 0.0, do_sample=False,
                                               ram_load_ratio=0.5, speculative=dict(OPTS))
        assert lazy == plain
        assert lstats["speculative"]["enabled"]
    finally:
        mx.set_default_device(previous)


def test_inference_tab_uses_the_checkpoint_tokenizer(tmp_path):
    from ui.tabs.inference_tab import _inference_tokenizer_dir
    ckpt_dir = tmp_path / "bundle" / "weights"
    (ckpt_dir / "tokenizer").mkdir(parents=True)
    (ckpt_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    ckpt = ckpt_dir / "model.pth"
    ckpt.write_bytes(b"")
    assert _inference_tokenizer_dir({}, str(ckpt)) == str(ckpt_dir / "tokenizer")
    other = tmp_path / "explicit"
    other.mkdir()
    (other / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert _inference_tokenizer_dir({"tokenizer_path": str(other)}, str(ckpt)) == str(other)
    import chronos.deps
    assert _inference_tokenizer_dir({}, "") == chronos.deps.get_tokenizer_path()
    with pytest.raises(FileNotFoundError):
        _inference_tokenizer_dir({"tokenizer_path": str(tmp_path / "empty")}, str(ckpt))


def test_mlx_runners_decode_in_eval_mode(tmp_path, monkeypatch):
    """Training-mode MLX randomly masks experts (fallback_mask_prob): repeated
    greedy runs and the speculative path then disagree.  Regression for the
    Inference tab converting a fresh (training-mode) torch model."""
    monkeypatch.chdir(tmp_path)
    mx, previous = _mlx_cpu()
    try:
        from ui.tabs.inference_tab import _run_mlx_full_dram
        cfg, path = _tiny_checkpoint(tmp_path, fallback_mask_prob=0.5)
        runs = [_run_mlx_full_dram(_clone(cfg), path, PROMPT, 16, 0.0, do_sample=False)[0] for _ in range(3)]
        spec, _ = _run_mlx_full_dram(_clone(cfg), path, PROMPT, 16, 0.0, do_sample=False, speculative=dict(OPTS))
        assert runs[0] == runs[1] == runs[2] == spec
        from chronos.mlx.inference import ChronosMLXInferenceEngine
        from chronos.mlx.model import ChronosMLXModel
        model = ChronosMLXModel(cfg)
        assert model.training
        engine = ChronosMLXInferenceEngine(model, cfg, ssd_dir=str(tmp_path / "e"))
        assert not model.training
        engine.stop()
    finally:
        mx.set_default_device(previous)


def test_torch_lazy_engine_mixes_speculative_and_plain_calls(tmp_path):
    """One persistent lazy engine (``chronos serve``) alternates speculative and
    plain requests; experts paged in by one must stay usable by the other."""
    import torch
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.runtime.inference_engine import ChronosInferenceEngine
    from ui.tabs.inference_tab import _apply_offload_expert_budget, _replace_torch_experts_with_placeholders
    cfg, path = _tiny_checkpoint(tmp_path)
    weights = torch.load(path)
    model = ChronosForCausalLM(cfg)        # training mode on purpose: the engine must not depend on it
    model.load_state_dict({k: v for k, v in weights.items() if ".mlp.experts." not in k}, strict=False)
    _replace_torch_experts_with_placeholders(model, cfg)
    model.eval()
    engine = ChronosInferenceEngine(model, cfg, ssd_dir=str(tmp_path / "cache"))
    _apply_offload_expert_budget(engine, cfg, 0.5)
    engine.setup_from_state_dict(weights, warm_expert_ids=[])
    ids = torch.tensor([PROMPT])
    try:
        outs = []
        for spec in (True, False, True, False):
            if spec:
                out = engine.generate_speculative(ids, max_new_tokens=12, miss_policy="sync_on_demand", **OPTS)
            else:
                out = engine.generate(ids, max_new_tokens=12, temperature=1.0, top_k=0, top_p=1.0, do_sample=False,
                                      eos_token_id=None, miss_policy="sync_on_demand")
            outs.append(out[0].tolist())
        assert outs[0] == outs[1] == outs[2] == outs[3]
    finally:
        engine.teardown()
