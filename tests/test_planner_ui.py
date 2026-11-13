"""Adaptive training planner, v3 presets, per-stage overrides, and the
tokenizer / serve / planner UI helpers."""
import json
import os

import pytest

from chronos.model.planner import (
    MEMORY_LEVELS, STAGE_KEYS, STAGES, PlanRequest, overrides_from_table, plan_training, profile_datasets,
    search_architecture, stage_table,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
ARCH_50M = os.path.join(FIXTURES, "arch_50m.json")   # 50M shape + meta-device counts


def _plan(**kw):
    kw.setdefault("ram_gb", 32)
    return plan_training(PlanRequest(**kw))


# ── architecture search ────────────────────────────────────────────────
def test_search_reproduces_the_50m_shape_by_experts_and_by_params():
    recipe = json.load(open(ARCH_50M))["params"]
    for kw in (dict(active_experts=2), dict(active_params=12.09e6)):
        arch = search_architecture(50e6, use_moe=True, num_experts=36, vocab_size=16384, depth=4, **kw)
        assert (arch.hidden_size, arch.num_hidden_layers, arch.ffn_width, arch.num_experts_per_tok) == (384, 6, 160, 2)
        assert (arch.total, arch.active, arch.union_upper) == (
            recipe["total"], recipe["active"], recipe["active_union_upper"])


def test_closed_form_matches_meta_device_count():
    from chronos.config_schema import model_config_kwargs
    from chronos.model.factory import count_parameters
    for kw in (dict(target_total=200e6, num_experts=32, active_experts=2),
               dict(target_total=30e6, use_moe=False, num_experts=1)):
        plan = _plan(vocab_size=6400, seq_len=512, **kw)
        counted = count_parameters(model_config_kwargs(plan["config"]))
        assert plan["report"]["architecture"]["params_source"].startswith("measured")
        assert counted["total"] == plan["report"]["architecture"]["total_params"]
        assert abs(counted["total"] / kw["target_total"] - 1) < 0.02
        assert not any("differs from the meta-device count" in n for n in plan["report"]["notes"])


def test_dense_plan_has_no_experts_and_a_dense_ffn():
    cfg = _plan(target_total=30e6, use_moe=False, num_experts=1, vocab_size=6400, seq_len=512)["config"]
    assert cfg["use_moe"] is False and cfg["intermediate_size"] > 0 and cfg["moe_intermediate_size"] == 0
    assert cfg["lambda_balance"] == 0.0 and cfg["ram_budget_mb"] is None


def test_gqa_head_count_avoids_single_kv_head():
    for target in (100e6, 200e6, 500e6):
        arch = search_architecture(target, use_moe=True, num_experts=32, vocab_size=6400, depth=4, active_experts=2)
        assert arch.num_attention_heads // arch.num_key_value_heads <= 4


# ── memory levels and stages ───────────────────────────────────────────
def test_memory_levels_are_monotonic():
    plans = [_plan(target_total=200e6, num_experts=32, active_experts=2, vocab_size=6400, seq_len=1024,
                   memory_saving=level) for level in MEMORY_LEVELS]
    peaks = [p["report"]["memory"]["training_peak_gb"] for p in plans]
    budgets = [p["config"]["ram_budget_mb"] for p in plans]
    assert peaks == sorted(peaks, reverse=True) and peaks[0] > peaks[-1]
    assert budgets[0] is None and budgets[1] > budgets[2] > budgets[3] > 0
    assert [p["config"]["kv_cache_dtype"] for p in plans] == ["auto", "int8", "int8", "int8"]
    assert plans[-1]["config"]["optimizer_paging"] is True and plans[0]["config"]["optimizer_paging"] is False


def test_every_stage_has_independent_values_and_v3_kernel():
    plan = _plan(target_total=50e6, num_experts=24, active_experts=2, vocab_size=6400, seq_len=512)
    cfg, stages = plan["config"], plan["stage_overrides"]
    assert set(stages) == set(STAGES) and cfg["stage_overrides"] == stages
    assert cfg["architecture_version"] == 3 and cfg["recurrent_attention_mode"] == "shared_first"
    lrs = {name: stages[name]["learning_rate"] for name in STAGES}
    assert lrs["pretrain"] > lrs["sft"] > lrs["dpo"] and lrs["grpo"] <= 2e-5
    assert stages["grpo"]["beta"] == 0.04 and stages["dpo"]["beta"] == 0.1
    assert stages["distill"]["temperature"] == 4.0 and stages["distill"]["alpha"] == 0.7
    for values in stages.values():
        assert set(values) <= set(STAGE_KEYS)
        assert 10 <= values["save_interval"] <= 100_000 and values["save_interval"] % 10 == 0
    assert cfg["learning_rate"] == stages["pretrain"]["learning_rate"]


def test_stage_table_round_trip_drops_blanks_and_coerces():
    stages = _plan(target_total=20e6, num_experts=8, active_experts=2, vocab_size=6400, seq_len=256)["stage_overrides"]
    headers, rows = stage_table(stages)
    assert headers[0] == "stage" and len(rows) == len(STAGES)
    assert overrides_from_table(rows) == stages
    rows = [[str(v) for v in row] for row in rows]           # Gradio returns strings
    rows[1][1 + STAGE_KEYS.index("learning_rate")] = ""
    back = overrides_from_table(rows)
    assert "learning_rate" not in back["sft"] and isinstance(back["pretrain"]["batch_size"], int)


def test_token_shortfall_falls_back_to_20_tokens_per_param():
    plan = _plan(target_total=200e6, num_experts=32, active_experts=2, data_paths=(FIXTURES,))
    assert plan["report"]["scaling"]["tokens"] == pytest.approx(20 * plan["report"]["architecture"]["total_params"])
    assert any("< 2 per parameter" in n for n in plan["report"]["notes"])


# ── dataset profiling ──────────────────────────────────────────────────
def test_profile_fixtures_classifies_every_stage():
    prof = profile_datasets([FIXTURES])
    assert {os.path.basename(f) for f in prof.kinds["pref"]} == {"tiny_dpo.jsonl"}
    assert {os.path.basename(f) for f in prof.kinds["rl"]} == {"tiny_grpo.jsonl"}
    assert {os.path.basename(f) for f in prof.kinds["sft"]} == {"tiny_sft.jsonl"}
    assert prof.pretrain_tokens > 0 and prof.pretrain_tokens_source == "estimated"
    assert prof.seq_p95_tokens and prof.sft_records > 0 and prof.rl_records > 0


def test_profile_manifest_and_tokenizer(tmp_path):
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer_report.json").write_text(json.dumps({"vocab_size": 8192, "heldout_eval": {
        "zh": {"texts": 500, "chars_per_token": 1.5}, "latin": {"texts": 500, "chars_per_token": 4.0}}}))
    (tok / "tokenizer.json").write_text("{}")
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "shards_manifest.json").write_text(json.dumps({
        "tokenizer_dir": str(tok), "pretrain_train_tokens": 3_000_000_000, "sft_train_records": 400_000,
        "sft_max_len": 256}))
    prof = profile_datasets([str(shards)])
    assert prof.vocab_size == 8192 and prof.pretrain_tokens == 3e9 and prof.pretrain_tokens_source == "measured"
    assert prof.seq_cap_tokens == 256
    plan = plan_training(PlanRequest(target_total=50e6, num_experts=24, active_experts=2, data_paths=(str(shards),),
                                     ram_gb=32))
    assert plan["config"]["vocab_size"] == 8192 and plan["config"]["max_seq_len"] == 256
    assert plan["config"]["tokenizer_path"] == str(tok)
    assert plan["report"]["scaling"]["tokens"] == 3e9


# ── presets ────────────────────────────────────────────────────────────
def test_v3_presets_build_and_fill_every_widget():
    from chronos.config_schema import model_config_kwargs
    from chronos.model.factory import count_parameters
    from ui.presets import (
        CONFIG_INPUT_ORDER, DEFAULT_PRESET, V3_PRESET_META, get_preset, preset_names, values_in_input_order,
    )
    names = preset_names()
    assert DEFAULT_PRESET in names and "Chronos-v2 legacy (768H, 6400 vocab)" in names
    assert sum(n.startswith("Chronos-v3") for n in names) == 5
    for name in names:
        cfg = get_preset(name)
        assert len(values_in_input_order(cfg)) == len(CONFIG_INPUT_ORDER)
        assert cfg["moe_intermediate_size"] % 8 == 0
        if name in V3_PRESET_META:
            assert cfg["architecture_version"] == 3 and cfg["recurrent_attention_mode"] == "shared_first"
            assert set(cfg["stage_overrides"]) == set(STAGES)
            assert count_parameters(model_config_kwargs(cfg))["total"] == V3_PRESET_META[name]["total_params"]
    # Old names keep working; returned dicts are independent copies.
    assert get_preset("Recommended-CN (≈192M)")["hidden_size"] == get_preset(DEFAULT_PRESET)["hidden_size"]
    a = get_preset(DEFAULT_PRESET)
    a["stage_overrides"]["sft"]["learning_rate"] = 1.0
    assert get_preset(DEFAULT_PRESET)["stage_overrides"]["sft"]["learning_rate"] != 1.0


def test_v3_presets_file_matches_the_planner():
    """presets_v3.json must be regenerated (python -m ui.presets) when the planner changes."""
    from ui.presets import V3_PRESETS_PATH, build_v3_presets
    stored = json.load(open(V3_PRESETS_PATH, encoding="utf-8"))
    fresh = build_v3_presets()
    assert set(stored["presets"]) == set(fresh["presets"])
    for name, cfg in fresh["presets"].items():
        assert stored["presets"][name] == cfg, name


# ── Train tab merge ────────────────────────────────────────────────────
def test_train_tab_merges_the_selected_stage_only():
    from ui.presets import DEFAULT_PRESET, get_preset
    from ui.tabs.train_tab import CHECKPOINT_TOPOLOGY_KEYS, _prepare_training_run_config
    cfg = get_preset(DEFAULT_PRESET)
    cfg["stage_overrides"]["sft"]["learning_rate"] = 1.25e-4
    del cfg["stage_overrides"]["grpo"]["learning_rate"]
    sft, eff = _prepare_training_run_config(cfg, "sft", "cpu", "", None, "", 0, 0.0, "", "")
    assert sft["learning_rate"] == 1.25e-4 and "learning_rate" in eff["stage_overrides_applied"]
    grpo, _ = _prepare_training_run_config(cfg, "grpo", "cpu", "", None, "", 0, 0.0, "", "")
    assert grpo["learning_rate"] == cfg["learning_rate"] and grpo["beta"] == 0.04
    pretrain, _ = _prepare_training_run_config(cfg, "pretrain", "cpu", "", None, "", 0, 0.0, "", "")
    assert pretrain["hidden_size"] == cfg["hidden_size"]          # architecture never stage-specific
    for key in ("recurrent_attention_mode", "rope_layout", "sliding_window_size", "router_scope"):
        assert key in CHECKPOINT_TOPOLOGY_KEYS


def test_train_tokenizer_path_resolution(tmp_path):
    from ui.tabs.train_tab import resolve_tokenizer_dir
    with pytest.raises(FileNotFoundError):
        resolve_tokenizer_dir(str(tmp_path))
    (tmp_path / "tokenizer.json").write_text("{}")
    assert resolve_tokenizer_dir(str(tmp_path)) == str(tmp_path)


def test_merge_config_inputs_keeps_pretrain_row_in_sync():
    from ui.presets import CONFIG_INPUT_ORDER, DEFAULT_PRESET, get_preset, values_in_input_order
    from ui.tabs.config_tab import merge_config_inputs
    cfg = get_preset(DEFAULT_PRESET)
    values = values_in_input_order(cfg)
    values[CONFIG_INPUT_ORDER.index("learning_rate")] = 7e-4
    merged = merge_config_inputs(cfg, values)
    assert merged["stage_overrides"]["pretrain"]["learning_rate"] == 7e-4
    assert merged["stage_overrides"]["sft"] == cfg["stage_overrides"]["sft"]


# ── estimator ──────────────────────────────────────────────────────────
def test_estimator_derives_recurrent_groups_like_the_model():
    from chronos.config_schema import model_config_kwargs
    from chronos.model.factory import count_parameters
    from ui.estimator import ArchConfig, active_param_bounds, compute_params, total_params
    model = json.load(open(ARCH_50M))["model_config"]
    arch = ArchConfig.from_config(model)
    counted = count_parameters(model_config_kwargs(model))
    assert (arch.prelude_layers, arch.recurrent_groups, arch.coda_layers) == (1, 4, 1)
    assert total_params(arch) == counted["total"]
    assert active_param_bounds(arch) == (counted["active"], counted["active_union_upper"])
    modes = {m: compute_params(ArchConfig.from_config({**model, "recurrent_attention_mode": m}))
             for m in ("first_only", "shared_first", "per_iteration")}
    assert modes["first_only"] < modes["shared_first"] < modes["per_iteration"]


# ── planner panel handlers ─────────────────────────────────────────────
def test_planner_panel_handlers(tmp_path):
    from ui.planner_panel import apply_stage_rows, plan_summary, run_plan, save_plan, stage_rows
    plan = run_plan(20, True, 8, "experts", 2, "balanced", "", 16, 0, 4, 6400, 256)
    assert plan["report"]["architecture"]["top_k"] == 2
    assert "estimated" in plan_summary(plan) and "measured" in plan_summary(plan)
    by_params = run_plan(20, True, 8, "params", 6, "none", "", 16, 0.1, 4, 6400, 256)
    assert by_params["report"]["scaling"]["tokens"] == 1e8
    with pytest.raises(ValueError, match="active_experts"):
        run_plan(20, True, 8, "experts", 9, "none", "", 16, 0, 4, 6400, 256)
    with pytest.raises(ValueError, match="target"):
        run_plan(0, True, 8, "experts", 2, "none", "", 16, 0, 4, 6400, 256)
    rows = [[str(v) for v in row] for row in stage_rows(plan["config"])]
    rows[0][1 + STAGE_KEYS.index("learning_rate")] = "0.0009"
    cfg = apply_stage_rows(plan["config"], rows)
    assert cfg["learning_rate"] == 0.0009 and cfg["stage_overrides"]["pretrain"]["learning_rate"] == 0.0009
    path = save_plan(dict(plan, config=cfg), str(tmp_path / "plan.json"))
    assert json.load(open(path))["config"]["learning_rate"] == 0.0009


# ── tokenizer / serve tab helpers ──────────────────────────────────────
def test_tokenizer_tab_helpers():
    from ui.tabs.tokenizer_tab import parse_paths, report_rows, report_summary
    assert parse_paths(' "~/a" , b\n\n c ') == [os.path.expanduser("~/a"), "b", "c"]
    report = {"vocab_size": 16384, "algorithm": "bpe", "heldout_eval": {
        "zh": {"texts": 10, "bytes_per_token": 3.1, "chars_per_token": 1.2}, "vocab_utilization": 0.8}}
    assert report_rows(report["heldout_eval"]) == [["zh", 10, 3.1, 1.2, None, None, None]]
    assert report_summary(report) == {"vocab_size": 16384, "algorithm": "bpe", "vocab_utilization": 0.8}
    assert report_summary({}) == {}


def test_serve_command_keeps_the_key_out_of_argv(tmp_path):
    from ui.tabs.serve_tab import build_serve_command, redact
    argv, env = build_serve_command(models_dir=str(tmp_path), kv_cache_dtype="int8", router_scope="checkpoint",
                                    ram_budget_mb=512)
    assert "--kv-cache-dtype" in argv and "--router-scope" not in argv and env == {}
    with pytest.raises(ValueError, match="remote_needs_key"):
        build_serve_command(models_dir=str(tmp_path), host="0.0.0.0", allow_remote=True)
    with pytest.raises(ValueError, match="remote_needs_key"):
        build_serve_command(models_dir=str(tmp_path), host="0.0.0.0", api_key="secret")
    argv, env = build_serve_command(models_dir=str(tmp_path), host="0.0.0.0", allow_remote=True, api_key="secret")
    assert "secret" not in " ".join(argv) and "secret" not in redact(argv) and env == {"CHRONOS_API_KEY": "secret"}
    for bad in (dict(models_dir=str(tmp_path / "missing")), dict(models_dir=str(tmp_path), port=70000)):
        with pytest.raises(ValueError):
            build_serve_command(**bad)


def test_inference_overrides_validate():
    from chronos.model.config import ChronosConfig
    from chronos.serving.runtime import apply_inference_overrides
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        num_experts=4)
    assert apply_inference_overrides(cfg, {"kv_cache_dtype": "int8", "ram_budget_mb": None}) == {"kv_cache_dtype": "int8"}
    assert cfg.kv_cache_dtype == "int8"
    # Regression: the 50M shape (6 heads / 2 KV) was validated against
    # class-default KV heads (4) and every override raised.
    recipe = json.load(open(ARCH_50M))["model_config"]
    big = ChronosConfig(**recipe)
    assert apply_inference_overrides(big, {"router_scope": "shared", "ram_budget_mb": 64}) == {
        "router_scope": "shared", "ram_budget_mb": 64}
    assert big.num_key_value_heads == 2 and big.router_scope == "shared"
    with pytest.raises(ValueError, match="not an inference option"):
        apply_inference_overrides(cfg, {"hidden_size": 8})
    with pytest.raises(ValueError, match="positive"):
        apply_inference_overrides(cfg, {"max_resident_experts": 0})


def test_new_ui_keys_exist_in_every_language():
    from ui.i18n import TRANSLATIONS
    keys = {k for k in TRANSLATIONS["en"] if k.split(".")[0] in ("plan", "tok", "serve")}
    assert len(keys) > 80
    for lang, table in TRANSLATIONS.items():
        missing = keys - set(table)
        assert not missing, (lang, sorted(missing)[:5])
