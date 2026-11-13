"""The 50M en→zh recipe: the kernel planner reproduces its shape, the
translation runner accepts it, and the model presets match it."""
import json
import os

import pytest

DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(DEMO))
RECIPE = os.path.join(DEMO, "recipes", "enzh_50m_72h.json")
SHARDS = os.path.join(REPO, "Translation Training Data", "shards_v1")


def _recipe():
    with open(RECIPE, encoding="utf-8") as fh:
        return json.load(fh)


def test_recipe_params_match_the_kernel_count():
    from chronos.model.factory import count_parameters
    recipe = _recipe()
    counted = count_parameters(recipe["model_config"])
    for key in ("total", "active", "active_union_upper"):
        assert counted[key] == recipe["extra"]["params"][key], key


def test_recipe_topology_uses_the_demo_base_and_tokens():
    from chronos_translate.model_presets import ENZH_50M_BASE, SPECIAL_TOKENS
    model = _recipe()["model_config"]
    for key in ("recurrent_depth", "recurrent_attention_mode", "num_experts_per_tok", "num_shared_experts",
                "kv_latent_dim", "sliding_window_size", "prelude_layers", "coda_layers"):
        assert model[key] == ENZH_50M_BASE[key], key
    assert SPECIAL_TOKENS[:6] == ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|en|>", "<|zh|>",
                                  "<|translate|>")
    assert len(SPECIAL_TOKENS) == 64


def test_translation_runner_loads_the_recipe_the_kernel_rejects():
    pytest.importorskip("mlx.core")
    from chronos.mlx.training.longrun import load_spec
    from chronos_translate.run import EXTRA_PATH_KEYS, TranslationRunSpec
    with pytest.raises(ValueError, match="chronos-translate run"):
        load_spec(RECIPE)                                    # bleu_samples is a task key
    spec = load_spec(RECIPE, spec_cls=TranslationRunSpec, extra_path_keys=EXTRA_PATH_KEYS)
    assert spec.bleu_samples == _recipe()["bleu_samples"]
    assert os.path.isabs(spec.run_dir) and os.path.isabs(spec.extra["raw_root"])
    assert spec.run_dir == os.path.join(REPO, "runs", "enzh_50m_v1")


@pytest.mark.skipif(not os.path.isfile(os.path.join(SHARDS, "shards_manifest.json")), reason="local shards absent")
def test_planner_on_production_shards_matches_recipe():
    from chronos.model.planner import PlanRequest, plan_training
    plan = plan_training(PlanRequest(target_total=50e6, num_experts=36, active_experts=2, data_paths=(SHARDS,),
                                     ram_gb=128))
    arch = plan["report"]["architecture"]
    assert (arch["total_params"], arch["active_params"]) == (49_691_717, 12_090_437)
    assert plan["config"]["max_seq_len"] == 512 and plan["config"]["vocab_size"] == 16384
