"""Focused invariants for the Chronos v2 execution and paging contracts."""
from __future__ import annotations

import tempfile
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Some environments ship an unrelated PyPI package named ``chronos``.  Test
# the workspace implementation deterministically.
loaded = sys.modules.get("chronos")
if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(_repo_root):
    for name in list(sys.modules):
        if name == "chronos" or name.startswith("chronos."):
            sys.modules.pop(name, None)

import torch

from chronos.data.flexible_dataset import multimodal_collate
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.recurrent import recurrent_auxiliary_state
from chronos.model.recurrent import RecurrentController
from chronos.trainer.optim_utils import PagedAdamW, build_paged_optimizer


def tiny_config(**overrides):
    values = dict(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_experts=3,
        num_experts_per_tok=1,
        num_shared_experts=1,
        max_position_embeddings=32,
        use_hybrid_attention=True,
        recurrent_enabled=True,
        prelude_layers=1,
        recurrent_groups=2,
        coda_layers=1,
        recurrent_depth=3,
        min_recurrent_depth=1,
        dropout=0.0,
    )
    values.update(overrides)
    return ChronosConfig(**values)


def test_act_mass_is_monotonic_and_final_halt_is_one():
    states = [torch.randn(2, 3, 8) for _ in range(4)]
    probs = [torch.full((2, 3), 0.3, requires_grad=True) for _ in states]
    for threshold in (0.0, 0.5, 0.99, 1.0):
        mixed, state = recurrent_auxiliary_state(
            states, probs, min_depth=2, threshold=threshold,
        )
        assert torch.isfinite(mixed).all()
        assert torch.all(state.halting_probs[..., 1:] >= state.halting_probs[..., :-1])
        assert torch.equal(state.halting_probs[..., -1], torch.ones_like(probs[0]))
        assert state.depth_used.min() >= 2
        assert state.depth_used.max() <= len(states)


def test_depth_one_uses_the_v1_execution_graph():
    cfg_v1 = tiny_config(
        architecture_version=1, recurrent_enabled=False, recurrent_depth=1,
    )
    cfg_v2 = tiny_config(
        architecture_version=2, recurrent_enabled=True, recurrent_depth=1,
    )
    torch.manual_seed(7)
    v1 = ChronosForCausalLM(cfg_v1).eval()
    v2 = ChronosForCausalLM(cfg_v2).eval()
    v2.load_state_dict(v1.state_dict(), strict=False)
    ids = torch.randint(0, cfg_v1.vocab_size, (2, 6))
    with torch.no_grad():
        out1 = v1(ids, labels=ids, use_cache=True)
        out2 = v2(ids, labels=ids, use_cache=True)
    torch.testing.assert_close(out1.logits, out2.logits, rtol=0, atol=0)
    torch.testing.assert_close(out1.loss, out2.loss, rtol=0, atol=0)
    for cache1, cache2 in zip(out1.past_key_values, out2.past_key_values):
        for tensor1, tensor2 in zip(cache1, cache2):
            torch.testing.assert_close(tensor1, tensor2, rtol=0, atol=0)


def test_recurrent_decode_appends_one_kv_position_per_token():
    model = ChronosForCausalLM(tiny_config()).eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.no_grad():
        output = model(ids, use_cache=True)
        lengths = [entry[0].shape[1] for entry in output.past_key_values]
        assert lengths == [4] * model.config.num_hidden_layers
        output = model(
            torch.randint(0, model.config.vocab_size, (1, 1)),
            past_key_values=output.past_key_values,
            use_cache=True,
        )
        assert [entry[0].shape[1] for entry in output.past_key_values] == [5] * 4


def test_latent_refinement_reduces_attention_calls():
    cfg = tiny_config(recurrent_depth=4, recurrent_latent_refinement=True)
    model = ChronosForCausalLM(cfg).eval()
    calls = {"count": 0}
    handles = []
    for group in range(cfg.recurrent_groups):
        layer = model.model.layers[cfg.prelude_layers + group]
        handles.append(layer.self_attn.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("count", calls["count"] + 1)
        ))
    try:
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (1, 4)), use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    assert calls["count"] == cfg.recurrent_groups


def test_adaptive_exit_uses_accumulated_halt_mass():
    cfg = tiny_config(
        recurrent_depth=4,
        min_recurrent_depth=1,
        adaptive_halting=True,
        halting_threshold=0.8,
        recurrent_stability_threshold=100.0,
    )
    model = ChronosForCausalLM(cfg).eval()
    # A raw p=0.5 should cross the ACT threshold after two steps.  This
    # specifically guards against checking the raw sigmoid against the
    # accumulated threshold, which would incorrectly force the final step.
    torch.nn.init.zeros_(model.model.recurrent_controller.halting_head.weight)
    torch.nn.init.zeros_(model.model.recurrent_controller.halting_head.bias)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        output = model(ids, use_cache=False)
    assert output.cache_hints["early_exit_ratio"].item() == 1.0
    assert output.recurrent_depth_used.mean().item() <= 2.0
    assert torch.allclose(
        output.halting_probs[..., -1],
        torch.ones_like(output.halting_probs[..., -1]),
    )


def test_recurrent_depth_curriculum_is_bounded_and_reports_contraction():
    cfg = tiny_config(recurrent_depth=6, min_recurrent_depth=2,
                      recurrent_strategy="scheduled", recurrent_depth_jitter=1.0)
    controller = RecurrentController(cfg)
    controller.set_training_progress(0, 100)
    assert controller.execution_depth(cfg, True) == 2
    controller.set_training_progress(100, 100)
    depth = controller.execution_depth(cfg, True)
    assert 2 <= depth <= 6
    states = [torch.ones(1, 2, 4), torch.ones(1, 2, 4) * 1.1]
    _, state = recurrent_auxiliary_state(states, [torch.full((1, 2), .5)] * 2,
                                         min_depth=1, threshold=.9)
    assert torch.isfinite(state.contraction)
    assert state.contraction.item() > 0


def test_mixed_modality_mask_is_per_modality_and_finite():
    cfg = tiny_config(
        recurrent_enabled=False,
        modalities=["image", "audio"],
        image_patch_size=8,
        max_image_tokens=4,
        max_audio_tokens=4,
    )
    model = ChronosForCausalLM(cfg).eval()

    def record(mask, media):
        ids = torch.randint(0, cfg.vocab_size, (6,))
        return {"input_ids": ids, "labels": ids.clone(),
                "modality_inputs": media, "modality_mask": torch.tensor(mask)}

    text_ids = torch.randint(0, cfg.vocab_size, (6,))
    batch = multimodal_collate([
        (text_ids, text_ids.clone()),
        record([1, 0], {"image_embeds": torch.randn(3, 16)}),
        record([0, 1], {"audio_embeds": torch.randn(2, 16)}),
    ])
    assert batch["modality_mask"].tolist() == [[0, 0], [1, 0], [0, 1]]
    assert batch["modality_inputs"]["image_embeds_mask"].shape == (3, 3)
    with torch.no_grad():
        output = model(**batch)
    assert torch.isfinite(output.logits).all()


def test_paged_optimizer_checkpoint_omits_expert_moments():
    model = ChronosForCausalLM(tiny_config(recurrent_enabled=False))
    with tempfile.TemporaryDirectory() as page_dir:
        optimizer = build_paged_optimizer(model, 1e-3, page_dir=page_dir)
        ids = torch.randint(0, model.config.vocab_size, (2, 6))
        model(ids, labels=ids).loss.backward()
        optimizer.step()
        payload = optimizer.state_dict()
        serialized_to_live = {}
        for serialized_group, live_group in zip(payload["param_groups"], optimizer.param_groups):
            serialized_to_live.update(zip(serialized_group["params"], live_group["params"]))
        for state_id in payload["state"]:
            name = optimizer._names.get(id(serialized_to_live[state_id]), "")
            assert optimizer._page_key(name) is None
        assert payload["chronos_paging"]["page_dir"] == page_dir
        assert isinstance(optimizer, PagedAdamW)
