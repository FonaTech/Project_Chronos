"""Chronos v3 recurrent-depth kernel invariants (Torch and MLX)."""
from __future__ import annotations

import os
import sys

import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import chronos.model.hybrid_attention as ha
from chronos.config_schema import recurrent_cache_layout, resolve_recurrent_attention_mode
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM, apply_repetition_penalty

MODES = ("shared_first", "per_iteration", "first_only")


def cfg_for(mode=None, **overrides):
    values = dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=128, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=64, max_position_embeddings=64, recurrent_depth=3,
        prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
    )
    if mode is not None:
        values["recurrent_attention_mode"] = mode
    values.update(overrides)
    return ChronosConfig(**values)


def attention_calls(model, ids):
    calls = {"n": 0}
    handles = [m.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
               for m in model.modules() if isinstance(m, (ha.MLAAttention, ha.SlidingWindowAttention))]
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return calls["n"]


def test_v3_default_mode_and_v2_resolution():
    assert cfg_for().recurrent_attention_mode == "shared_first"
    assert cfg_for(architecture_version=2).recurrent_attention_mode == "first_only"
    assert cfg_for(recurrent_latent_refinement=False).recurrent_attention_mode == "per_iteration"
    assert resolve_recurrent_attention_mode({"recurrent_latent_refinement": True}) == "first_only"
    assert resolve_recurrent_attention_mode({"architecture_version": 3}) == "shared_first"
    with pytest.raises(ValueError):
        cfg_for("bogus")


@pytest.mark.parametrize("mode", MODES)
def test_attention_runs_every_iteration_except_first_only(mode):
    cfg = cfg_for(mode)
    model = ChronosForCausalLM(cfg).eval()
    ids = torch.randint(3, cfg.vocab_size, (1, 6))
    expected = cfg.virtual_num_hidden_layers if mode != "first_only" else cfg.num_hidden_layers
    assert attention_calls(model, ids) == expected


@pytest.mark.parametrize("window", [64, 3])
@pytest.mark.parametrize("mode", MODES)
def test_prefill_decode_matches_full_forward(mode, window):
    torch.manual_seed(3)
    cfg = cfg_for(mode, sliding_window_size=window)
    model = ChronosForCausalLM(cfg).eval()
    ids = torch.randint(3, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :5], use_cache=True)
        steps, cache = [out.logits[:, -1]], out.past_key_values
        assert len(cache) == recurrent_cache_layout(cfg, cfg.max_inference_depth).num_slots
        for t in range(5, 11):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits[:, -1])
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 4:11], atol=1e-5, rtol=1e-4)


def test_shared_first_kv_slots_equal_physical_layers_and_per_iteration_scales():
    shared, per_iter = cfg_for("shared_first"), cfg_for("per_iteration")
    assert recurrent_cache_layout(shared, 3).num_slots == shared.num_hidden_layers
    assert recurrent_cache_layout(per_iter, 3).num_slots == per_iter.virtual_num_hidden_layers
    layout = recurrent_cache_layout(per_iter, 3)
    assert [layout.physical_layer(s) for s in range(layout.num_slots)] == [0, 1, 1, 1, 2, 2, 2, 3]


@pytest.mark.parametrize("mode", MODES)
def test_gradients_reach_attention_and_gate(mode):
    torch.manual_seed(5)
    model = ChronosForCausalLM(cfg_for(mode)).train()
    ids = torch.randint(3, model.config.vocab_size, (2, 8))
    model(ids, labels=ids).loss.backward()
    named = dict(model.named_parameters())
    assert named["model.layers.1.self_attn.q_proj.weight"].grad.abs().sum() > 0
    assert named["model.layers.1.mlp.gate.weight"].grad.abs().sum() > 0


def test_shared_first_decode_early_exit_keeps_cache_consistent():
    torch.manual_seed(7)
    cfg = cfg_for("shared_first", adaptive_halting=True, halting_threshold=0.5,
                  recurrent_stability_threshold=1e9, recurrent_depth=4)
    model = ChronosForCausalLM(cfg).eval()
    torch.nn.init.zeros_(model.model.recurrent_controller.halting_head.weight)
    torch.nn.init.constant_(model.model.recurrent_controller.halting_head.bias, 5.0)
    ids = torch.randint(3, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        out = model(ids[:, :4], use_cache=True)
        out = model(ids[:, 4:5], past_key_values=out.past_key_values, use_cache=True)
        assert out.cache_hints["early_exit_ratio"].item() == 1.0
        assert [entry[0].shape[1] for entry in out.past_key_values[:1]] == [5]
        assert torch.isfinite(out.logits).all()


def test_repetition_penalty_shrinks_both_signs():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    seen = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits.clone(), seen, 2.0)
    assert torch.allclose(out, torch.tensor([[1.0, -4.0, 1.0]]))


def test_v2_checkpoint_numerics_unchanged():
    ckpt = os.path.join(_repo_root, "out", "chronos_256_moe.pth")
    golden = os.path.join(_repo_root, "tests", "fixtures", "v2_golden_logits.pt")
    if not (os.path.exists(ckpt) and os.path.exists(golden)):
        pytest.skip("v2 reference checkpoint not present")
    from chronos.model.checkpoint import chronos_config_from_checkpoint, load_checkpoint_state_dict
    cfg, _ = chronos_config_from_checkpoint(ckpt)
    assert cfg.recurrent_attention_mode == "first_only"
    model = ChronosForCausalLM(cfg).eval()
    model.load_state_dict(load_checkpoint_state_dict(ckpt, map_location="cpu"), strict=False)
    ref = torch.load(golden)
    with torch.no_grad():
        full = model(ref["ids"]).logits
    torch.testing.assert_close(full[..., :512], ref["full"], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_mlx_matches_torch_for_every_mode(mode):
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.model import ChronosMLXModel
    torch.manual_seed(11)
    cfg = cfg_for(mode, sliding_window_size=3)
    pt = ChronosForCausalLM(cfg).eval()
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()
        cache_pt = cache_mx = None
        for chunk in (torch.tensor([[1, 4, 2, 5, 7]]), torch.tensor([[6]]), torch.tensor([[9]])):
            with torch.no_grad():
                out = pt(chunk, past_key_values=cache_pt, use_cache=True)
            logits, _, cache_mx = model(mx.array(chunk.numpy()), cache=cache_mx)
            mx.eval(logits)
            np.testing.assert_allclose(np.array(logits), out.logits.numpy(), atol=2e-5, rtol=2e-4)
            cache_pt = out.past_key_values
        assert len(cache_mx) == len(cache_pt)


@pytest.mark.parametrize("mode", ["shared_first", "per_iteration"])
def test_mlx_full_model_gradients_match_torch(mode):
    """End-to-end CE gradient parity (fp32, CPU) through attention, sparse MoE and recurrence."""
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as mlx_nn
    from mlx.utils import tree_flatten
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.io import mlx_state_to_torch
    from chronos.mlx.training.trainer import _masked_lm_loss
    torch.manual_seed(13)
    cfg = cfg_for(mode, lambda_balance=0.0)
    pt = ChronosForCausalLM(cfg).train()
    ids = torch.randint(3, cfg.vocab_size, (2, 9))
    pt_loss = pt(ids, labels=ids).loss
    pt_loss.backward()
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg).train()
        mids = mx.array(ids.numpy())

        def loss_fn(inp):
            logits, _, _ = model(inp)
            return _masked_lm_loss(logits, inp)

        value, grads = mlx_nn.value_and_grad(model, loss_fn)(mids)
        mx.eval(value, grads)
        assert abs(value.item() - pt_loss.item()) < 1e-4
        # Map MLX gradients onto Torch parameter names via the checkpoint bridge.
        model.update(grads)
        mlx_grads = mlx_state_to_torch(model, cfg)
    compared = []
    for name, param in pt.named_parameters():
        if name not in mlx_grads or param.grad is None or name == "lm_head.weight":
            continue
        ref = param.grad
        scale = ref.abs().max().clamp_min(1e-8)
        err = ((mlx_grads[name].float() - ref).abs().max() / scale).item()
        assert err < 1e-3, (name, err)
        compared.append(name)
    # Guard against a vacuous pass: attention, routed experts, gate and
    # recurrent controller gradients must all have been compared.
    assert len(compared) > 40
    for needle in ("self_attn.q_proj", "mlp.experts.", "mlp.gate", "mlp.shared_experts", "embed_tokens"):
        assert any(needle in n for n in compared), needle


def _adaptive_model(seed=31, bias=0.0):
    torch.manual_seed(seed)
    cfg = cfg_for("shared_first", adaptive_halting=True, halting_threshold=0.7,
                  recurrent_depth=4, min_recurrent_depth=1)
    model = ChronosForCausalLM(cfg).eval()
    head = model.model.recurrent_controller.halting_head
    torch.nn.init.normal_(head.weight, std=0.5)   # token-dependent halting
    torch.nn.init.constant_(head.bias, bias)
    return model


@pytest.mark.parametrize("bias", [-1.0, 0.0, 2.0])
def test_adaptive_act_exit_is_exact_and_batch_invariant(bias):
    """Each token's output must not depend on the other tokens in its chunk."""
    model = _adaptive_model(bias=bias)
    ids = torch.randint(3, model.config.vocab_size, (1, 10))
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :4], use_cache=True)
        steps, cache = [out.logits[:, -1]], out.past_key_values
        for t in range(4, 9):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits[:, -1])
        chunked = model(ids[:, 6:10], past_key_values=truncate_to(cache, 6), use_cache=True).logits
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 3:9], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(chunked, full[:, 6:10], atol=1e-5, rtol=1e-4)


def truncate_to(cache, length):
    from chronos.runtime.speculative import truncate_cache
    return truncate_cache(cache, length)


def test_speculative_greedy_exact_with_adaptive_halting():
    from chronos.runtime.speculative import speculative_generate
    model = _adaptive_model(bias=0.5)
    ids = torch.randint(3, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        out = model(ids, use_cache=True)
        cache, seq, nxt = out.past_key_values, ids[0].tolist(), int(out.logits[0, -1].argmax())
        for _ in range(16):
            seq.append(nxt)
            out = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            cache, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
        spec, stats = speculative_generate(model, ids, max_new_tokens=16, gamma=4, draft_depth=1)
    assert spec[0].tolist() == seq


def test_routing_is_recorded_for_every_virtual_layer():
    from chronos.trainer.loss_mixin import collect_iteration_router_probs, first_iteration_router_probs
    cfg = cfg_for("shared_first")
    model = ChronosForCausalLM(cfg).train()
    ids = torch.randint(3, cfg.vocab_size, (2, 7))
    model(ids, labels=ids)
    every = collect_iteration_router_probs(model)
    assert every.shape[2] == cfg.virtual_num_hidden_layers
    assert first_iteration_router_probs(model).shape[2] == cfg.num_hidden_layers
    model(ids, labels=ids)                     # reset per forward, no accumulation
    assert collect_iteration_router_probs(model).shape[2] == cfg.virtual_num_hidden_layers


def test_mlx_routing_is_recorded_for_every_virtual_layer():
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.trainer import _collect_router_probs, _first_iteration_router_probs
    cfg = cfg_for("shared_first")
    pt = ChronosForCausalLM(cfg).eval()
    model = ChronosMLXModel.from_chronos_pytorch(pt, cfg).train()
    ids = mx.array(torch.randint(3, cfg.vocab_size, (2, 7)).numpy())
    for _ in range(2):
        model(ids)
        assert _collect_router_probs(model).shape[2] == cfg.virtual_num_hidden_layers
        assert _first_iteration_router_probs(model).shape[2] == cfg.num_hidden_layers
