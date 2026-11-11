"""recurrent_order="block": (A B C ...) x depth instead of A x depth, B x depth, ...

The default ("layer") must stay bit-identical to 0.3.2; "block" must be a
different, self-consistent computation: KV-cache decode equals full forward,
Torch and MLX agree, speculative decoding stays exact, gradients reach every
recurrent layer, and the config validates.
"""
from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402


def _cfg(order, **kw):
    base = dict(architecture_version=3, hidden_size=64, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
        recurrent_depth=3, min_recurrent_depth=2, max_inference_depth=3, num_attention_heads=4,
        num_key_value_heads=2, kv_latent_dim=16, rope_dim=8, num_experts=4, num_experts_per_tok=2,
        num_shared_experts=1, moe_intermediate_size=32, vocab_size=97, max_position_embeddings=64,
        sliding_window_size=64, dropout=0.0, fallback_mask_prob=0.0, adaptive_halting=False,
        recurrent_order=order)
    base.update(kw)
    return ChronosConfig(**base)


def _torch_model(order, seed=5, **kw):
    torch.manual_seed(seed)
    return ChronosForCausalLM(_cfg(order, **kw)).eval()


def test_default_is_layer_order_and_value_is_validated():
    assert _cfg("layer").recurrent_order == "layer"
    assert ChronosConfig(hidden_size=64, num_hidden_layers=4).recurrent_order == "layer"
    assert _cfg("BLOCK").recurrent_order == "block"
    with pytest.raises(ValueError):
        _cfg("diagonal")


def test_order_is_a_model_field_and_survives_a_config_round_trip():
    from chronos.config_schema import MODEL_FIELDS
    assert "recurrent_order" in MODEL_FIELDS
    cfg = _cfg("block")
    assert ChronosConfig(**{k: v for k, v in cfg.to_dict().items() if k in MODEL_FIELDS}).recurrent_order == "block"


def test_block_and_layer_order_differ_only_when_there_is_something_to_reorder():
    ids = torch.randint(3, 97, (2, 9))
    with torch.no_grad():
        layer = _torch_model("layer")(ids).logits
        block = _torch_model("block")(ids).logits
    assert (layer - block).abs().max() > 1e-3        # a genuinely different computation
    # One recurrent layer per block: A x D and (A) x D are the same sequence.
    with torch.no_grad():
        one_layer = _torch_model("layer", num_hidden_layers=3, recurrent_groups=1)(ids).logits
        one_block = _torch_model("block", num_hidden_layers=3, recurrent_groups=1)(ids).logits
    assert torch.equal(one_layer, one_block)


@pytest.mark.parametrize("mode", ["shared_first", "first_only", "per_iteration"])
def test_block_order_decode_with_cache_equals_full_forward(mode):
    model = _torch_model("block", recurrent_attention_mode=mode)
    ids = torch.randint(3, 97, (1, 12))
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :8], use_cache=True)
        cache, steps = out.past_key_values, [out.logits]
        for t in range(8, 12):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits)
    got = torch.cat(steps, dim=1)
    assert (got - full).abs().max() < 2e-4


def test_block_order_cache_has_one_slot_per_physical_layer_in_shared_first():
    model = _torch_model("block")
    out = model(torch.randint(3, 97, (1, 6)), use_cache=True)
    assert len(out.past_key_values) == model.config.num_hidden_layers


def test_block_order_trains_every_recurrent_layer_and_the_controller():
    model = _torch_model("block").train()
    ids = torch.randint(3, 97, (2, 10))
    out = model(ids, labels=ids)
    out.loss.backward()
    cfg = model.config
    for li in range(cfg.prelude_layers, cfg.prelude_layers + int(cfg.recurrent_groups)):
        grads = [p.grad for p in model.model.layers[li].parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads), f"layer {li} received no gradient"
    ctrl = model.model.recurrent_controller
    assert float(ctrl.step_sizes.grad.abs().sum()) > 0
    assert float(ctrl.loop_embeddings.grad.abs().sum()) > 0


def test_block_order_has_one_halting_trajectory_and_reports_group_axis():
    model = _torch_model("block", adaptive_halting=True).eval()
    with torch.no_grad():
        model(torch.randint(3, 97, (2, 7)))
    state = model.model.last_recurrent_state
    g = int(model.config.recurrent_groups)
    assert state["halting_probs"].shape[-2] == 1          # one block trajectory, not one per layer
    assert state["depth_used"].shape[-1] == 1
    assert g > 1


def test_block_order_executes_virtual_layers_in_interleaved_sequence():
    model = _torch_model("block")
    order = []
    for li, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda _m, _i, _o, li=li: order.append(li))
    with torch.no_grad():
        model(torch.randint(3, 97, (1, 5)))
    p, g, d = 1, int(model.config.recurrent_groups), 3
    assert order == [0] + [p + k for _ in range(d) for k in range(g)] + list(range(p + g, len(model.model.layers)))


def test_layer_order_executes_virtual_layers_consecutively():
    model = _torch_model("layer")
    order = []
    for li, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda _m, _i, _o, li=li: order.append(li))
    with torch.no_grad():
        model(torch.randint(3, 97, (1, 5)))
    p, g, d = 1, int(model.config.recurrent_groups), 3
    assert order == [0] + [p + k for k in range(g) for _ in range(d)] + list(range(p + g, len(model.model.layers)))


# ---- MLX ---------------------------------------------------------------

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def _mlx_pair(order, seed=5, **kw):
    from chronos.mlx.model import ChronosMLXModel
    cfg = _cfg(order, **kw)
    torch.manual_seed(seed)
    pt = ChronosForCausalLM(cfg).eval()
    return pt, ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()


@pytest.mark.parametrize("mode", ["shared_first", "per_iteration"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_mlx_matches_torch_for_block_order(mode, adaptive):
    pt, mm = _mlx_pair("block", recurrent_attention_mode=mode, adaptive_halting=adaptive)
    ids = torch.randint(3, 97, (2, 11))
    with torch.no_grad():
        want = pt(ids).logits.numpy()
    with mx.stream(mx.cpu):
        got, _, _ = mm(mx.array(ids.numpy(), dtype=mx.int32), logits_dtype=mx.float32)
        got = np.array(got)
    assert np.abs(got - want).max() < 5e-2, np.abs(got - want).max()


def test_mlx_block_order_decode_with_cache_equals_full_forward():
    _, mm = _mlx_pair("block")
    ids = mx.array(np.random.default_rng(1).integers(3, 97, size=(1, 12)), dtype=mx.int32)
    with mx.stream(mx.cpu):
        full, _, _ = mm(ids, logits_dtype=mx.float32)
        part, _, cache = mm(ids[:, :8], logits_dtype=mx.float32)
        rows = [part]
        for t in range(8, 12):
            step, _, cache = mm(ids[:, t:t + 1], cache=cache, logits_dtype=mx.float32)
            rows.append(step)
        got = mx.concatenate(rows, axis=1)
    assert float(mx.abs(got - full).max().item()) < 2e-4


@pytest.mark.parametrize("gamma,draft_depth", [(4, 1), (6, 2)])
def test_mlx_speculative_is_exact_in_block_order(gamma, draft_depth):
    from chronos.mlx.speculative import speculative_generate
    _, mm = _mlx_pair("block")
    ids = [int(t) for t in np.random.default_rng(3).integers(3, 97, size=7)]
    with mx.stream(mx.cpu):
        logits, _, cache = mm(mx.array([ids], dtype=mx.int32))
        ref, nxt = list(ids), int(mx.argmax(logits[0, -1]).item())
        for _ in range(20):
            ref.append(nxt)
            logits, _, cache = mm(mx.array([[nxt]], dtype=mx.int32), cache=cache)
            nxt = int(mx.argmax(logits[0, -1]).item())
        out, _ = speculative_generate(mm, ids, max_new_tokens=20, gamma=gamma, draft_depth=draft_depth, gamma_max=8)
    assert out == ref


def _engine(tmp_path, **kw):
    from chronos.mlx.inference import ChronosMLXInferenceEngine
    pt, mm = _mlx_pair("block", sliding_window_size=4, **kw)
    return mm, ChronosMLXInferenceEngine(mm, mm.config, ssd_dir=str(tmp_path / "experts"))


def test_mlx_engine_block_order_speculative_matches_plain_greedy(tmp_path):
    with mx.stream(mx.cpu):
        _, engine = _engine(tmp_path)
        engine.store.storage_format = "full_dram"
        try:
            ids = mx.array([[5, 9, 14, 3, 22, 7, 31]], dtype=mx.int32)
            plain = list(engine.generate(ids, max_new_tokens=16, temperature=0.0))
            spec = list(engine.generate(ids, max_new_tokens=16, temperature=0.0, speculative=True, draft_depth=1, gamma=4))
        finally:
            engine.stop()
    assert spec == plain and len(spec) == 16


def test_mlx_engine_block_order_runs_with_lazy_experts(tmp_path):
    with mx.stream(mx.cpu):
        _, engine = _engine(tmp_path, recommended_resident_experts=2, recommended_ram_experts=2)
        try:
            engine.setup(warm_expert_ids=[0, 1])
            assert engine._all_resident_fast_path_enabled() is False
            toks = list(engine.generate(mx.array([[5, 9, 14, 3, 22, 7]], dtype=mx.int32), max_new_tokens=10, temperature=0.0, speculative=True, draft_depth=1, gamma=3))
            stats = engine.last_stats
        finally:
            engine.stop()
    assert len(toks) == 10 and all(0 <= t < 97 for t in toks)
    assert stats["speculative"]["generated"] == 10
