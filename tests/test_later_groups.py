"""recurrent_later_groups (prefix blocks): block-order iterations after the first run only the first N layers.

The default (0) must change nothing.  A positive value must skip the later layers of the block in iterations
1.., keep iteration 0 complete (every layer still writes its KV), agree between Torch and MLX, keep cache decode
and speculative decoding exact, and still train every layer.
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
mx = pytest.importorskip("mlx.core")

from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402


def _cfg(later, **kw):
    base = dict(architecture_version=3, hidden_size=64, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
        recurrent_depth=3, min_recurrent_depth=2, max_inference_depth=3, num_attention_heads=4,
        num_key_value_heads=2, kv_latent_dim=16, rope_dim=8, num_experts=6, num_experts_per_tok=2,
        num_shared_experts=1, moe_intermediate_size=32, vocab_size=97, max_position_embeddings=64,
        sliding_window_size=64, dropout=0.0, fallback_mask_prob=0.0, adaptive_halting=False,
        recurrent_order="block", recurrent_later_groups=later)
    base.update(kw)
    return ChronosConfig(**base)


def _torch_model(later, seed=5, **kw):
    torch.manual_seed(seed)
    return ChronosForCausalLM(_cfg(later, **kw)).eval()


def _mlx_pair(later, seed=5, **kw):
    from chronos.mlx.model import ChronosMLXModel
    cfg = _cfg(later, **kw)
    torch.manual_seed(seed)
    pt = ChronosForCausalLM(cfg).eval()
    return pt, ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()


# ---- config ---------------------------------------------------------------

def test_default_is_off_and_value_is_validated():
    assert _cfg(0).recurrent_later_groups == 0
    assert ChronosConfig(hidden_size=64, num_hidden_layers=4).recurrent_later_groups == 0
    assert _cfg(2).recurrent_later_groups == 2
    with pytest.raises(ValueError):
        _cfg(-1)
    with pytest.raises(ValueError):
        _cfg(2, recurrent_order="layer")
    with pytest.raises(ValueError):
        _cfg(2, recurrent_attention_mode="per_iteration")


def test_field_is_a_model_field_and_survives_a_config_round_trip():
    from chronos.config_schema import MODEL_FIELDS
    assert "recurrent_later_groups" in MODEL_FIELDS
    cfg = _cfg(2)
    assert ChronosConfig(**{k: v for k, v in cfg.to_dict().items() if k in MODEL_FIELDS}).recurrent_later_groups == 2


# ---- which layers run in which iteration ---------------------------------------

def _count_calls(model):
    calls = {}
    for li, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda _m, _a, _o, li=li: calls.__setitem__(li, calls.get(li, 0) + 1))
    return calls


@pytest.mark.parametrize("mode", ["shared_first", "first_only"])
@pytest.mark.parametrize("later", [1, 2, 3])
def test_later_iterations_run_only_the_prefix_layers(mode, later):
    model = _torch_model(later, recurrent_attention_mode=mode)
    calls = _count_calls(model)
    with torch.no_grad():
        model(torch.randint(3, 97, (1, 5)))
    cfg = model.config
    p, g, d = cfg.prelude_layers, int(cfg.recurrent_groups), int(cfg.recurrent_depth)
    assert g > later
    for li in range(p, p + g):
        assert calls[li] == (d if li - p < later else 1), (li, calls)
    assert all(calls[li] == 1 for li in range(len(model.model.layers)) if not p <= li < p + g)


def test_default_runs_every_layer_in_every_iteration():
    model = _torch_model(0)
    calls = _count_calls(model)
    with torch.no_grad():
        model(torch.randint(3, 97, (1, 5)))
    cfg = model.config
    p, g, d = cfg.prelude_layers, int(cfg.recurrent_groups), int(cfg.recurrent_depth)
    assert all(calls[li] == d for li in range(p, p + g))


def test_prefix_equal_to_the_block_size_changes_nothing_and_a_shorter_prefix_does():
    ids = torch.randint(3, 97, (2, 9))
    g = int(_cfg(0).recurrent_groups)
    with torch.no_grad():
        base = _torch_model(0)(ids).logits
        same = _torch_model(g)(ids).logits
        short = _torch_model(1)(ids).logits
    assert torch.equal(base, same)
    assert (base - short).abs().max() > 1e-4


@pytest.mark.parametrize("mode", ["shared_first", "first_only"])
def test_cache_decode_equals_full_forward_with_prefix_blocks(mode):
    model = _torch_model(2, recurrent_attention_mode=mode)
    ids = torch.randint(3, 97, (1, 10))
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :6], use_cache=True)
        rows, cache = [out.logits], out.past_key_values
        for t in range(6, 10):
            step = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            rows.append(step.logits)
            cache = step.past_key_values
    assert (torch.cat(rows, dim=1) - full).abs().max() < 2e-4


def test_prefix_blocks_train_every_layer_and_the_controller():
    model = _torch_model(1).train()
    ids = torch.randint(3, 97, (2, 9))
    loss = model(ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    cfg = model.config
    for li in range(cfg.prelude_layers, cfg.prelude_layers + int(cfg.recurrent_groups)):
        layer = model.model.layers[li]
        assert float(layer.mlp.gate.weight.grad.abs().sum()) > 0, f"router of layer {li} received no gradient"
    assert float(model.model.recurrent_controller.step_sizes.grad.abs().sum()) > 0


# ---- MLX -----------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["shared_first", "first_only"])
def test_mlx_matches_torch_with_prefix_blocks(mode):
    pt, mm = _mlx_pair(2, recurrent_attention_mode=mode)
    ids = torch.randint(3, 97, (2, 11))
    with torch.no_grad():
        want = pt(ids).logits.numpy()
    with mx.stream(mx.cpu):
        got, _, _ = mm(mx.array(ids.numpy(), dtype=mx.int32), logits_dtype=mx.float32)
        got = np.array(got)
    assert np.abs(got - want).max() < (2e-1 if mode == "first_only" else 5e-2), np.abs(got - want).max()


def test_mlx_prefix_blocks_differ_from_the_full_block():
    _, full = _mlx_pair(0)
    _, short = _mlx_pair(1)
    ids = mx.array(np.random.default_rng(2).integers(3, 97, size=(1, 9)), dtype=mx.int32)
    with mx.stream(mx.cpu):
        a, _, _ = full(ids, logits_dtype=mx.float32)
        b, _, _ = short(ids, logits_dtype=mx.float32)
    assert float(mx.abs(a - b).max().item()) > 1e-4


def test_mlx_decode_with_cache_equals_full_forward_with_prefix_blocks():
    _, mm = _mlx_pair(2)
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
def test_mlx_speculative_is_exact_with_prefix_blocks(gamma, draft_depth):
    from chronos.mlx.speculative import speculative_generate
    _, mm = _mlx_pair(2)
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
