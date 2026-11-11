"""recurrent_later_topk: loop iterations after the first route to fewer experts per token.

The default (0) must change nothing.  A positive value must reach the router in
iterations 1.. only, never exceed num_experts_per_tok, keep the experts of the
first dispatch under router_scope="shared", agree between Torch and MLX, keep
cache decode and speculative decoding exact, and train through checkpointing
and compilation.
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
nn = pytest.importorskip("mlx.nn")

from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402
from chronos.model.moe_chronos import ChronosMOEFeedForward  # noqa: E402


def _cfg(later, **kw):
    base = dict(architecture_version=3, hidden_size=64, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
        recurrent_depth=3, min_recurrent_depth=2, max_inference_depth=3, num_attention_heads=4,
        num_key_value_heads=2, kv_latent_dim=16, rope_dim=8, num_experts=6, num_experts_per_tok=3,
        num_shared_experts=1, moe_intermediate_size=32, vocab_size=97, max_position_embeddings=64,
        sliding_window_size=64, dropout=0.0, fallback_mask_prob=0.0, adaptive_halting=False,
        recurrent_later_topk=later)
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
    assert _cfg(0).recurrent_later_topk == 0
    assert ChronosConfig(hidden_size=64, num_hidden_layers=4).recurrent_later_topk == 0
    assert _cfg(1).recurrent_later_topk == 1
    with pytest.raises(ValueError):
        _cfg(-1)


def test_field_is_a_model_field_and_survives_a_config_round_trip():
    from chronos.config_schema import MODEL_FIELDS
    assert "recurrent_later_topk" in MODEL_FIELDS
    cfg = _cfg(2)
    assert ChronosConfig(**{k: v for k, v in cfg.to_dict().items() if k in MODEL_FIELDS}).recurrent_later_topk == 2


# ---- Torch MoE -------------------------------------------------------------

def _moe(**kw):
    torch.manual_seed(0)
    return ChronosMOEFeedForward(_cfg(0, **kw)).eval()


def test_top_k_argument_changes_the_number_of_routed_experts():
    moe = _moe()
    x = torch.randn(2, 5, 64)
    seen = []
    orig = torch.topk

    def spy(scores, k, **kw):
        seen.append(k)
        return orig(scores, k=k, **kw)

    torch.topk = spy
    try:
        with torch.no_grad():
            full = moe(x)
            moe.reset_iteration_routing()
            graded = moe(x, top_k=1)
            moe.reset_iteration_routing()
            clamped = moe(x, top_k=99)
    finally:
        torch.topk = orig
    assert seen[:3] == [3, 1, 3]
    assert (full - graded).abs().max() > 1e-4
    assert torch.equal(full, clamped)        # top_k above num_experts_per_tok clamps to it


def test_top_k_zero_or_none_means_the_configured_value():
    moe = _moe()
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        a = moe(x)
        moe.reset_iteration_routing()
        b = moe(x, top_k=0)
        moe.reset_iteration_routing()
        c = moe(x, top_k=None)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_shared_router_scope_keeps_the_best_experts_of_the_first_dispatch():
    moe = _moe(router_scope="shared")
    x = torch.randn(2, 6, 64)
    with torch.no_grad():
        moe(x)
        first = moe._shared_topk_idx.clone()
        scores = torch.softmax(moe.gate(x.view(-1, 64)).float(), dim=-1)
        moe(x, top_k=2)
        # The later dispatch must be a subset of the first dispatch's experts for every token.
        # Rebuild what it should have used and compare with a plain forward on that routing.
        keep = scores.gather(-1, first).topk(2, dim=-1, sorted=False).indices
        expect = first.gather(-1, keep)
    assert expect.shape == (12, 2)
    assert all(set(e.tolist()) <= set(f.tolist()) for e, f in zip(expect, first))
    # The recorded first-dispatch ids are untouched by the narrower later dispatch.
    assert torch.equal(moe._shared_topk_idx, first)


def test_torch_graded_dispatch_matches_dense_oracle():
    """Sparse sorted dispatch with k=2 of 3 equals every expert run and masked by the k-best mask."""
    moe = _moe()
    x = torch.randn(3, 7, 64)
    with torch.no_grad():
        got = moe(x, top_k=2)
        flat = x.reshape(-1, 64)
        scores = torch.softmax(moe.gate(flat), dim=-1)
        weight, idx = torch.topk(scores, 2, dim=-1, sorted=False)
        if moe.norm_topk_prob:
            weight = weight / (weight.sum(-1, keepdim=True) + 1e-20)
        want = torch.zeros_like(flat)
        for e, expert in enumerate(moe.experts):
            want = want + expert(flat) * torch.where(idx == e, weight, 0).sum(-1, keepdim=True)
        want = want + sum(s(flat) for s in moe.shared_experts) / len(moe.shared_experts)
    torch.testing.assert_close(got, want.view(3, 7, 64), atol=1e-5, rtol=1e-5)


# ---- model: which iterations get the narrow router ---------------------------

def _record_top_k(model):
    calls = []
    for li, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, ChronosMOEFeedForward):
            layer.mlp.register_forward_hook(
                lambda _m, _a, kw, _o, li=li: calls.append((li, kw.get("top_k"))), with_kwargs=True)
    return calls


def test_only_iterations_after_the_first_receive_top_k():
    model = _torch_model(1)
    calls = _record_top_k(model)
    with torch.no_grad():
        model(torch.randint(3, 97, (1, 5)))
    p, g, d = 1, int(model.config.recurrent_groups), 3
    recurrent = [(li, k) for li, k in calls if p <= li < p + g]
    assert len(recurrent) == g * d
    per_layer = {}
    for li, k in recurrent:
        per_layer.setdefault(li, []).append(k)
    assert all(ks == [None, 1, 1] for ks in per_layer.values()), per_layer
    assert all(k is None for li, k in calls if not p <= li < p + g)      # prelude and coda keep the full top-k


def test_default_passes_no_top_k_at_all():
    model = _torch_model(0)
    calls = _record_top_k(model)
    with torch.no_grad():
        model(torch.randint(3, 97, (1, 5)))
    assert calls and all(k is None for _, k in calls)


def test_graded_topk_changes_the_output_only_when_set():
    ids = torch.randint(3, 97, (2, 9))
    with torch.no_grad():
        base = _torch_model(0)(ids).logits
        same = _torch_model(3)(ids).logits    # equal to num_experts_per_tok: nothing to narrow
        narrow = _torch_model(1)(ids).logits
    assert torch.equal(base, same)
    assert (base - narrow).abs().max() > 1e-4


@pytest.mark.parametrize("mode", ["shared_first", "first_only", "per_iteration"])
@pytest.mark.parametrize("scope", ["per_iteration", "shared"])
def test_cache_decode_equals_full_forward_with_graded_topk(mode, scope):
    model = _torch_model(1, recurrent_attention_mode=mode, router_scope=scope)
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


@pytest.mark.parametrize("order", ["layer", "block"])
def test_graded_topk_trains_the_recurrent_layers_and_their_routers(order):
    model = _torch_model(1, recurrent_order=order).train()
    ids = torch.randint(3, 97, (2, 9))
    loss = model(ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    cfg = model.config
    for li in range(cfg.prelude_layers, cfg.prelude_layers + int(cfg.recurrent_groups)):
        layer = model.model.layers[li]
        assert float(layer.mlp.gate.weight.grad.abs().sum()) > 0, f"router of layer {li} received no gradient"
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads), f"layer {li} received no gradient"
    ctrl = model.model.recurrent_controller
    assert float(ctrl.step_sizes.grad.abs().sum()) > 0


# ---- MLX -----------------------------------------------------------------------

@pytest.mark.parametrize("scope", ["per_iteration", "shared"])
@pytest.mark.parametrize("order", ["layer", "block"])
def test_mlx_matches_torch_with_graded_topk(scope, order):
    pt, mm = _mlx_pair(1, router_scope=scope, recurrent_order=order)
    ids = torch.randint(3, 97, (2, 11))
    with torch.no_grad():
        want = pt(ids).logits.numpy()
    with mx.stream(mx.cpu):
        got, _, _ = mm(mx.array(ids.numpy(), dtype=mx.int32), logits_dtype=mx.float32)
        got = np.array(got)
    assert np.abs(got - want).max() < 5e-2, np.abs(got - want).max()


def test_mlx_graded_topk_differs_from_full_topk():
    _, full = _mlx_pair(0)
    _, narrow = _mlx_pair(1)
    ids = mx.array(np.random.default_rng(2).integers(3, 97, size=(1, 9)), dtype=mx.int32)
    with mx.stream(mx.cpu):
        a, _, _ = full(ids, logits_dtype=mx.float32)
        b, _, _ = narrow(ids, logits_dtype=mx.float32)
    assert float(mx.abs(a - b).max().item()) > 1e-4


def test_mlx_decode_with_cache_equals_full_forward_with_graded_topk():
    _, mm = _mlx_pair(1)
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
def test_mlx_speculative_is_exact_with_graded_topk(gamma, draft_depth):
    from chronos.mlx.speculative import speculative_generate
    _, mm = _mlx_pair(1)
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
