"""MLX activation recomputation and the compiled training step: exactness.

Both are training-only transforms, so every test compares against the
plain eager step on the same weights, batch and RNG seed: loss, every
gradient, router side outputs, loss components and the RNG stream.
"""
from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten  # noqa: E402

CASES = {
    "shared_first": {},
    "per_iteration": dict(recurrent_attention_mode="per_iteration"),
    "first_only": dict(recurrent_attention_mode="first_only"),
    "router_shared": dict(router_scope="shared"),
    "no_recurrence": dict(recurrent_depth=1),
    "block_order": dict(recurrent_order="block"),
    "later_topk": dict(recurrent_later_topk=1),
    "later_topk_shared": dict(recurrent_later_topk=1, router_scope="shared"),
    "dense_attention": dict(use_hybrid_attention=False),
    # SWA window >= sequence: the cache slice k[:, -keep:] is the same graph
    # node as the shared KV view (short SFT buckets hit this).
    "window_covers_sequence": dict(sliding_window_size=64),
}


def _config(**kw):
    from chronos.model.config import ChronosConfig
    base = dict(architecture_version=3, hidden_size=64, num_hidden_layers=4, prelude_layers=1, coda_layers=1,
                recurrent_depth=3, min_recurrent_depth=2, num_attention_heads=4, num_key_value_heads=2,
                kv_latent_dim=16, rope_dim=8, num_experts=6, num_experts_per_tok=2, num_shared_experts=1,
                moe_intermediate_size=32, vocab_size=97, max_position_embeddings=64, sliding_window_size=8,
                dropout=0.1, fallback_mask_prob=0.2, adaptive_halting=True, lookahead_steps=2,
                layer_aware_lookahead=True)
    base.update(kw)
    return ChronosConfig(**base)


def _model(cfg):
    from chronos.mlx.model import ChronosMLXModel
    mx.random.seed(0)
    model = ChronosMLXModel(cfg)
    mx.eval(model.parameters())
    model.train()
    return model


def _loss(model, ids, labels):
    from chronos.mlx.training.trainer import _chronos_regularized_loss
    logits, lookahead, _ = model(ids, logits_dtype=mx.float32)
    ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")
    return _chronos_regularized_loss(model, ce, lookahead, model.config, metrics={"ce": ce, "primary": ce})


def _batch(cfg, b=2, s=12):
    mx.random.seed(1)
    return mx.random.randint(3, cfg.vocab_size, (b, s)), mx.random.randint(3, cfg.vocab_size, (b, s))


def _step(model, grad_fn, ids, labels, seed=3):
    from chronos.mlx.training.trainer import _collect_router_probs
    mx.random.seed(seed)
    loss, grads = grad_fn(model, ids, labels)
    probs = _collect_router_probs(model)
    comps = dict(model._last_loss_components)
    after = mx.random.state[0]
    mx.eval(loss, grads, probs, comps, after)
    return dict(loss=loss, grads=dict(tree_flatten(grads)), probs=probs, comps=comps, rng=after,
                lens=[len(layer.mlp.iteration_router_probs) for layer in model.layers])


def _assert_same(a, b, tol):
    assert abs(float(a["loss"]) - float(b["loss"])) <= tol
    assert a["grads"].keys() == b["grads"].keys()
    worst = max(float(mx.abs(a["grads"][k] - b["grads"][k]).max()) for k in a["grads"])
    assert worst <= tol, worst
    assert a["lens"] == b["lens"]
    assert float(mx.abs(a["probs"] - b["probs"]).max()) <= tol
    assert a["comps"].keys() == b["comps"].keys()
    for k in a["comps"]:
        assert float(mx.abs(a["comps"][k] - b["comps"][k]).max()) <= tol, k
    assert bool(mx.all(a["rng"] == b["rng"])), "RNG stream consumed differently"


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize("case", sorted(CASES))
def test_recomputation_matches_plain_backward(case):
    """Dropout and the cache-miss simulation are replayed, so the gradients
    are those of the network the forward pass ran (a plain ``mx.checkpoint``
    redraws them: same loss, different gradients)."""
    cfg = _config(**CASES[case])
    model = _model(cfg)
    ids, labels = _batch(cfg)
    grad_fn = nn.value_and_grad(model, _loss)
    plain = _step(model, grad_fn, ids, labels)
    model.config.activation_checkpointing = True
    recomputed = _step(model, grad_fn, ids, labels)
    _assert_same(plain, recomputed, 1e-6)
    zero = [k for k, g in recomputed["grads"].items() if float(mx.abs(g).max()) == 0.0]
    assert zero == [k for k, g in plain["grads"].items() if float(mx.abs(g).max()) == 0.0]


def test_repeated_region_outputs_keep_their_gradients():
    """mlx 0.32.4 drops the cotangent of a repeated checkpoint output; the
    shared_first block returns its KV as both new cache and shared view, and
    a full-length cache slice is the same node without being the same object."""
    from chronos.mlx.checkpointing import call_block

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.array([0.5, -1.5])

        def __call__(self, x):
            y = mx.sin(x * self.w)
            # Same object, a full-length slice and a same-dtype cast: in mlx
            # 0.32.4 all three are the same graph node as ``y``.
            return y, (y, y[-10:], y.astype(y.dtype), 2)

    block, x = Twice(), mx.array([0.3, 1.2])

    def loss(m, x, wrap):
        out, (again, sliced, cast, k) = call_block(m, x) if wrap else m(x)
        assert k == 2
        return (out + 3 * again + 5 * sliced + 7 * cast).sum()

    ref = nn.value_and_grad(block, lambda m, x: loss(m, x, False))(block, x)[1]["w"]
    got = nn.value_and_grad(block, lambda m, x: loss(m, x, True))(block, x)[1]["w"]
    assert float(mx.abs(ref - got).max()) < 1e-6 and float(mx.abs(ref).max()) > 0


def test_recomputation_skips_cached_calls_and_inference():
    """A block that receives a KV cache (GRPO rollouts in train mode) and any
    eval-mode forward run eagerly and return real caches."""
    cfg = _config(activation_checkpointing=True, dropout=0.0, fallback_mask_prob=0.0)
    model = _model(cfg)
    ids, _ = _batch(cfg, b=1, s=6)
    full, _, _ = model(ids)
    _, _, cache = model(ids[:, :5])
    step, _, _ = model(ids[:, 5:], cache=cache)
    assert float(mx.abs(step[:, -1] - full[:, -1]).max()) < 1e-4
    model.eval()
    ref, _, _ = model(ids)
    model.config.activation_checkpointing = False
    assert float(mx.abs(ref - model(ids)[0]).max()) == 0.0


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("case", ["shared_first", "per_iteration", "router_shared", "dense_attention", "block_order",
      "later_topk", "later_topk_shared"])
def test_compiled_step_matches_eager(case, checkpointing):
    from chronos.mlx.training.compiled import CompiledStep
    cfg = _config(**CASES[case], activation_checkpointing=checkpointing)
    model = _model(cfg)
    ids, labels = _batch(cfg)
    eager = nn.value_and_grad(model, _loss)
    step = CompiledStep(model, _loss, mode="on")
    for seed in (3, 4):                                        # trace, then replay
        _assert_same(_step(model, eager, ids, labels, seed), _step(model, step, ids, labels, seed), 1e-6)
    assert step.stats == {"compiled": 2, "eager": 0, "traces": 1}


def test_compiled_step_retraces_on_python_decisions():
    """Shapes, the ACT switch (training progress) and a toggled config field
    each select their own graph; replays never reuse a stale one."""
    from chronos.mlx.training.compiled import CompiledStep
    cfg = _config(depth_schedule="fixed_then_adaptive", depth_schedule_warmup_fraction=0.5)
    model = _model(cfg)
    eager = nn.value_and_grad(model, _loss)
    step = CompiledStep(model, _loss, mode="on")
    controller = model.recurrent_controller
    for progress, shape, coef in ((0.1, (2, 12), 0.0), (0.9, (2, 12), 0.0), (0.9, (3, 8), 0.0),
                                  (0.9, (2, 12), 0.01), (0.1, (2, 12), 0.0)):
        controller.set_training_progress(progress, 1.0)
        model.config.recurrent_distill_coef = coef
        ids, labels = _batch(cfg, *shape)
        _assert_same(_step(model, eager, ids, labels), _step(model, step, ids, labels), 1e-6)
    assert step.stats["traces"] == 4 and step.stats["compiled"] == 5


def test_compiled_step_pins_one_sampled_depth_per_step():
    """With depth jitter the depth is drawn once (Python RNG) and the graph
    for that depth is used, so eager and compiled see the same depth."""
    import random
    from chronos.mlx.training.compiled import CompiledStep
    cfg = _config(recurrent_depth=4, min_recurrent_depth=1, recurrent_depth_jitter=1.0,
                  depth_schedule_warmup_fraction=0.0)
    model = _model(cfg)
    model.recurrent_controller.set_training_progress(1.0, 1.0)
    ids, labels = _batch(cfg)
    eager = nn.value_and_grad(model, _loss)
    step = CompiledStep(model, _loss, mode="on")
    for seed in range(6):
        random.seed(seed)
        a = _step(model, eager, ids, labels, seed)
        random.seed(seed)
        b = _step(model, step, ids, labels, seed)
        _assert_same(a, b, 1e-6)
    assert step.stats["traces"] >= 2                         # several depths were drawn


def test_compiled_step_falls_back_for_unsupported_inputs():
    from chronos.mlx.training.compiled import CompiledStep
    cfg = _config()
    model = _model(cfg)
    lines = []
    step = CompiledStep(model, lambda m, batch: _loss(m, *batch), mode="auto", log=lines.append)
    ids, labels = _batch(cfg)
    loss, _ = step(model, (ids, labels))
    mx.eval(loss)
    assert step.stats["eager"] == 1 and lines == ["[mlx-compile] eager step: non-array batch argument"]
    with pytest.raises(ValueError):
        CompiledStep(model, _loss, mode="sometimes")
