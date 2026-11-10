"""Segment-mode cluster locality under mx.compile, and recovery from a failed trace (0.3.2a3)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip('mlx.core')
nn = pytest.importorskip('mlx.nn')
from mlx.utils import tree_flatten  # noqa: E402


def _config(**kw):
    from chronos.model.config import ChronosConfig
    base = dict(architecture_version=3, hidden_size=64, num_hidden_layers=4, prelude_layers=1, coda_layers=1,
        recurrent_depth=3, min_recurrent_depth=2, num_attention_heads=4, num_key_value_heads=2,
        kv_latent_dim=16, rope_dim=8, num_experts=6, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=32, vocab_size=97, max_position_embeddings=128, sliding_window_size=8,
        dropout=0.0, fallback_mask_prob=0.0, adaptive_halting=True, lookahead_steps=2,
        layer_aware_lookahead=True, lambda_cluster_locality=0.5, cluster_locality_window=8,
        cluster_locality_mode='segment')
    base.update(kw)
    return ChronosConfig(**base)


def _model(cfg):
    from chronos.mlx.model import ChronosMLXModel
    mx.random.seed(0)
    model = ChronosMLXModel(cfg)
    mx.eval(model.parameters())
    model.train()
    return model


def _batch(cfg, b=2, s=40):
    mx.random.seed(1)
    ids = mx.random.randint(3, cfg.vocab_size, (b, s))
    ids = mx.where(mx.arange(s)[None, :] % 11 == 10, 0, ids)       # several documents per row
    labels = mx.random.randint(3, cfg.vocab_size, (b, s))
    return ids, labels


def _live(*trees) -> bool:
    try:
        mx.eval(*trees)
        return True
    except Exception:
        return False


def test_segment_ids_mx_equals_numpy_with_chunks():
    from chronos.expert_clusters import segment_ids, segment_ids_mx
    rng = np.random.default_rng(7)
    for _ in range(120):
        b, s = int(rng.integers(1, 4)), int(rng.integers(2, 70))
        ids = rng.integers(0, 6, (b, s))
        chunk = int(rng.choice([0, 1, 3, 8, 256]))
        assert np.array_equal(np.array(segment_ids_mx(mx.array(ids), chunk=chunk)), segment_ids(ids, chunk))


def test_static_segment_loss_matches_torch_eager_and_compiled():
    torch = pytest.importorskip('torch')
    from chronos.expert_clusters import segment_ids, segment_ids_mx
    from chronos.mlx.training.trainer import _segment_cluster_loss
    from chronos.model.temporal_loss import segment_cluster_loss
    rng = np.random.default_rng(3)
    mem = mx.array(np.eye(6, 3, dtype=np.float32) + np.eye(6, 3, k=-3, dtype=np.float32))
    for _ in range(25):
        b, s = int(rng.integers(1, 4)), int(rng.integers(4, 48))
        ids = rng.integers(0, 5, (b, s))
        r = mx.softmax(mx.array(rng.normal(size=(b, s, 3, 6)).astype(np.float32)), axis=-1)
        mask = mx.array(rng.random((b, s)) > 0.2)
        chunk = int(rng.choice([0, 4, 16]))
        want = float(segment_cluster_loss(torch.tensor(np.array(r)), torch.tensor(np.array(mem)),
            segment_ids(ids, chunk), torch.tensor(np.array(mask))))
        eager = float(_segment_cluster_loss(r, mem, mx.array(segment_ids(ids, chunk)), mask))
        traced = float(mx.compile(lambda rr, ii, mm: _segment_cluster_loss(
            rr, mem, segment_ids_mx(ii, chunk=chunk), mm))(r, mx.array(ids), mask))
        assert eager == pytest.approx(want, abs=1e-4)
        assert traced == pytest.approx(want, abs=1e-4)


def test_single_segment_batch_has_no_locality_loss():
    from chronos.mlx.training.trainer import _segment_cluster_loss
    r = mx.softmax(mx.random.normal((2, 8, 2, 6)), axis=-1)
    mem = mx.array(np.eye(6, 3, dtype=np.float32) + np.eye(6, 3, k=-3, dtype=np.float32))
    one_row = mx.zeros((1, 8), dtype=mx.int32)
    assert float(_segment_cluster_loss(r[:1], mem, one_row)) == 0.0


def test_segment_mode_runs_in_the_compiled_step_and_matches_eager():
    from chronos.mlx.training.compiled import make_grad_fn
    from chronos.mlx.training.trainer import _ce_stage_loss, _collect_router_probs
    cfg = _config()
    ids, labels = _batch(cfg)
    model = _model(cfg)
    eager_loss, eager_grads = nn.value_and_grad(model, _ce_stage_loss)(model, ids, labels)
    mx.eval(eager_loss, eager_grads)
    want = dict(tree_flatten(eager_grads))
    model2 = _model(cfg)
    step = make_grad_fn(model2, _ce_stage_loss, mode='on', log=lambda m: None)
    loss, grads = step(model2, ids, labels)
    mx.eval(loss, grads)
    assert step.stats['compiled'] == 1 and step.stats['eager'] == 0
    assert abs(loss.item() - eager_loss.item()) < 1e-4
    for name, g in dict(tree_flatten(grads)).items():
        denom = float(mx.maximum(mx.abs(want[name]).max(), 1e-12).item())
        assert float((mx.abs(g - want[name]).max()).item()) / denom < 1e-3, name
    assert _live(_collect_router_probs(model2))
    assert model2._last_loss_components['cluster_locality_loss'].item() > 0.0


def _failing_loss(model, ids, labels):
    from chronos.mlx.training.trainer import _ce_stage_loss
    loss = _ce_stage_loss(model, ids, labels)
    np.asarray(ids)        # a host read of a traced array: always fails while tracing
    return loss


def test_failed_trace_leaves_weights_rng_and_side_outputs_usable():
    """A trace that raises used to leave tracers in the weights, the RNG key and the router outputs;
    the eager fallback then ran on them and the next mx.eval raised 'array without a primitive'."""
    from chronos.mlx.training.compiled import CompiledStep
    from chronos.mlx.training.trainer import _collect_router_probs
    cfg = _config(dropout=0.1, fallback_mask_prob=0.2)
    ids, labels = _batch(cfg)
    lines: list = []
    model = _model(cfg)
    mx.random.seed(11)
    ref_loss, ref_grads = nn.value_and_grad(model, _failing_loss_eager)(model, ids, labels)
    mx.eval(ref_loss, ref_grads)
    ref_next = mx.random.normal((4,))
    mx.eval(ref_next)
    model2 = _model(cfg)
    step = CompiledStep(model2, _failing_loss, mode='auto', log=lines.append)
    mx.random.seed(11)
    loss, grads = step(model2, ids, labels)
    assert any('compile failed' in line for line in lines)
    assert _live(model2.parameters(), mx.random.state, loss, grads)
    assert _live(_collect_router_probs(model2), list(model2._last_loss_components.values()))
    assert abs(loss.item() - ref_loss.item()) < 1e-6
    nxt = mx.random.normal((4,))
    mx.eval(nxt)
    assert np.array_equal(np.array(nxt), np.array(ref_next))     # the RNG stream continues exactly
    again_loss, _ = step(model2, ids, labels)          # a second step takes the eager path
    assert _live(again_loss) and step.stats['eager'] == 2


def _failing_loss_eager(model, ids, labels):
    from chronos.mlx.training.trainer import _ce_stage_loss
    return _ce_stage_loss(model, ids, labels)
