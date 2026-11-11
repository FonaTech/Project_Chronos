"""Iteration-axis cluster locality (lambda_iteration_cluster) and the lookahead teacher scope.

The term keeps every loop iteration of a token inside the same expert clusters without fixing which expert
inside a cluster runs, so a lazy store that pages whole clusters serves all D iterations with one read.
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
from chronos.model.temporal_loss import iteration_cluster_spread  # noqa: E402

E, C = 8, 4


def _membership():
    m = np.zeros((E, C), np.float32)
    for e in range(E):
        m[e, e // (E // C)] = 1.0
    return m


def _probs(seed=0, b=2, s=5, g=3, d=3):
    rng = np.random.default_rng(seed)
    p = rng.random((b, s, g, d, E)).astype(np.float32) + 0.05
    return p / p.sum(-1, keepdims=True)


def test_config_fields_default_off_and_validate():
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4)
    assert cfg.lambda_iteration_cluster == 0.0 and cfg.lookahead_teacher_scope == "first"
    assert ChronosConfig(hidden_size=64, num_hidden_layers=4, lookahead_teacher_scope="ITERATIONS").lookahead_teacher_scope == "iterations"
    with pytest.raises(ValueError):
        ChronosConfig(hidden_size=64, num_hidden_layers=4, lookahead_teacher_scope="all")
    from chronos.config_schema import MODEL_FIELDS
    assert "lambda_iteration_cluster" in MODEL_FIELDS and "lookahead_teacher_scope" in MODEL_FIELDS


def test_spread_is_zero_when_iterations_agree_and_positive_otherwise():
    same = np.repeat(_probs(d=1), 3, axis=3)
    assert float(iteration_cluster_spread(torch.tensor(same), torch.tensor(_membership()))) == pytest.approx(0.0, abs=1e-7)
    # iterations that differ only inside a cluster cost nothing
    inside = np.zeros((1, 1, 1, 2, E), np.float32)
    inside[0, 0, 0, 0, 0] = 1.0
    inside[0, 0, 0, 1, 1] = 1.0 # experts 0 and 1 share cluster 0
    assert float(iteration_cluster_spread(torch.tensor(inside), torch.tensor(_membership()))) == pytest.approx(0.0, abs=1e-7)
    # iterations in different clusters cost the maximum: total variation 0.5 from the mean
    across = np.zeros((1, 1, 1, 2, E), np.float32)
    across[0, 0, 0, 0, 0] = 1.0
    across[0, 0, 0, 1, 4] = 1.0        # cluster 0 vs cluster 2
    assert float(iteration_cluster_spread(torch.tensor(across), torch.tensor(_membership()))) == pytest.approx(0.5, abs=1e-6)


def test_spread_needs_two_iterations_and_two_clusters():
    one = torch.tensor(_probs(d=1))
    assert float(iteration_cluster_spread(one, torch.tensor(_membership()))) == 0.0
    assert float(iteration_cluster_spread(torch.tensor(_probs()), torch.ones(E, 1))) == 0.0


def test_token_mask_drops_padding_positions():
    p = _probs(b=1, s=4, g=1, d=2)
    p[0, 2:] = 0.0
    p[0, 2:, 0, 0, 0], p[0, 2:, 0, 1, 4] = 1.0, 1.0        # padding tokens that maximise the spread
    mask = torch.tensor([[1, 1, 0, 0]])
    full = float(iteration_cluster_spread(torch.tensor(p), torch.tensor(_membership())))
    real = float(iteration_cluster_spread(torch.tensor(p), torch.tensor(_membership()), mask))
    first_two = float(iteration_cluster_spread(torch.tensor(p[:, :2]), torch.tensor(_membership())))
    assert real == pytest.approx(first_two, abs=1e-7) and full > real


def test_mlx_matches_torch():
    from chronos.mlx.training.trainer import _iteration_cluster_spread
    p = _probs(seed=3)
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], np.float32)
    for m_np, m_t in ((None, None), (mask, torch.tensor(mask))):
        want = float(iteration_cluster_spread(torch.tensor(p), torch.tensor(_membership()), m_t))
        with mx.stream(mx.cpu):
            got = float(_iteration_cluster_spread(mx.array(p), mx.array(_membership()), None if m_np is None else mx.array(m_np)))
        assert got == pytest.approx(want, abs=1e-6)


def _mlx_model(**kw):
    from chronos.mlx.model import ChronosMLXModel
    base = dict(architecture_version=3, hidden_size=64, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
        recurrent_depth=3, min_recurrent_depth=2, num_attention_heads=4, num_key_value_heads=2, kv_latent_dim=16,
        rope_dim=8, num_experts=8, num_experts_per_tok=2, num_shared_experts=1, moe_intermediate_size=32,
        vocab_size=97, max_position_embeddings=64, sliding_window_size=64, dropout=0.0, fallback_mask_prob=0.0,
        adaptive_halting=False, lambda_iteration_cluster=0.5, cluster_locality_clusters=C)
    base.update(kw)
    cfg = ChronosConfig(**base)
    mx.random.seed(0)
    model = ChronosMLXModel(cfg)
    mx.eval(model.parameters())
    model.train()
    return cfg, model


@pytest.mark.parametrize("order", ["layer", "block"])
def test_mlx_training_loss_contains_the_term_and_trains_the_routers(order):
    from chronos.mlx.training.trainer import _chronos_regularized_loss
    cfg, model = _mlx_model(recurrent_order=order)
    ids = mx.random.randint(3, 97, (2, 10))
    labels = mx.random.randint(3, 97, (2, 10))
    
    def loss_fn(m, x, y):
        logits, look, _ = m(x, logits_dtype=mx.float32)
        ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")
        return _chronos_regularized_loss(m, ce, look, m.config, metrics={"ce": ce, "primary": ce})
    
    with mx.stream(mx.cpu):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, ids, labels)
        mx.eval(loss, grads)
        comps = {k: float(v) for k, v in model._last_loss_components.items()}
        assert "iteration_cluster_loss" in comps and comps["iteration_cluster_loss"] > 0.0
        from mlx.utils import tree_flatten
        gates = [float(mx.abs(g).sum()) for k, g in tree_flatten(grads) if k.endswith("mlp.gate.weight")]
        assert gates and max(gates) > 0.0


def test_mlx_term_is_absent_when_off_or_without_iterations():
    from chronos.mlx.training.trainer import _chronos_regularized_loss
    for kw in (dict(lambda_iteration_cluster=0.0), dict(recurrent_depth=1)):
        cfg, model = _mlx_model(**kw)
        ids = mx.random.randint(3, 97, (2, 10))
        with mx.stream(mx.cpu):
            logits, look, _ = model(ids, logits_dtype=mx.float32)
            ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction="mean")
            _chronos_regularized_loss(model, ce, look, model.config, metrics={"ce": ce, "primary": ce})
        assert "iteration_cluster_loss" not in model._last_loss_components


def test_mlx_compiled_step_matches_eager_with_the_term():
    from chronos.mlx.training.compiled import CompiledStep
    from chronos.mlx.training.trainer import _chronos_regularized_loss
    cfg, model = _mlx_model(recurrent_order="block", lambda_cluster_locality=0.3)
    ids = mx.random.randint(3, 97, (2, 12))
    labels = mx.random.randint(3, 97, (2, 12))
    
    def loss_fn(m, x, y):
        logits, look, _ = m(x, logits_dtype=mx.float32)
        ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="mean")
        return _chronos_regularized_loss(m, ce, look, m.config, metrics={"ce": ce, "primary": ce})
    
    with mx.stream(mx.cpu):
        eager = nn.value_and_grad(model, loss_fn)
        step = CompiledStep(model, loss_fn, mode="on")
        a, ga = eager(model, ids, labels)
        b, gb = step(model, ids, labels)
        mx.eval(a, b, ga, gb)
        assert abs(float(a) - float(b)) < 1e-6
        from mlx.utils import tree_flatten
        fa, fb = dict(tree_flatten(ga)), dict(tree_flatten(gb))
        assert max(float(mx.abs(fa[k] - fb[k]).max()) for k in fa) < 1e-6


def test_torch_loss_mixin_adds_the_term():
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.trainer.loss_mixin import chronos_loss_term, recurrent_iteration_probs
    cfg = ChronosConfig(architecture_version=3, hidden_size=64, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
        recurrent_depth=3, min_recurrent_depth=2, num_attention_heads=4, num_key_value_heads=2, kv_latent_dim=16,
        rope_dim=8, num_experts=8, num_experts_per_tok=2, num_shared_experts=1, moe_intermediate_size=32,
        vocab_size=97, max_position_embeddings=64, sliding_window_size=64, dropout=0.0, fallback_mask_prob=0.0,
        adaptive_halting=False, lambda_iteration_cluster=0.0, cluster_locality_clusters=C)
    torch.manual_seed(0)
    model = ChronosForCausalLM(cfg).train()
    ids = torch.randint(3, 97, (2, 10))
    out = model(ids, labels=ids)
    probs = recurrent_iteration_probs(model, cfg)
    assert probs is not None and probs.shape[2:4] == (int(cfg.recurrent_groups), 3)
    base = chronos_loss_term(model, out.loss, None, cfg, input_ids=ids, labels=ids)
    cfg.lambda_iteration_cluster = 0.5
    with_term = chronos_loss_term(model, out.loss, None, cfg, input_ids=ids, labels=ids)
    assert float(with_term - base) > 0.0
    with_term.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters() if "gate" in n)


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_lookahead_teacher_scope_changes_the_target_not_the_default(backend):
    if backend == "torch":
        from chronos.model.model_chronos import ChronosForCausalLM
        from chronos.trainer.loss_mixin import first_iteration_router_probs, iteration_mean_router_probs
        cfg = ChronosConfig(architecture_version=3, hidden_size=64, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
            recurrent_depth=3, min_recurrent_depth=2, num_attention_heads=4, num_key_value_heads=2, kv_latent_dim=16,
            rope_dim=8, num_experts=8, num_experts_per_tok=2, num_shared_experts=1, moe_intermediate_size=32,
            vocab_size=97, max_position_embeddings=64, sliding_window_size=64, dropout=0.0, fallback_mask_prob=0.0)
        torch.manual_seed(0)
        model = ChronosForCausalLM(cfg).eval()
        with torch.no_grad():
            model(torch.randint(3, 97, (1, 8)))
        first, mean = first_iteration_router_probs(model), iteration_mean_router_probs(model)
        diff = float((first - mean).abs().max())
    else:
        from chronos.mlx.training.trainer import _first_iteration_router_probs, _iteration_mean_router_probs
        cfg, model = _mlx_model()
        with mx.stream(mx.cpu):
            model(mx.random.randint(3, 97, (1, 8)))
            first, mean = _first_iteration_router_probs(model), _iteration_mean_router_probs(model)
            diff = float(mx.abs(first - mean).max())
    assert first.shape == mean.shape
    assert diff > 1e-4   # later iterations route differently, so the two teachers are genuinely different
