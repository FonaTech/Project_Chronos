"""Cluster objective of 0.3.5: the budget objective, dispatch mass and the gate-only (detached) cluster gradient,
with Torch/MLX parity, the config defaults and the pre-0.3.5 checkpoint backfill."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chronos.expert_clusters import cluster_budget
from chronos.model.config import ChronosConfig
from chronos.model.temporal_loss import cluster_locality_loss, dispatch_mass, segment_cluster_loss

E = 12


@pytest.fixture
def mx():
    """MLX on the CPU: the GPU matmul rounds fp32 inputs to a TF32-class mantissa."""
    core = pytest.importorskip("mlx.core")
    old = core.default_device()
    core.set_default_device(core.cpu)
    yield core
    core.set_default_device(old)


def _router(seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(2, 16, 3, E)) * 2.0
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)).astype(np.float32)


def _membership(c):
    return np.array([[1.0 if e * c // E == j else 0.0 for j in range(c)] for e in range(E)], np.float32)


def _window_pw(r, memb, window):
    B, S, V, _ = r.shape
    mass = np.einsum("bsve,ec->bsvc", r, memb)
    return mass.reshape(B, S // window, window, V, -1).mean(axis=(2, 3)).reshape(B * (S // window), -1)


def _cfg(**kw):
    base = dict(hidden_size=32, num_experts=E, num_experts_per_tok=2, moe_intermediate_size=16, num_shared_experts=1,
                cluster_locality_window=8)
    base.update(kw)
    return ChronosConfig(**base)


def test_defaults_and_budget_rule():
    cfg = ChronosConfig()
    assert (cfg.cluster_locality_objective, cfg.cluster_locality_budget, cfg.cluster_locality_mass,
            cfg.cluster_input_detach) == ("budget", 0, "dispatch", True)
    assert cluster_budget(_cfg()) == 2                                  # ceil(E / 2k) = 3 clusters, half rounded up
    assert cluster_budget(_cfg(cluster_locality_clusters=4)) == 2
    assert cluster_budget(_cfg(cluster_locality_clusters=6)) == 3
    assert cluster_budget(_cfg(cluster_locality_clusters=4, cluster_locality_budget=3)) == 3
    assert cluster_budget(_cfg(cluster_locality_clusters=4, cluster_locality_budget=9)) == 4
    assert cluster_budget(_cfg(cluster_locality_objective="mutual_info")) == 0
    assert cluster_budget(SimpleNamespace(num_experts=E, num_experts_per_tok=2)) == 0     # predates the fields
    for bad in (dict(cluster_locality_objective="x"), dict(cluster_locality_mass="x"), dict(cluster_locality_budget=-1)):
        with pytest.raises(ValueError):
            ChronosConfig(**bad)
    assert ChronosConfig(cluster_input_detach=False).cluster_input_detach is False
    assert ChronosConfig(cluster_input_detach=None).cluster_input_detach is True


def test_fields_survive_the_schema():
    from chronos.config_schema import MODEL_FIELDS, model_config_kwargs
    raw = dict(cluster_locality_objective="mutual_info", cluster_locality_budget=3, cluster_locality_mass="soft",
               cluster_input_detach=False)
    assert set(raw) <= set(MODEL_FIELDS)
    kept = model_config_kwargs(raw)
    assert all(kept[k] == v for k, v in raw.items())


def test_pre_035_checkpoint_keeps_the_old_objective(tmp_path):
    from chronos.model.checkpoint import checkpoint_config_path, read_checkpoint_config
    ckpt = str(tmp_path / "model.pth")
    old = {"num_experts": E, "lambda_cluster_locality": 0.3, "recurrent_enabled": True}
    with open(checkpoint_config_path(ckpt), "w") as fh:
        json.dump({"config": old, "checkpoint_config_version": 2}, fh)
    cfg, _ = read_checkpoint_config(ckpt)
    assert (cfg["cluster_locality_objective"], cfg["cluster_locality_mass"], cfg["cluster_input_detach"]) == (
        "mutual_info", "soft", False)
    new = dict(old, cluster_locality_objective="budget")
    with open(checkpoint_config_path(ckpt), "w") as fh:
        json.dump({"config": new, "checkpoint_config_version": 2}, fh)
    cfg, _ = read_checkpoint_config(ckpt)
    assert cfg["cluster_locality_objective"] == "budget" and "cluster_input_detach" not in cfg


def test_budget_is_balance_plus_spill():
    r, memb = _router(3), _membership(4)
    pw = _window_pw(r, memb, 8)
    marg = pw.mean(axis=0)
    balance = 1.0 + float((marg * np.log(marg)).sum()) / np.log(4)
    for budget in (1, 2, 3, 4):
        spill = float((1.0 - np.sort(pw, axis=-1)[:, -budget:].sum(axis=-1)).mean())
        got = float(cluster_locality_loss(torch.tensor(r), torch.tensor(memb), 8, budget=budget))
        assert got == pytest.approx(balance + spill, abs=1e-5)
    assert float(cluster_locality_loss(torch.tensor(r), torch.tensor(memb), 8, budget=4)) == pytest.approx(balance, abs=1e-5)


def test_budget_leaves_the_split_inside_the_resident_clusters_free():
    memb = _membership(4)
    r = np.zeros((2, 16, 3, E), np.float32)
    for b in range(2):
        for w in range(2):
            pair = (0, 1) if (b + w) % 2 == 0 else (2, 3)
            experts = [e for e in range(E) if e * 4 // E in pair]
            r[b, w * 8:(w + 1) * 8, :, experts] = 1.0 / len(experts)
    rt = torch.tensor(r)
    assert float(cluster_locality_loss(rt, torch.tensor(memb), 8, budget=2)) == pytest.approx(0.0, abs=1e-5)
    assert float(cluster_locality_loss(rt, torch.tensor(memb), 8)) == pytest.approx(0.5, abs=1e-4)      # mutual info
    assert float(cluster_locality_loss(rt, torch.tensor(memb), 8, budget=1)) == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize("detach", [False, True])
def test_torch_detach_reaches_the_gate_only(detach):
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    torch.manual_seed(0)
    moe = ChronosMOEFeedForward(_cfg(cluster_input_detach=detach)).train()
    x = torch.randn(2, 16, 32, requires_grad=True)
    moe(x)
    probs = moe.iteration_cluster_probs_grad if detach else moe.iteration_router_probs_grad
    assert len(probs) == 1
    loss = cluster_locality_loss(torch.stack(probs, dim=2), torch.tensor(_membership(3)), 8, budget=2)
    gate, gx = torch.autograd.grad(loss, [moe.gate.weight, x], allow_unused=True)
    assert float(gate.abs().sum()) > 0.0
    assert (gx is None or float(gx.abs().sum()) == 0.0) == detach
    moe.reset_iteration_routing()
    moe.eval()
    with torch.no_grad():
        moe(torch.randn(1, 4, 32))
    assert moe.iteration_cluster_probs_grad == []                       # recorded in training only


def test_torch_cluster_source_reads_the_detached_dispatch_mass():
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    from chronos.trainer.loss_mixin import cluster_source_probs
    torch.manual_seed(0)
    cfg = _cfg()
    moe = ChronosMOEFeedForward(cfg).train()
    x = torch.randn(2, 16, 32, requires_grad=True)
    moe(x)
    router = torch.stack(moe.iteration_router_probs_grad, dim=2)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]))
    src = cluster_source_probs(model, cfg, router)
    assert torch.allclose(src, dispatch_mass(router, 2), atol=1e-6)
    assert ((src > 0).sum(-1) == 2).all()
    loss = cluster_locality_loss(src, torch.tensor(_membership(3)), 8, budget=cluster_budget(cfg))
    gate, gx = torch.autograd.grad(loss, [moe.gate.weight, x], allow_unused=True)
    assert float(gate.abs().sum()) > 0.0 and gx is None
    assert cluster_source_probs(model, _cfg(cluster_input_detach=False, cluster_locality_mass="soft"), router) is router


def test_mlx_dispatch_mass_and_budget_match_torch(mx):
    from chronos.mlx.training.trainer import _cluster_locality_loss, _dispatch_mass, _segment_cluster_loss
    r = _router(2)
    assert np.allclose(dispatch_mass(torch.tensor(r), 2).numpy(), np.array(_dispatch_mass(mx.array(r), 2)), atol=1e-6)
    memb = _membership(4)
    seg = np.array([[0] * 8 + [1] * 8, [0] * 5 + [1] * 11])
    mask = np.ones((2, 16), np.float32)
    mask[1, 14:] = 0.0
    for budget in (1, 2, 3, 4):
        t_win = float(cluster_locality_loss(torch.tensor(r), torch.tensor(memb), 8, budget=budget))
        m_win = float(_cluster_locality_loss(mx.array(r), mx.array(memb), 8, None, budget=budget))
        m_ones = float(_cluster_locality_loss(mx.array(r), mx.array(memb), 8, mx.ones((2, 16)), budget=budget))
        t_seg = float(segment_cluster_loss(torch.tensor(r), torch.tensor(memb), seg, torch.tensor(mask), budget=budget))
        m_seg = float(_segment_cluster_loss(mx.array(r), mx.array(memb), mx.array(seg), mx.array(mask), budget=budget))
        assert t_win == pytest.approx(m_win, abs=1e-5)
        assert m_ones == pytest.approx(m_win, abs=1e-5)
        assert t_seg == pytest.approx(m_seg, abs=1e-5)


@pytest.mark.parametrize("detach", [False, True])
def test_mlx_objective_reads_the_detached_gate(mx, detach):
    from chronos.mlx.moe import ChronosMLXMOE
    from chronos.mlx.training.trainer import _cluster_locality_loss, _cluster_membership, _cluster_objective, _dispatch_mass
    cfg = _cfg(cluster_input_detach=detach)
    mx.random.seed(0)
    moe = ChronosMLXMOE(cfg)
    moe.train()
    model = SimpleNamespace(layers=[SimpleNamespace(mlp=moe)])
    x = mx.random.normal((2, 16, 32))

    def f(params, x):
        moe.update(params)
        moe.reset_iteration_routing()
        y = moe(x)
        router = mx.stack(moe.iteration_router_probs, axis=2)
        return _cluster_objective(router, cfg, None, None, model) + 0.0 * y.sum()

    value = f(moe.trainable_parameters(), x)
    router = mx.stack(moe.iteration_router_probs, axis=2)
    want = _cluster_locality_loss(_dispatch_mass(router, 2), _cluster_membership(cfg), 8, None, budget=cluster_budget(cfg))
    assert float(value) == pytest.approx(float(want), abs=1e-5)
    assert len(moe.iteration_cluster_probs) == (1 if detach else 0)
    gp, gx = mx.grad(f, argnums=(0, 1))(moe.trainable_parameters(), x)
    assert float(mx.abs(gp["gate"]["weight"]).sum()) > 0.0
    assert (float(mx.abs(gx).sum()) == 0.0) == detach


def test_mlx_mutual_info_soft_attached_is_the_034_objective(mx):
    from chronos.mlx.training.trainer import _cluster_locality_loss, _cluster_membership, _cluster_objective
    cfg = _cfg(cluster_locality_objective="mutual_info", cluster_locality_mass="soft", cluster_input_detach=False)
    r = mx.array(_router(4))
    want = float(_cluster_locality_loss(r, _cluster_membership(cfg), 8))
    assert float(_cluster_objective(r, cfg)) == pytest.approx(want, abs=1e-6)
    assert float(_cluster_objective(r, _cfg())) != pytest.approx(want, abs=1e-4)


@pytest.mark.parametrize("mode,recompute", [("window", False), ("segment", False), ("window", True)])
def test_mlx_compiled_step_matches_eager_with_the_035_defaults(mx, mode, recompute):
    """The gate-only record is read inside the compiled (and recomputed) step; loss and gradients match eager."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.compiled import make_grad_fn
    from chronos.mlx.training.trainer import _ce_stage_loss
    cfg = ChronosConfig(architecture_version=3, hidden_size=64, num_hidden_layers=4, prelude_layers=1, coda_layers=1,
        recurrent_depth=3, min_recurrent_depth=2, num_attention_heads=4, num_key_value_heads=2, kv_latent_dim=16,
        rope_dim=8, num_experts=6, num_experts_per_tok=2, num_shared_experts=1, moe_intermediate_size=32,
        vocab_size=97, max_position_embeddings=128, sliding_window_size=8, dropout=0.0, fallback_mask_prob=0.0,
        adaptive_halting=True, lookahead_steps=2, layer_aware_lookahead=True, lambda_cluster_locality=0.5,
        cluster_locality_window=8, cluster_locality_mode=mode, activation_checkpointing=recompute)
    assert cluster_budget(cfg) == 1 and cfg.cluster_input_detach
    mx.random.seed(1)
    ids = mx.random.randint(3, cfg.vocab_size, (2, 40))
    ids = mx.where(mx.arange(40)[None, :] % 11 == 10, 0, ids)
    labels = mx.random.randint(3, cfg.vocab_size, (2, 40))

    def build():
        mx.random.seed(0)
        model = ChronosMLXModel(cfg)
        mx.eval(model.parameters())
        model.train()
        return model

    model = build()
    eager_loss, eager_grads = nn.value_and_grad(model, _ce_stage_loss)(model, ids, labels)
    mx.eval(eager_loss, eager_grads)
    want = dict(tree_flatten(eager_grads))
    model2 = build()
    step = make_grad_fn(model2, _ce_stage_loss, mode="on", log=lambda m: None)
    loss, grads = step(model2, ids, labels)
    mx.eval(loss, grads)
    assert step.stats["compiled"] == 1 and step.stats["eager"] == 0
    assert abs(loss.item() - eager_loss.item()) < 1e-4
    for name, g in dict(tree_flatten(grads)).items():
        denom = float(mx.maximum(mx.abs(want[name]).max(), 1e-12).item())
        assert float(mx.abs(g - want[name]).max().item()) / denom < 1e-3, name
    assert model2._last_loss_components["cluster_locality_loss"].item() > 0.0
