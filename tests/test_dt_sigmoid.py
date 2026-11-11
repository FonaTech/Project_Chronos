"""Sigmoid step-size parameterisation: same start as clamp, live gradient in the old dead zone, Torch == MLX."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _cfg(param, init=0.25, lo=0.05, hi=1.0):
    from chronos.model.config import ChronosConfig
    return ChronosConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1, moe_intermediate_size=16, max_position_embeddings=64,
        recurrent_depth=2, prelude_layers=1, coda_layers=1, dropout=0.0, kv_latent_dim=8, rope_dim=4, sliding_window_size=32,
        use_moe=True, recurrent_dt_min=lo, recurrent_dt_max=hi, recurrent_dt_init=init, recurrent_dt_param=param)


def test_config_rejects_unknown_param():
    with pytest.raises(ValueError):
        _cfg("tanh")


@pytest.mark.parametrize("init", [0.05, 0.0625, 0.25, 0.5, 1.0])
def test_sigmoid_starts_at_configured_dt(init):
    from chronos.model.recurrent import RecurrentController
    clamp = RecurrentController(_cfg("clamp", init)).step_size(0).item()
    sigmoid = RecurrentController(_cfg("sigmoid", init)).step_size(0).item()
    assert sigmoid == pytest.approx(clamp, abs=1e-5)


def test_sigmoid_gradient_alive_where_clamp_is_dead():
    from chronos.model.recurrent import RecurrentController
    before, after = torch.full((1, 1, 4), 0.2), torch.full((1, 1, 4), 1.0)
    grads = {}
    for param in ("clamp", "sigmoid"):
        ctl = RecurrentController(_cfg(param, init=0.05))
        if param == "clamp":
            with torch.no_grad():
                ctl.step_sizes[:] = 0.03   # below dt_min: the clamp dead zone
        ctl.contract(before, after, 0).sum().backward()
        grads[param] = float(ctl.step_sizes.grad[0])
    assert grads["clamp"] == 0.0
    assert grads["sigmoid"] > 0.0


def test_recommended_dt_init_holds_the_horizon_fixed():
    from chronos.model.recurrent import recommended_dt_init
    assert recommended_dt_init(4) == pytest.approx(0.5)
    assert recommended_dt_init(6) == pytest.approx(1 / 3)
    assert recommended_dt_init(8) == pytest.approx(0.25)
    for depth in (4, 6, 8, 12, 16):
        assert depth * recommended_dt_init(depth) == pytest.approx(2.0)        # T = R * dt
    assert recommended_dt_init(8) < recommended_dt_init(6) < recommended_dt_init(4)


def test_recommended_dt_init_is_capped_where_it_was_never_measured():
    from chronos.model.recurrent import recommended_dt_init
    assert recommended_dt_init(1) == recommended_dt_init(2) == recommended_dt_init(3) == 0.5
    assert recommended_dt_init(2, cap=1.0) == pytest.approx(1.0)
    assert recommended_dt_init(8, horizon=1.0) == pytest.approx(0.125)


@pytest.mark.parametrize("raw", [-6.0, -1.0, 0.0, 2.5])
def test_mlx_sigmoid_matches_torch(raw):
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.recurrent import RecurrentControllerMLX
    from chronos.model.recurrent import RecurrentController
    cfg = _cfg("sigmoid")
    before, after = np.full((1, 1, 4), 0.2, dtype=np.float32), np.full((1, 1, 4), 1.0, dtype=np.float32)
    torch_ctl = RecurrentController(cfg)
    with torch.no_grad():
        torch_ctl.step_sizes[:] = raw
    out = torch_ctl.contract(torch.tensor(before), torch.tensor(after), 0)
    out.sum().backward()
    with mx.stream(mx.cpu):
        mlx_ctl = RecurrentControllerMLX(cfg)
        assert float(mlx_ctl.step_sizes[0]) == pytest.approx(float(RecurrentController(cfg).step_sizes.detach()[0]), abs=1e-6)

        def f(sizes):
            mlx_ctl.step_sizes = sizes
            return mlx_ctl.contract(mx.array(before), mx.array(after), 0, cfg).sum()

        value, grad = mx.value_and_grad(f)(mx.full(mlx_ctl.step_sizes.shape, raw))
        assert float(value) == pytest.approx(float(out.sum()), abs=1e-5)
        assert float(grad[0]) == pytest.approx(float(torch_ctl.step_sizes.grad[0]), rel=1e-4, abs=1e-7)
