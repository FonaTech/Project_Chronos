"""The MLX step-size clip has the same gradient as Torch's clamp, including on the bounds."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def _cfg(lo, hi, init):
    from chronos.model.config import ChronosConfig
    return ChronosConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        num_experts=4, num_experts_per_tok=2, num_shared_experts=1, moe_intermediate_size=16, max_position_embeddings=64,
        recurrent_depth=2, prelude_layers=1, coda_layers=1, dropout=0.0, kv_latent_dim=8, rope_dim=4, sliding_window_size=32,
        use_moe=True, recurrent_dt_min=lo, recurrent_dt_max=hi, recurrent_dt_init=init)


@pytest.mark.parametrize("raw", [0.5, 0.5001, 0.75, 0.9999, 1.0, 1.0001, 0.4999])
def test_mlx_contract_gradient_matches_torch_clamp(raw):
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.recurrent import RecurrentControllerMLX as MlxController
    from chronos.model.recurrent import RecurrentController as TorchController
    cfg = _cfg(0.5, 1.0, 0.75)
    before, after = np.full((1, 1, 4), 0.2, dtype=np.float32), np.full((1, 1, 4), 1.0, dtype=np.float32)
    torch_ctl = TorchController(cfg)
    with torch.no_grad():
        torch_ctl.step_sizes[:] = raw
    out = torch_ctl.contract(torch.tensor(before), torch.tensor(after), 0)
    out.sum().backward()
    expected_grad = float(torch_ctl.step_sizes.grad[0])
    expected_out = out.detach().numpy()
    with mx.stream(mx.cpu):
        mlx_ctl = MlxController(cfg)
        mlx_ctl.step_sizes = mx.full(mlx_ctl.step_sizes.shape, raw)
        def f(sizes):
            mlx_ctl.step_sizes = sizes
            return mlx_ctl.contract(mx.array(before), mx.array(after), 0, cfg).sum()
        value, grad = mx.value_and_grad(f)(mlx_ctl.step_sizes)
        np.testing.assert_allclose(np.array(mlx_ctl.contract(mx.array(before), mx.array(after), 0, cfg)), expected_out, atol=1e-6)
        assert float(grad[0]) == pytest.approx(expected_grad)
