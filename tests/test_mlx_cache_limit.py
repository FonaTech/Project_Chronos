"""Kernel-level MLX cache limit: min(RAM/8, 16 GiB); callers may only lower it."""
from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.mlx import memory  # noqa: E402

GIB = 1024 ** 3


@pytest.mark.parametrize("total_gib, want", [(16, 2), (64, 8), (128, 16), (512, 16)])
def test_ceiling_is_an_eighth_of_ram_capped_at_16_gib(total_gib, want):
    assert memory.ceiling_bytes(total_gib * GIB) == want * GIB


def test_requests_lower_but_never_raise_the_ceiling(monkeypatch):
    monkeypatch.delenv(memory.ENV, raising=False)
    total = 64 * GIB
    assert memory.resolve_cache_limit(None, total) == (8 * GIB, "default min(RAM/8, 16 GiB)")
    assert memory.resolve_cache_limit(4, total) == (4 * GIB, "recipe")
    limit, src = memory.resolve_cache_limit(12, total)
    assert limit == 8 * GIB and "clamped" in src
    monkeypatch.setenv(memory.ENV, "2.5")
    assert memory.resolve_cache_limit(None, total) == (int(2.5 * GIB), "env")
    for bad in (0, -1, "x"):
        with pytest.raises(ValueError):
            memory.resolve_cache_limit(bad, total)


def test_model_construction_applies_the_limit_and_longrun_records_it(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    monkeypatch.delenv(memory.ENV, raising=False)
    monkeypatch.setattr(memory, "_applied", None)
    before = mx.set_cache_limit(0)
    mx.set_cache_limit(before)
    try:
        from chronos.mlx.model import ChronosMLXModel
        from chronos.model.config import ChronosConfig
        cfg = ChronosConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            vocab_size=48, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
                            max_position_embeddings=32, prelude_layers=0, coda_layers=0)
        ChronosMLXModel(cfg)
        got = memory.applied()
        assert got is not None and got["limit_gb"] == pytest.approx(memory.ceiling_bytes() / GIB, abs=1e-3)
        assert mx.set_cache_limit(int(got["limit_gb"] * GIB)) == int(got["limit_gb"] * GIB)
        # A second model in the same process does not reset a caller's lower value.
        memory.apply_cache_limit(1, force=True)
        ChronosMLXModel(cfg)
        assert memory.applied()["limit_gb"] == pytest.approx(1.0)
    finally:
        mx.set_cache_limit(before)
        memory._applied = None
