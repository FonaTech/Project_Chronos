"""tools/bench_speculative.mlx_runner paging layouts and tools/bench_residency page-read accounting."""
from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


@pytest.fixture
def trained_ckpt(tmp_path, monkeypatch):
    from test_lazy_experts import _save
    monkeypatch.setenv("CHRONOS_EXPERT_CACHE_ROOT", str(tmp_path / "cache"))
    return _save(tmp_path, 0.3, "trained")


def _runner_store(ckpt, workdir, paging):
    import mlx.core as mx
    from chronos.expert_clusters import inference_clusters
    from chronos.model.checkpoint import chronos_config_from_checkpoint
    from tools.bench_speculative import mlx_runner
    from ui.tabs.inference_tab import _clone_model_cfg
    cfg, _ = chronos_config_from_checkpoint(ckpt)
    cfg = _clone_model_cfg(cfg)
    with mx.stream(mx.cpu):
        _decode, _info, close = mlx_runner(cfg, ckpt, lazy=True, ratio=0.5, device="cpu", workdir=str(workdir), paging=paging)
        engine = close.__self__
        stats = engine.store.stats()
        engine.stop()
    return stats, len(inference_clusters(cfg, "cluster")), int(cfg.num_experts)


@pytest.mark.parametrize("paging,per_expert", [("expert", True), ("cluster", False), ("auto", False)])
def test_mlx_runner_pages_by_expert_or_by_training_cluster(trained_ckpt, tmp_path, paging, per_expert):
    pytest.importorskip("mlx.core")
    stats, n_clusters, n_experts = _runner_store(trained_ckpt, tmp_path / paging, paging)
    assert n_clusters < n_experts
    assert stats["num_clusters"] == (n_experts if per_expert else n_clusters)
