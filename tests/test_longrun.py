"""Time-budgeted MLX run: deterministic resume and data sources."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip("mlx.core")


def _tiny_shards(root):
    os.makedirs(os.path.join(root, "pretrain", "train"))
    os.makedirs(os.path.join(root, "pretrain", "valid"))
    rng = np.random.default_rng(0)
    rng.integers(3, 120, 20000).astype(np.uint16).tofile(os.path.join(root, "pretrain", "train", "a.bin"))
    rng.integers(3, 120, 4000).astype(np.uint16).tofile(os.path.join(root, "pretrain", "valid", "a.bin"))


def _spec(run_dir, shards):
    from chronos.mlx.training.longrun import PhaseSpec, RunSpec
    model = dict(architecture_version=3, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, vocab_size=128, num_experts=4, num_experts_per_tok=2,
                 num_shared_experts=1, moe_intermediate_size=64, max_position_embeddings=64,
                 recurrent_depth=3, prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0)
    phase = PhaseSpec(name="pretrain", budget_fraction=1.0, peak_lr=1e-3, tokens_per_step=256,
                      micro_tokens=128, seq_len=32, warmup_fraction=0.1)
    return RunSpec(run_dir=run_dir, shards_dir=shards, tokenizer_dir=shards, model_config=model,
                   phases=[phase], budget_hours=1.0, bleu_samples=0, validate_every_s=1e9,
                   checkpoint_every_s=1e9, dtype="float32")


def _train(run, steps):
    import time
    if run.state["phase_total"] is None:
        run.state["phase_total"] = 12          # fixed schedule (skip the throughput probe)
    run.run_phase(0, time.monotonic() + 3600, max_steps=steps)


def test_resume_is_bit_exact(tmp_path):
    """CPU stream: bit-exact.  (GPU reductions are not bitwise deterministic;
    measured straight-vs-resumed 7.5e-9 == straight-vs-straight on M5.)"""
    from mlx.utils import tree_flatten
    from chronos.mlx.training.longrun import LongRun
    shards = str(tmp_path / "shards"); _tiny_shards(shards)
    with mx.stream(mx.cpu):
        straight = LongRun(_spec(str(tmp_path / "a"), shards), log=lambda *_: None)
        _train(straight, 6)
        first = LongRun(_spec(str(tmp_path / "b"), shards), log=lambda *_: None)
        _train(first, 3)
        first.save_state()
        resumed = LongRun(_spec(str(tmp_path / "b"), shards), log=lambda *_: None)
        assert resumed.try_resume() and resumed.state["phase_step"] == 3
        _train(resumed, 3)
    a = dict(tree_flatten(straight.model.parameters()))
    b = dict(tree_flatten(resumed.model.parameters()))
    assert a.keys() == b.keys()
    worst = max(float(mx.abs(a[k] - b[k]).max().item()) for k in a)
    assert worst == 0.0, worst
