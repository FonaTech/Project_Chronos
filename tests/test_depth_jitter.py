"""recurrent_depth_jitter samples ONE depth per optimizer step, shared by every micro-batch.
Before the fix each micro-batch resampled, so a step mixed virtual-layer counts and the router-stat
accumulator raised 'broadcast_shapes (14,E) and (10,E)'."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip("mlx.core")


def _shards(root):
    os.makedirs(os.path.join(root, "pretrain", "train"))
    os.makedirs(os.path.join(root, "pretrain", "valid"))
    rng = np.random.default_rng(0)
    rng.integers(3, 120, 20000).astype(np.uint16).tofile(os.path.join(root, "pretrain", "train", "a.bin"))
    rng.integers(3, 120, 4000).astype(np.uint16).tofile(os.path.join(root, "pretrain", "valid", "a.bin"))


@pytest.mark.parametrize("compile_mode", ["off", "auto"])
def test_depth_jitter_uses_one_depth_per_step(tmp_path, compile_mode):
    import time
    from chronos.mlx.training.longrun import LongRun, PhaseSpec, RunSpec
    shards = str(tmp_path / "shards")
    _shards(shards)
    model = dict(architecture_version=3, hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=64, max_position_embeddings=64, recurrent_depth=4, prelude_layers=1, coda_layers=1,
        dropout=0.0, fallback_mask_prob=0.0, recurrent_order="block", recurrent_strategy="adaptive",
        recurrent_depth_jitter=1.0, min_recurrent_depth=2, depth_schedule_warmup_fraction=0.0)
    phase = PhaseSpec(name="pretrain", budget_fraction=1.0, peak_lr=1e-3, tokens_per_step=256, micro_tokens=64,
        seq_len=32, warmup_fraction=0.1)
    spec = RunSpec(run_dir=str(tmp_path / "run"), shards_dir=shards, tokenizer_dir=shards, model_config=model,
        phases=[phase], budget_hours=1.0, validate_every_s=1e9, checkpoint_every_s=1e9, dtype="float32",
        log_every_steps=1, extra={"mlx_compile": compile_mode})
    with mx.stream(mx.cpu):
        run = LongRun(spec, log=lambda *_: None)
        run.state["phase_total"] = 12
        run.spec.schedule = "steps"
        controller = run.model.recurrent_controller
        seen = []
        base = controller.execution_depth
        def record(cfg, training):
            d = base(cfg, training)
            if training:
                seen.append((int(d), getattr(controller, "_step_depth", None)))
            return d
        controller.execution_depth = record
        run.run_phase(0, time.monotonic() + 3600, max_steps=4)
    assert seen, "no training depth sampled"
    per_step = [s for d, s in seen if s is not None]
    assert per_step and all(d == s for d, s in seen if s is not None)
    assert len({int(x) for x in per_step}) >= 1 and min(int(x) for x in per_step) >= 2
