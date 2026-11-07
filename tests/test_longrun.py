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
                   phases=[phase], budget_hours=1.0, validate_every_s=1e9,
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


def test_task_eval_hook_feeds_validation_and_final_rows(tmp_path):
    """A task extension adds metrics by overriding ``task_eval``; the kernel has none."""
    import json
    from chronos.mlx.training.longrun import LongRun
    shards = str(tmp_path / "shards")
    _tiny_shards(shards)

    class Task(LongRun):
        calls = []

        def task_eval(self, phase_name, n, *, final=False):
            self.calls.append((phase_name, n, final, self.model.training))
            self.write_samples([{"input": "x", "output": "y"}])
            if final:
                raise RuntimeError("boom")                 # logged, never stops the run
            return {"qa_accuracy": 0.5}

    spec = _spec(str(tmp_path / "run"), shards)
    spec.eval_samples, spec.final_eval_samples = 3, 7
    spec.validate_every_s = 0.0                                   # validate after every update
    with mx.stream(mx.cpu):
        base = LongRun(_spec(str(tmp_path / "plain"), shards), log=lambda *_: None)
        assert base.task_eval("pretrain", 3) == {} and base.eval_samples_for("pretrain", final=False) == 0
        run = Task(spec, log=lambda *_: None)
        _train(run, 2)
        final = run._safe_task_eval("pretrain", final=True)
    assert run.calls[0] == ("pretrain", 3, False, False)          # eval mode during the hook
    assert run.calls[-1][1:3] == (7, True) and run.model.training
    assert final["task_eval_error"].startswith("RuntimeError")
    rows = [json.loads(line) for line in open(run.metrics_path, encoding="utf-8")]
    assert any(r.get("event") == "validation" and r.get("qa_accuracy") == 0.5 for r in rows)
    samples = [json.loads(line) for line in open(tmp_path / "run" / "samples.jsonl", encoding="utf-8")]
    assert samples[0]["output"] == "y" and samples[0]["phase"] == "pretrain"


def test_load_spec_resolves_relative_paths_and_rejects_task_keys(tmp_path):
    import json
    from chronos.mlx.training.longrun import load_spec
    recipe = {"run_dir": "run", "shards_dir": "data/shards", "tokenizer_dir": "/abs/tok", "model_config": {},
              "phases": [{"name": "pretrain", "budget_fraction": 1.0, "peak_lr": 1e-3, "tokens_per_step": 8}],
              "extra": {"eval_root": "data/eval"}, "_comment": "ignored"}
    path = tmp_path / "pkg" / "recipe.json"
    path.parent.mkdir()
    path.write_text(json.dumps(recipe), encoding="utf-8")
    spec = load_spec(str(path), budget_hours=2, extra_path_keys=("eval_root",))
    assert spec.shards_dir == str(tmp_path / "pkg" / "data" / "shards") and spec.tokenizer_dir == "/abs/tok"
    assert spec.extra["eval_root"] == str(tmp_path / "pkg" / "data" / "eval") and spec.budget_hours == 2.0
    path.write_text(json.dumps(dict(recipe, bleu_samples=10)), encoding="utf-8")
    with pytest.raises(ValueError, match="bleu_samples"):
        load_spec(str(path))
