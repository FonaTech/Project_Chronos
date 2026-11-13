"""Out-of-range token ids are rejected before they reach an MLX embedding gather (0.3.2a3)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")


def test_check_ids_in_vocab_accepts_valid_and_rejects_out_of_range():
    from chronos.mlx.training.trainer import _check_ids_in_vocab
    _check_ids_in_vocab(np.array([[0, 5, 63]]), 64, "x")
    _check_ids_in_vocab(torch.tensor([[0, 63]]), 64, "x")
    _check_ids_in_vocab(mx.array([[0, 63]]), 64, "x")
    _check_ids_in_vocab(np.zeros((0, 3), dtype=np.int64), 64, "empty")
    _check_ids_in_vocab(np.array([[0.5]]), 64, "float ids are not token ids")
    for bad in (np.array([[0, 64]]), torch.tensor([[-1, 3]]), mx.array([[6170]])):
        with pytest.raises(ValueError, match="vocab_size is 64"):
            _check_ids_in_vocab(bad, 64, "x")


def test_check_batch_ids_covers_tuples_dicts_and_dpo_pairs():
    from chronos.mlx.training.trainer import _check_batch_ids
    ok = torch.tensor([[1, 2, 3]])
    _check_batch_ids((ok, ok), 10, "sft")
    _check_batch_ids({"input_ids": ok, "labels": ok}, 10, "sft")
    _check_batch_ids({"x_chosen": ok, "x_rejected": ok}, 10, "dpo")
    with pytest.raises(ValueError, match="sft batch ids"):
        _check_batch_ids((torch.tensor([[1, 10]]), ok), 10, "sft")
    with pytest.raises(ValueError, match="x_rejected"):
        _check_batch_ids({"x_chosen": ok, "x_rejected": torch.tensor([[10]])}, 10, "dpo")


def test_run_mlx_stage_refuses_a_batch_larger_than_the_vocab(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from chronos.mlx.training import run_mlx_stage
    from chronos.model.config import ChronosConfig
    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2, num_experts_per_tok=1,
        num_shared_experts=1, num_attention_heads=2, num_key_value_heads=2, kv_latent_dim=8,
        rope_dim=4, max_seq_len=16, vocab_size=64, use_moe=True)
    
    class Args:
        learning_rate = 1e-4
        weight_decay = 0.0
        steps = 1
        epochs = 1
        max_seq_len = 8
    
    ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    ids[0, 3] = 6170        # an id from a larger vocabulary
    with pytest.raises(ValueError, match="vocab_size is 64"):
        run_mlx_stage(stage="pretrain", config=cfg, checkpoint_path=None, save_dir=str(tmp_path),
            loader=DataLoader(TensorDataset(ids, ids.clone()), batch_size=1), args=Args())
