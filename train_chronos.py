"""
train_chronos.py — Phase 1 training entry point for Project Chronos.

Usage:
    python train_chronos.py \
        --data_path /path/to/pretrain.jsonl \
        --hidden_size 512 \
        --num_hidden_layers 8 \
        --num_experts 4 \
        --lambda_temporal 1e-3 \
        --epochs 2 \
        --device cuda:0
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from trainer.trainer_utils import setup_seed, Logger
from dataset.lm_dataset import PretrainDataset
from chronos.model.config import ChronosConfig
from chronos.trainer.chronos_trainer import ChronosTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Project Chronos Pretraining")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./out")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    # Model
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--num_experts_per_tok", type=int, default=1)
    p.add_argument("--num_shared_experts", type=int, default=1)
    p.add_argument("--lookahead_steps", type=int, default=2)
    # Loss coefficients
    p.add_argument("--lambda_balance", type=float, default=5e-4)
    p.add_argument("--lambda_temporal", type=float, default=1e-3)
    # VRAM budget
    p.add_argument("--vram_budget_gb", type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(42)

    config = ChronosConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        num_shared_experts=args.num_shared_experts,
        lookahead_steps=args.lookahead_steps,
        lambda_balance=args.lambda_balance,
        lambda_temporal=args.lambda_temporal,
        vram_budget_gb=args.vram_budget_gb,
        use_moe=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        chronos.deps.get_tokenizer_path()
    )
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    trainer = ChronosTrainer(config, args)
    total_params = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    Logger(f"Chronos model: {total_params:.2f}M params | "
           f"experts={config.num_experts} shared={config.num_shared_experts} "
           f"lookahead={config.lookahead_steps} "
           f"λ1={config.lambda_balance} λ2={config.lambda_temporal}")

    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, loader, len(loader))

    Logger("Training complete.")


if __name__ == "__main__":
    main()
