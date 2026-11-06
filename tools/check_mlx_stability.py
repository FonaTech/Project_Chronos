"""Reproduce the 476-microstep BF16 training case without touching ./out.

Run from the repository root: python tools/check_mlx_stability.py
Results, JSONL metrics and checkpoints go to a new temporary directory.
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional config JSON or checkpoint sidecar")
    parser.add_argument("--data", default=str(ROOT / "tests/fixtures/tiny_pretrain.jsonl"))
    parser.add_argument("--output", help="Use a new directory; existing paths are refused")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    import torch
    import mlx.core as mx
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer
    from chronos.config_schema import model_config_kwargs
    from chronos.data.flexible_dataset import FlexibleDataset
    from chronos.deps import get_tokenizer_path
    from chronos.model.config import ChronosConfig
    from chronos.mlx.training import run_mlx_stage
    from ui.presets import get_preset

    random.seed(args.seed); torch.manual_seed(args.seed); mx.random.seed(args.seed)
    torch.set_num_threads(8)
    raw = get_preset("Small (256H × 4L × 4E)")
    if args.config:
        loaded = json.loads(Path(args.config).read_text())
        raw.update(loaded.get("config", loaded))
    config = ChronosConfig(**model_config_kwargs(raw))
    output = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="chronos-stability-"))
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_path())
    dataset = FlexibleDataset(args.data, tokenizer, max_length=256)
    train = DataLoader(Subset(dataset, [i for i in range(len(dataset)) if i % 20]), batch_size=8, shuffle=True)
    val = DataLoader(Subset(dataset, [i for i in range(len(dataset)) if i % 20 == 0]), batch_size=8)
    train_args = SimpleNamespace(dtype="bfloat16", learning_rate=5e-4, weight_decay=.01,
        steps=args.max_steps or None, epochs=args.epochs, accumulation_steps=8,
        grad_clip=1., save_interval=1000, log_interval=10)
    (output / "run_config.json").write_text(json.dumps({"seed": args.seed, "model": model_config_kwargs(config),
        "training": vars(train_args), "dataset": args.data}, indent=2))
    print(f"Results: {output}", flush=True)
    events = []
    with (output / "metrics.jsonl").open("w") as log:
        def report(event):
            events.append(event)
            line = json.dumps(event)
            log.write(line + "\n"); log.flush()
            if event["event"] == "step":
                print(f"step={event['step']} loss={event['loss']:.4f} CE={event.get('ce', 0):.4f} "
                      f"recurrent={event['recurrent_loss']:.5f} tokens/s={event['tokens_per_s']:.0f}", flush=True)
            else:
                print(line, flush=True)
        result = run_mlx_stage(stage="pretrain", config=config, checkpoint_path=None, save_dir=str(output),
            loader=train, validation_loader=val, args=train_args, progress_callback=report)
    summary = dict(vars(result), validation=[e for e in events if e["event"] == "validation"],
                   peak_mlx_bytes=mx.get_peak_memory())
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    if result.failure_reason:
        raise SystemExit(f"Training failed: {result.failure_reason}")


if __name__ == "__main__":
    main()
