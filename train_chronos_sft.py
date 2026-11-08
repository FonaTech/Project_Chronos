"""train_chronos_sft.py — Stage 2 (SFT) entry point."""
import argparse


from chronos.backend import resolve_training_device
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.sft_trainer import ChronosSFTTrainer, build_sft_loader
from chronos.data.flexible_dataset import dataset_modality_summary
from chronos.trainer.stage_utils import (
    add_topology_args,
    build_config_from_upstream,
    load_stage_tokenizer,
    load_required_checkpoint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--from_weight", default="chronos",
                   help="Pretrained checkpoint stem (loaded from save_dir/<stem>_<H>_moe.pth).")
    p.add_argument("--save_dir", default="out")
    p.add_argument("--steps", type=int, default=None,
                   help="If set, stop after N steps (smoke runs).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--num_workers", default="auto")
    p.add_argument("--cpu_threads", default="auto")
    p.add_argument("--cpu_budget_percent", default=100, type=float)
    add_topology_args(p, defaults=False)
    p.add_argument("--lambda_temporal", type=float, default=None)
    p.add_argument("--lambda_lookahead", type=float, default=None)
    p.add_argument("--lambda_lookahead_topk", type=float, default=None)
    p.add_argument("--lambda_lookahead_union", type=float, default=None)
    p.add_argument("--lambda_router_locality", type=float, default=None)
    p.add_argument("--lambda_cluster_locality", type=float, default=None,
                   help="default: inherited from the upstream checkpoint (0 = off)")
    p.add_argument("--cluster_locality_window", type=int, default=None)
    p.add_argument("--fallback_mask_prob", type=float, default=0.04)
    p.add_argument("--lambda_router_anchor", type=float, default=0.01,
                   help="SFT default: weak anchor.")
    args = p.parse_args()
    selected_backend, resolved_device = resolve_training_device(args.device)
    args.device = resolved_device

    cfg, ckp_path, sources = build_config_from_upstream(
        args,
        default_stem="chronos",
        max_positions=args.max_seq_len,
        lambda_router_anchor=args.lambda_router_anchor,
    )
    modality_summary = dataset_modality_summary(args.data_path)
    if cfg.auto_detect_modalities:
        cfg.modalities = modality_summary["detected_modalities"]
    tokenizer = load_stage_tokenizer(args, cfg, ckp_path)
    loader = build_sft_loader(
        args.data_path,
        tokenizer,
        args.max_seq_len,
        args.batch_size,
        device=selected_backend if selected_backend == "mlx" else args.device,
        num_workers=args.num_workers,
        modalities=cfg.modalities, image_size=cfg.image_size,
        audio_sample_rate=cfg.audio_sample_rate,
        max_audio_seconds=cfg.max_audio_seconds,
        media_error_policy=cfg.media_error_policy,
    )
    if selected_backend == "mlx":
        from chronos.mlx.training import run_mlx_stage

        print(f"[SFT][MLX] Loaded pretrain weights from {ckp_path}")
        print(f"[SFT][MLX] Topology sources: {', '.join(sources)}")
        run_mlx_stage(
            stage="sft",
            config=cfg,
            checkpoint_path=ckp_path,
            save_dir=args.save_dir,
            loader=loader,
            args=args,
        )
        return

    model = ChronosForCausalLM(cfg)
    load_required_checkpoint(model, ckp_path, args.device)
    model = model.to(args.device)
    print(f"[SFT] Loaded pretrain weights from {ckp_path}")
    print(f"[SFT] Topology sources: {', '.join(sources)}")
    print(f"[SFT] Training backend: {selected_backend}  device={args.device}")

    trainer = ChronosSFTTrainer(model, cfg, args, tokenizer)

    # Capture router reference on the first batch BEFORE training
    if cfg.lambda_router_anchor > 0:
        first_batch = next(iter(loader))
        if isinstance(first_batch, dict):
            trainer.set_calibration_batch(
                first_batch["input_ids"], modality_inputs=first_batch.get("modality_inputs"),
                modality_mask=first_batch.get("modality_mask"),
            )
        else:
            trainer.set_calibration_batch(first_batch[0])
        print("[SFT] Router anchor reference captured.")

    iters = len(loader) if args.steps is None else min(args.steps, len(loader))
    global_step = 0
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, loader, iters, start_step=global_step, max_steps=args.steps)
        global_step += iters
        if args.steps is not None and global_step >= int(args.steps):
            break
    trainer._save(epoch=args.epochs - 1, step=iters)
    print(f"[SFT] Done. Checkpoint: {args.save_dir}/sft_{cfg.hidden_size}_moe.pth")


if __name__ == "__main__":
    main()
