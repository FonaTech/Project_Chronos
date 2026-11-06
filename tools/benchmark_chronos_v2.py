"""Reproducible Chronos v2 architecture, throughput and convergence benchmark.

This benchmark deliberately uses a tiny deterministic next-token task rather
than random labels.  That makes convergence and generalization measurable in a
few minutes while still exercising the real Chronos model, recurrent controller,
hybrid KV cache and optional SSD expert runtime.

Examples::

    python tools/benchmark_chronos_v2.py --device cpu --train_steps 60
    python tools/benchmark_chronos_v2.py --device mps --train_steps 20 --skip_offload

The JSON report separates model quality from runtime diagnostics.  It never
turns a fallback/offload result into a strict-equivalence claim: exact token
match is reported against a full-resident reference for each policy.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

import chronos.deps  # noqa: F401 - registers the bundled MiniMind package
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.runtime.inference_engine import ChronosInferenceEngine
from chronos.trainer.loss_mixin import chronos_loss_term


PATTERN = torch.tensor([7, 13, 29, 41, 53, 67, 79, 97], dtype=torch.long)


@dataclass
class VariantResult:
    name: str
    architecture_version: int
    recurrent_enabled: bool
    recurrent_depth: int
    adaptive_halting: bool
    virtual_num_hidden_layers: int
    train_initial_ce: float
    train_final_ce: float
    train_best_ce: float
    train_tokens_per_sec: float
    eval_ce: float
    eval_ppl: float
    eval_next_token_accuracy: float
    prefill_ms: float
    decode_ms_per_token: float
    decode_tokens_per_sec: float
    kv_cache_kb: float
    kv_cache_seq_len: int
    output_unique_ratio: float
    nan_or_inf: bool
    prefill_average_depth: float
    prefill_early_exit_ratio: float
    decode_average_depth: float
    decode_early_exit_ratio: float
    halt_final_mass: float
    ponder_loss: float
    parameter_mb: float
    expert_parameter_mb: float
    estimated_active_expert_mb: float
    rss_after_model_gb: float
    rss_after_run_gb: float


def _rss_gb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 ** 3)
    except Exception:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Darwin reports bytes, Linux reports KiB.
        if os.uname().sysname == "Darwin":
            return value / (1024 ** 3)
        return value / (1024 ** 3 * 1024)


def _json_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().item()
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _as_output(value):
    """Normalize canonical Chronos output and legacy tuple output."""
    # ModelOutput implements tuple-like indexing; indexing it would discard
    # Chronos-only fields (lookahead/halting). Prefer the canonical object.
    if hasattr(value, "logits"):
        return value
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _build_config(variant: str, args: argparse.Namespace) -> ChronosConfig:
    common = dict(
        architecture_version=1 if variant == "v1" else 2,
        recurrent_enabled=variant != "v1",
        recurrent_depth=1 if variant == "v1" else (2 if variant == "v2_fixed2" else 4),
        min_recurrent_depth=1 if variant in {"v1", "v2_adaptive4"} else (2 if variant == "v2_fixed2" else 1),
        adaptive_halting=variant == "v2_adaptive4",
        depth_schedule="adaptive" if variant == "v2_adaptive4" else "fixed",
        # The adaptive profile uses a deliberately explicit 0.70 budget so a
        # four-step model can demonstrate physical prefill early-exit after
        # training.  Fixed-depth variants ignore this field.
        halting_threshold=0.70 if variant == "v2_adaptive4" else 0.80,
        halting_epsilon=0.01,
        recurrent_stability_threshold=0.25,
        max_inference_depth=1 if variant == "v1" else (2 if variant == "v2_fixed2" else 4),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.seq_len + args.decode_tokens + 32,
        use_moe=True,
        num_experts=args.num_experts,
        num_experts_per_tok=1,
        num_shared_experts=1,
        use_hybrid_attention=True,
        flash_attn=False,
        dropout=0.0,
        vram_budget_gb=4.0,
    )
    return ChronosConfig(**common)


def _batch(batch_size: int, seq_len: int, phase: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.long)
    rows = []
    for row in range(batch_size):
        values = PATTERN[(positions + phase + row) % PATTERN.numel()]
        rows.append(values.remainder(vocab_size))
    ids = torch.stack(rows)
    return ids, ids.clone()


def _parameter_bytes(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    experts = 0
    for name, parameter in model.named_parameters():
        size = parameter.numel() * parameter.element_size()
        total += size
        if ".experts." in name:
            experts += size
    return total, experts


def _kv_bytes(past_key_values) -> tuple[int, int]:
    total = 0
    seq_len = 0
    for kv in past_key_values or []:
        if kv is None:
            continue
        if kv and hasattr(kv[0], "shape"):
            seq_len = max(seq_len, int(kv[0].shape[1]))
        for tensor in kv:
            total += tensor.numel() * tensor.element_size()
    return total, seq_len


def _recurrent_metrics(output) -> tuple[float, float, float, float]:
    depth = output.recurrent_depth_used
    average_depth = _json_float(depth) if depth is not None else 1.0
    hints = output.cache_hints or {}
    early = hints.get("early_exit_ratio", 0.0)
    early = _json_float(early)
    halts = output.halting_probs
    halt_final = _json_float(halts[..., -1]) if halts is not None else 1.0
    ponder = _json_float(output.ponder_loss) if output.ponder_loss is not None else 0.0
    return average_depth, early, halt_final, ponder


def _train(model, config, args, device: str) -> tuple[list[float], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    losses: list[float] = []
    tokens = 0
    start = time.perf_counter()
    model.train()
    for step in range(args.train_steps):
        ids, labels = _batch(args.batch_size, args.seq_len, step % PATTERN.numel(), config.vocab_size)
        ids, labels = ids.to(device), labels.to(device)
        output = _as_output(model(ids, labels=labels))
        total = chronos_loss_term(
            model,
            output.loss,
            output.lookahead_probs,
            config,
            aux_loss=output.aux_loss,
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite training loss at step {step}: {total.item()}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output.loss.detach().item()))
        tokens += ids.numel()
    elapsed = max(time.perf_counter() - start, 1e-9)
    return losses, tokens / elapsed


@torch.no_grad()
def _evaluate(model, args, device: str) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for phase in range(args.eval_batches):
        ids, labels = _batch(args.batch_size, args.seq_len, (phase + 3) % PATTERN.numel(), args.vocab_size)
        ids, labels = ids.to(device), labels.to(device)
        output = _as_output(model(ids, labels=labels))
        n = int((labels[:, 1:] != -100).sum().item())
        total_loss += float(output.loss.item()) * max(n, 1)
        total_tokens += max(n, 1)
        prediction = output.logits[:, :-1].argmax(dim=-1)
        correct += int((prediction == labels[:, 1:]).sum().item())
    ce = total_loss / max(total_tokens, 1)
    return ce, math.exp(min(ce, 20.0)), correct / max(total_tokens, 1)


@torch.no_grad()
def _decode(model, args, device: str) -> dict[str, Any]:
    model.eval()
    prompt, _ = _batch(1, args.prompt_len, 0, args.vocab_size)
    prompt = prompt.to(device)
    # Warmup is outside the measured interval.
    warm = _as_output(model(prompt, use_cache=True))
    past = warm.past_key_values
    token = warm.logits[:, -1:].argmax(dim=-1)
    for _ in range(2):
        warm = _as_output(model(token, past_key_values=past, use_cache=True))
        past = warm.past_key_values
        token = warm.logits[:, -1:].argmax(dim=-1)

    start = time.perf_counter()
    prefill = _as_output(model(prompt, use_cache=True))
    prefill_ms = (time.perf_counter() - start) * 1000.0
    past = prefill.past_key_values
    generated = []
    start = time.perf_counter()
    output = prefill
    for _ in range(args.decode_tokens):
        token = output.logits[:, -1:].argmax(dim=-1)
        generated.append(token)
        output = _as_output(model(token, past_key_values=past, use_cache=True))
        past = output.past_key_values
    decode_s = max(time.perf_counter() - start, 1e-9)
    kv_bytes, kv_seq_len = _kv_bytes(past)
    ids = torch.cat(generated, dim=-1) if generated else torch.empty((1, 0), device=device, dtype=torch.long)
    unique_ratio = float(ids.unique().numel() / max(ids.numel(), 1))
    depth, early, halt, ponder = _recurrent_metrics(output)
    return {
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_s * 1000.0 / max(args.decode_tokens, 1),
        "decode_tokens_per_sec": args.decode_tokens / decode_s,
        "kv_cache_kb": kv_bytes / 1024.0,
        "kv_cache_seq_len": kv_seq_len,
        "output_unique_ratio": unique_ratio,
        "nan_or_inf": bool(not torch.isfinite(output.logits).all().item()),
        "decode_average_depth": depth,
        "decode_early_exit_ratio": early,
        "halt_final_mass": halt,
        "ponder_loss": ponder,
    }


@torch.no_grad()
def _prefill_recurrent_diagnostics(model, args, device: str) -> tuple[float, float]:
    model.eval()
    ids, _ = _batch(1, args.seq_len, 2, args.vocab_size)
    output = _as_output(model(ids.to(device), use_cache=False))
    depth, early, _halt, _ponder = _recurrent_metrics(output)
    return depth, early


@torch.no_grad()
def _greedy_reference(model, prompt: torch.Tensor, steps: int) -> torch.Tensor:
    output = _as_output(model(prompt, use_cache=True))
    past = output.past_key_values
    result = [prompt]
    for _ in range(steps):
        token = output.logits[:, -1:].argmax(dim=-1)
        result.append(token)
        output = _as_output(model(token, past_key_values=past, use_cache=True))
        past = output.past_key_values
    return torch.cat(result, dim=-1)


def _run_variant(name: str, args: argparse.Namespace, device: str) -> tuple[VariantResult, ChronosForCausalLM]:
    torch.manual_seed(args.seed)
    config = _build_config(name, args)
    model = ChronosForCausalLM(config).to(device)
    param_bytes, expert_bytes = _parameter_bytes(model)
    rss_model = _rss_gb()
    losses, train_tps = _train(model, config, args, device)
    eval_ce, eval_ppl, eval_acc = _evaluate(model, args, device)
    prefill_depth, prefill_exit = _prefill_recurrent_diagnostics(model, args, device)
    decode = _decode(model, args, device)
    active_expert_bytes = (
        expert_bytes * config.num_experts_per_tok / max(config.num_experts, 1)
    )
    result = VariantResult(
        name=name,
        architecture_version=int(config.architecture_version),
        recurrent_enabled=bool(config.recurrent_enabled),
        recurrent_depth=int(config.recurrent_depth),
        adaptive_halting=bool(config.adaptive_halting),
        virtual_num_hidden_layers=int(config.virtual_num_hidden_layers),
        train_initial_ce=losses[0],
        train_final_ce=losses[-1],
        train_best_ce=min(losses),
        train_tokens_per_sec=train_tps,
        eval_ce=eval_ce,
        eval_ppl=eval_ppl,
        eval_next_token_accuracy=eval_acc,
        prefill_ms=decode["prefill_ms"],
        decode_ms_per_token=decode["decode_ms_per_token"],
        decode_tokens_per_sec=decode["decode_tokens_per_sec"],
        kv_cache_kb=decode["kv_cache_kb"],
        kv_cache_seq_len=int(decode["kv_cache_seq_len"]),
        output_unique_ratio=decode["output_unique_ratio"],
        nan_or_inf=decode["nan_or_inf"],
        prefill_average_depth=prefill_depth,
        prefill_early_exit_ratio=prefill_exit,
        decode_average_depth=decode["decode_average_depth"],
        decode_early_exit_ratio=decode["decode_early_exit_ratio"],
        halt_final_mass=decode["halt_final_mass"],
        ponder_loss=decode["ponder_loss"],
        parameter_mb=param_bytes / 1e6,
        expert_parameter_mb=expert_bytes / 1e6,
        estimated_active_expert_mb=active_expert_bytes / 1e6,
        rss_after_model_gb=rss_model,
        rss_after_run_gb=_rss_gb(),
    )
    return result, model


def _run_offload(args: argparse.Namespace, source_model: ChronosForCausalLM, device: str) -> dict[str, Any]:
    config = _build_config("v2_fixed2", args)
    config.storage_format = "pt"
    config.layer_aware_cache = False
    config.vram_budget_gb = args.offload_vram_budget_gb
    reference = ChronosForCausalLM(config).to(device).eval()
    reference.load_state_dict(source_model.state_dict(), strict=False)
    _total_param_bytes, full_expert_bytes = _parameter_bytes(reference)
    prompt, _ = _batch(1, args.prompt_len, 0, args.vocab_size)
    prompt = prompt.to(device)
    reference_ids = _greedy_reference(reference, prompt, args.offload_decode_tokens)
    rows: dict[str, Any] = {}
    for policy in ("sync_on_demand", "on_demand", "fallback_diagnostic"):
        model = ChronosForCausalLM(config).to(device).eval()
        model.load_state_dict(source_model.state_dict(), strict=False)
        with tempfile.TemporaryDirectory(prefix="chronos_v2_bench_") as ssd_dir:
            engine = ChronosInferenceEngine(model, config, ssd_dir=ssd_dir)
            try:
                os.environ["CHRONOS_SIM_SSD_MS"] = str(args.simulated_ssd_ms)
                engine.setup(warm_expert_ids=[0])
                output = engine.generate(
                    prompt,
                    max_new_tokens=args.offload_decode_tokens,
                    do_sample=False,
                    miss_policy=policy,
                )
                stats = dict(engine.last_stats)
            finally:
                engine.teardown()
                os.environ.pop("CHRONOS_SIM_SSD_MS", None)
            generated = output.detach().cpu()
            target = reference_ids.detach().cpu()
            n = min(generated.shape[-1], target.shape[-1])
            exact = float((generated[..., :n] == target[..., :n]).float().mean().item())
            stats.update({
                "exact_token_match_to_full_resident": exact,
                "full_expert_weight_mb": full_expert_bytes / 1e6,
                "resident_expert_weight_mb": (
                    float(stats.get("expert_size_kb", 0.0))
                    * 1024.0
                    * float(stats.get("vram_experts", 0.0))
                    / 1e6
                ),
                "ssd_artifact_bytes": sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, _dirs, files in os.walk(ssd_dir)
                    for file in files
                ),
            })
            stats["estimated_expert_memory_savings"] = 1.0 - (
                stats["resident_expert_weight_mb"]
                / max(stats["full_expert_weight_mb"], 1e-12)
            )
            rows[policy] = stats
        del model
        gc.collect()
    del reference
    gc.collect()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--train_steps", type=int, default=60)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument("--decode_tokens", type=int, default=24)
    parser.add_argument("--offload_decode_tokens", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--offload_vram_budget_gb", type=float, default=1e-5)
    parser.add_argument("--simulated_ssd_ms", type=int, default=1)
    parser.add_argument("--skip_offload", action="store_true")
    parser.add_argument("--output", default="results/chronos_v2_benchmark.json")
    args = parser.parse_args()

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS was requested but torch.backends.mps.is_available() is false")

    print("== Chronos v2 architecture benchmark ==")
    print(f"device={device} seed={args.seed} train_steps={args.train_steps}")
    rows = []
    models: dict[str, ChronosForCausalLM] = {}
    for name in ("v1", "v2_fixed2", "v2_fixed4", "v2_adaptive4"):
        print(f"[benchmark] {name}")
        result, model = _run_variant(name, args, device)
        rows.append(asdict(result))
        models[name] = model
        print(
            f"  train CE {result.train_initial_ce:.4f} -> {result.train_final_ce:.4f}; "
            f"eval PPL={result.eval_ppl:.2f}; decode={result.decode_tokens_per_sec:.1f} tok/s; "
            f"prefill depth={result.prefill_average_depth:.2f} exit={result.prefill_early_exit_ratio:.2f}"
        )

    offload = None
    if not args.skip_offload:
        print("[benchmark] paged expert runtime (v2_fixed2)")
        offload = _run_offload(args, models["v2_fixed2"], device)
        for policy, stats in offload.items():
            print(
                f"  {policy}: {stats.get('tokens_per_sec', 0.0):.1f} tok/s, "
                f"resident_hit={stats.get('resident_vram_hit_rate', 0.0):.3f}, "
                f"prediction_recall={stats.get('prediction_topk_recall', 0.0):.3f}, "
                f"exact_match={stats.get('exact_token_match_to_full_resident', 0.0):.3f}"
            )

    report = {
        "environment": {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "cpu_threads": torch.get_num_threads(),
        },
        "benchmark_config": vars(args),
        "variants": rows,
        "paged_expert_runtime": offload,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
