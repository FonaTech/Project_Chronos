"""Inspect a LongRun state checkpoint without disturbing the training process.

Loads ``<run>/checkpoints/latest.state.pkl`` on the MLX **CPU** device (so a
running GPU job keeps its throughput), then reports:

* held-out CE / perplexity on the phase's fixed validation windows;
* routing balance and recurrent-depth statistics on those windows;
* greedy en→zh / zh→en continuations on FLORES+ dev with BLEU / chrF;

and optionally exports a serveable ``.pth`` + ``.config.json`` (with the
run's own tokenizer recorded) for ``chronos serve``.  The report is written to
``<run>/inspections/<phase>_s<step>.json`` so the monitor can show it.

    python -m chronos.mlx.training.inspect_checkpoint --run-dir runs/enzh_50m_v1 \
        --export serve_models/chronos_enzh_pretrain_s300.pth
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time

import mlx.core as mx
import mlx.nn as nn


def load_run_model(run_dir: str, *, device: str = "cpu", state_file: str = "latest.state.pkl"):
    from chronos.model.config import ChronosConfig
    from chronos.mlx.training.longrun import _restore_arrays
    from chronos.mlx.training.trainer import _load_mlx_from_checkpoint
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    spec = json.load(open(os.path.join(run_dir, "run_spec.json"), encoding="utf-8"))
    config = ChronosConfig(**spec["model_config"])
    model = _load_mlx_from_checkpoint(config, None, dtype_name=spec.get("dtype"))
    with open(os.path.join(run_dir, "checkpoints", state_file), "rb") as fh:
        payload = pickle.load(fh)          # written by this project's own trainer
    model.update(_restore_arrays(payload["params"], payload["dtypes"]))
    # The policy picked the compute dtype at build time (CPU -> FP32: exact
    # upcast of the BF16 weights; CPU GatherMM is FP32-only).  Restored
    # arrays keep their saved dtype, so convert them once to that dtype.
    model.set_dtype(model._chronos_precision.mx_dtype())
    mx.eval(model.parameters())
    model.eval()
    return spec, config, model, payload["state"]


def _phase(spec: dict, state: dict) -> dict:
    return spec["phases"][min(int(state["phase_index"]), len(spec["phases"]) - 1)]


def held_out(spec, config, model, phase, n_batches: int = 8) -> dict:
    from chronos.mlx.training.longrun import PretrainSource, SFTSource, recurrent_metrics, routing_metrics
    from chronos.mlx.training.trainer import _collect_router_probs
    source = (PretrainSource(spec["shards_dir"], phase["seq_len"], spec["seed"]) if phase["name"] == "pretrain"
              else SFTSource(spec["shards_dir"], spec["seed"]))
    total, count, extra = 0.0, 0, {}
    for i, (ids, labels) in enumerate(source.valid_batches(n_batches, phase["micro_tokens"])):
        logits, _, _ = model(mx.array(ids, dtype=mx.int32), logits_dtype=mx.float32)
        lb = mx.array(labels, dtype=mx.int32).reshape(-1)
        mask = lb != -100
        ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     mx.where(mask, lb, 0), reduction="none")
        s, n = (ce * mask).sum(), mask.sum()
        mx.eval(s, n)
        total, count = total + float(s.item()), count + int(n.item())
        if i == 0:
            extra = {**routing_metrics(_collect_router_probs(model), int(config.num_experts_per_tok)),
                     **recurrent_metrics(getattr(model, "last_recurrent_state", None))}
    ce = total / max(count, 1)
    return {"valid_ce": round(ce, 4), "valid_ppl": round(math.exp(min(ce, 30)), 2), "valid_tokens": count, **extra}


def greedy(model, tok, prompt_ids: list[int], stop_ids: set[int], limit: int) -> list[int]:
    logits, _, cache = model(mx.array([prompt_ids], dtype=mx.int32))
    out = []
    for _ in range(limit):
        nxt = int(mx.argmax(logits[0, -1]).item())
        if nxt in stop_ids:
            break
        out.append(nxt)
        logits, _, cache = model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
    return out


def translation_probe(spec, model, phase_name: str, n: int) -> dict:
    """Pretrain checkpoints are probed with the ``<|en|>…<|zh|>`` format they
    were trained on; SFT checkpoints with the chat prompt."""
    import sacrebleu
    from tokenizers import Tokenizer
    from chronos.data.corpus.evalsets import load_eval_sets
    from chronos.data.corpus.shards import SFT_PROMPT_EN_ZH
    tok = Tokenizer.from_file(os.path.join(spec["tokenizer_dir"], "tokenizer.json"))
    tid = tok.token_to_id
    stop = {tid(t) for t in ("<|endoftext|>", "<|im_end|>", "<|en|>", "<|zh|>")}
    raw = (spec.get("extra") or {}).get("raw_root")
    pairs = load_eval_sets(raw, ["flores_dev"]).get("flores_dev", [])[:n] if raw else []
    if not pairs:
        return {"probe": "skipped: FLORES+ dev not found"}
    out, t0 = {}, time.monotonic()
    directions = [("en_zh", 0, 1)] + ([("zh_en", 1, 0)] if phase_name == "pretrain" else [])
    for name, si, ti in directions:
        hyps, samples = [], []
        for pair in pairs:
            src, ref = pair[si], pair[ti]
            if phase_name == "pretrain":
                tags = ("<|en|>", "<|zh|>") if name == "en_zh" else ("<|zh|>", "<|en|>")
                prompt = [tid(tags[0])] + tok.encode(src, add_special_tokens=False).ids + [tid(tags[1])]
            else:
                text = f"<|im_start|>user\n{SFT_PROMPT_EN_ZH}{src}<|im_end|>\n<|im_start|>assistant\n"
                prompt = tok.encode(text, add_special_tokens=False).ids
            src_len = len(tok.encode(src, add_special_tokens=False).ids)
            hyp = tok.decode(greedy(model, tok, prompt, stop, int(1.6 * src_len) + 16), skip_special_tokens=True)
            hyps.append(hyp)
            if len(samples) < 6:
                samples.append({"src": src, "ref": ref, "hyp": hyp})
        refs = [p[ti] for p in pairs]
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh" if ti == 1 else "13a")
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        out[name] = {"n": len(pairs), "bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2),
                     "samples": samples}
    out["probe_s"] = round(time.monotonic() - t0, 1)
    out["probe_set"] = "flores_dev"
    return out


def export(model, config, spec, out_path: str, stage: str) -> str:
    from chronos.model.checkpoint import tokenizer_metadata
    from chronos.mlx.training.trainer import _save_mlx_checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _save_mlx_checkpoint(model, config, out_path, stage=stage)
    cfg_path = out_path[:-4] + ".config.json"
    meta = json.load(open(cfg_path, encoding="utf-8"))
    meta["tokenizer"] = tokenizer_metadata(os.path.abspath(spec["tokenizer_dir"]))
    json.dump(meta, open(cfg_path, "w", encoding="utf-8"), indent=2, sort_keys=True)
    return out_path


def inspect_run(run_dir: str, *, samples: int = 24, n_batches: int = 8, export_path: str | None = None,
                device: str = "cpu") -> dict:
    t0 = time.monotonic()
    spec, config, model, state = load_run_model(run_dir, device=device)
    phase = _phase(spec, state)
    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_dir": os.path.abspath(run_dir),
              "phase": phase["name"], "step": int(state["phase_step"]), "tokens_seen": int(state["tokens_seen"]),
              "device": device, "compute_dtype": model._chronos_precision.compute,
              "precision": model._chronos_precision.as_dict(),
              "train_dtype": spec.get("dtype"), "load_s": round(time.monotonic() - t0, 1)}
    report["held_out"] = held_out(spec, config, model, phase, n_batches)
    if samples:
        report["translation"] = translation_probe(spec, model, phase["name"], samples)
    if export_path:
        report["exported"] = export(model, config, spec, export_path, phase["name"])
    report["elapsed_s"] = round(time.monotonic() - t0, 1)
    out_dir = os.path.join(run_dir, "inspections")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{phase['name']}_s{report['step']}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--samples", type=int, default=24, help="FLORES+ dev sentences per direction (0 = skip)")
    p.add_argument("--batches", type=int, default=8, help="validation micro-batches")
    p.add_argument("--export", default=None, help="write a serveable .pth here")
    p.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    a = p.parse_args(argv)
    print(json.dumps(inspect_run(a.run_dir, samples=a.samples, n_batches=a.batches, export_path=a.export,
                                 device=a.device), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
