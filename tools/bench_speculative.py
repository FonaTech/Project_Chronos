"""Measure self-speculative decoding against plain decoding on the real runtimes.

Every configuration loads one checkpoint once, then decodes the same prompts
twice per prompt (plain greedy, speculative greedy) through the code paths the
Web UI Inference tab uses:

* ``mlx-full``    — ChronosMLXInferenceEngine, all experts resident (DRAM)
* ``mlx-lazy``    — ChronosMLXInferenceEngine, SSD expert pages under a budget
* ``torch-full``  — plain KV-cache loop vs ``speculative_generate`` (MPS/CPU/CUDA)
* ``torch-lazy``  — ChronosInferenceEngine.generate vs .generate_speculative

Greedy outputs must be identical (sync miss policy for lazy Torch); the script
reports wall time, tokens/s, speed-up, acceptance and identical-output counts
as measured numbers, plus the exact settings, so nothing needs re-deriving.

    python tools/bench_speculative.py --checkpoint <ckpt.pth> --prompts <file.jsonl|txt> \
        --configs mlx-full,mlx-lazy --device gpu --out reports/spec_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def load_prompts(path: str, field: str, limit: int) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if path.endswith((".jsonl", ".json")):
                rec = json.loads(line)
                value = rec.get(field) if isinstance(rec, dict) else None
                if not isinstance(value, str):
                    continue
                line = value
            out.append(line)
            if len(out) >= limit:
                break
    return out


def encode(tokenizer, text: str, fmt: str) -> list[int]:
    """``tags:<a>,<b>`` → ``<a> text <b>``; ``chat`` → tokenizer chat template; ``raw`` → text."""
    if fmt.startswith("tags:"):
        a, b = fmt[5:].split(",")
        return ([tokenizer.token_to_id(a)] + tokenizer.encode(text, add_special_tokens=False).ids
                + [tokenizer.token_to_id(b)])
    if fmt == "raw":
        return tokenizer.encode(text, add_special_tokens=False).ids
    raise ValueError("format must be raw or tags:<a>,<b> (chat prompts: pre-render them into the file)")


# ---- runners: each returns (decode(prompt, limit, speculative) -> (tokens, seconds, stats), info) ----------
def mlx_runner(cfg, ckpt, *, lazy: bool, ratio: float, device: str, workdir: str, dtype: str = "auto",
    paging: str = "expert"):
    import mlx.core as mx
    from ui.tabs import inference_tab as it
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.mlx.inference import ChronosMLXInferenceEngine
    from chronos.mlx.model import ChronosMLXModel
    from chronos.precision import resolve_mlx
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    precision = resolve_mlx(dtype, "cpu" if device == "cpu" else "gpu")
    weights = load_checkpoint_state_dict(ckpt, map_location="cpu")
    info = {"dtype": precision.compute}
    if lazy:
        budget = it._bounded_offload_expert_budget(cfg, ratio)
        cfg.recommended_resident_experts = int(budget["effective_vram_expert_budget"])
        cfg.recommended_ram_experts = int(budget["effective_ram_expert_budget"])
        info["budget"] = budget
        pt = ChronosForCausalLM(cfg)
        load_state_dict_controlled(pt, {k: v for k, v in weights.items() if ".mlp.experts." not in k},
                                   allow_missing_substrings=(".mlp.experts.",))
        it._replace_torch_experts_with_placeholders(pt, cfg)
        model = ChronosMLXModel.from_chronos_pytorch(pt.eval(), cfg, include_experts=False, dtype=precision.compute)
        it._replace_mlx_experts_with_placeholders(model)
        cache_dir = os.path.join(workdir, "mlx_expert_cache")
        engine = ChronosMLXInferenceEngine(model, cfg, ssd_dir=cache_dir)
        if not engine.store.attach_cluster_manifest(cache_dir):
            # Page layout: "expert" (one page per expert, the historical benchmark layout), or "cluster" / "auto"
            # (the training partition, what the inference engine uses for cluster-trained checkpoints).
            from chronos.expert_clusters import inference_clusters
            clusters = None if paging == "expert" else inference_clusters(cfg, paging)
            it._rebuild_mlx_lazy_cache(engine.store, cache_dir, None, weights, clusters=clusters)
        warm = int(budget["effective_vram_expert_budget"])
        engine.store.warm_up(list(range(min(int(cfg.num_experts), warm))))
    else:
        pt = ChronosForCausalLM(cfg)
        load_state_dict_controlled(pt, weights)
        model = ChronosMLXModel.from_chronos_pytorch(pt.eval(), cfg, dtype=precision.compute)
        engine = ChronosMLXInferenceEngine(model, cfg)
        engine.store.storage_format = "full_dram"
    del weights, pt

    def decode(prompt, limit, speculative, opts, eos):
        ids = mx.array([prompt], dtype=mx.int32)
        t0 = time.perf_counter()
        toks = list(engine.generate(ids, max_new_tokens=limit, temperature=0.0, top_p=1.0, eos_token_id=eos,
                                    speculative=speculative, **(opts if speculative else {})))
        return toks, time.perf_counter() - t0, dict(engine.last_stats)
    return decode, info, getattr(engine, "stop", lambda: None)


def torch_runner(cfg, ckpt, *, lazy: bool, ratio: float, device: str, workdir: str, miss_policy: str):
    import torch
    from ui.tabs import inference_tab as it
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.precision import place_torch_model
    from chronos.runtime.inference_engine import ChronosInferenceEngine
    from chronos.runtime.speculative import speculative_generate
    weights = load_checkpoint_state_dict(ckpt, map_location="cpu")
    model = ChronosForCausalLM(cfg)
    if lazy:
        load_state_dict_controlled(model, {k: v for k, v in weights.items() if ".mlp.experts." not in k},
                                   allow_missing_substrings=(".mlp.experts.",))
        it._replace_torch_experts_with_placeholders(model, cfg)
    else:
        load_state_dict_controlled(model, weights)
    model, precision = place_torch_model(model, device)
    model.eval()
    info = {"dtype": str(precision.compute)}
    if not lazy:
        del weights

        def sync():
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()

        def decode(prompt, limit, speculative, opts, eos):
            ids = torch.tensor([prompt], device=device)
            sync()
            t0 = time.perf_counter()
            with torch.inference_mode():
                if speculative:
                    out, spec = speculative_generate(model, ids, max_new_tokens=limit, eos_token_id=eos, **opts)
                    toks, stats = out[0, len(prompt):].tolist(), {"speculative": spec.as_dict()}
                else:
                    toks, cache, cur = [], None, ids
                    for _ in range(limit):
                        res = model(cur, past_key_values=cache, use_cache=True)
                        cache = res.past_key_values
                        nxt = int(torch.argmax(res.logits[0, -1]))
                        toks.append(nxt)
                        if nxt == eos:
                            break
                        cur = torch.tensor([[nxt]], device=device)
                    stats = {}
            sync()
            return toks, time.perf_counter() - t0, stats
        return decode, info, lambda: None

    engine = ChronosInferenceEngine(model, cfg, ssd_dir=os.path.join(workdir, "torch_expert_cache"))
    info["budget"] = it._apply_offload_expert_budget(engine, cfg, ratio)
    engine.setup_from_state_dict(weights, warm_expert_ids=[])
    del weights

    def decode(prompt, limit, speculative, opts, eos):
        ids = torch.tensor([prompt], device=device)
        t0 = time.perf_counter()
        if speculative:
            out = engine.generate_speculative(ids, max_new_tokens=limit, eos_token_id=eos,
                                              miss_policy=miss_policy, **opts)
        else:
            out = engine.generate(ids, max_new_tokens=limit, temperature=1.0, top_k=0, top_p=1.0,
                                  do_sample=False, eos_token_id=eos, miss_policy=miss_policy)
        return out[0, len(prompt):].tolist(), time.perf_counter() - t0, dict(engine.last_stats)
    return decode, info, engine.teardown

def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "mlx.core"):
        try:
            out[mod] = __import__(mod, fromlist=["__version__"]).__version__
        except Exception:  # pragma: no cover - optional backend
            pass
    try:
        import subprocess
        out["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True, timeout=5).stdout.strip() or None
    except Exception:  # pragma: no cover - non-macOS
        pass
    return out


def _display_path(path: str) -> str:
    """Repo-relative path, or the basename for files outside the repo (no user paths in reports)."""
    full = os.path.abspath(path)
    return os.path.relpath(full, _repo) if full.startswith(_repo + os.sep) else os.path.basename(full)


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl (use --field) or one prompt per line")
    ap.add_argument("--field", default="en")
    ap.add_argument("--format", default="raw", help="raw | tags:<a>,<b> (wrap each prompt in two control tokens)")
    ap.add_argument("--eos", default="<|endoftext|>")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--configs", default="mlx-full,mlx-lazy,torch-full,torch-lazy")
    ap.add_argument("--mlx-device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--torch-device", default="mps")
    ap.add_argument("--mlx-dtype", default="auto", help="auto = bf16 on GPU, fp32 on CPU")
    ap.add_argument("--only", default="", help="comma-separated prompt indices to run (re-checks)")
    ap.add_argument("--ratio", type=float, default=0.5, help="lazy expert budget (Inference-tab RAM load ratio)")
    ap.add_argument("--miss-policy", default="sync_on_demand")
    ap.add_argument("--draft-depth", type=int, default=1)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="", help="free-text run condition, stored in the report")
    args = ap.parse_args(argv)

    import shutil
    import tempfile
    from tokenizers import Tokenizer
    from chronos.model.checkpoint import chronos_config_from_checkpoint, resolve_checkpoint_tokenizer
    from ui.tabs.inference_tab import _clone_model_cfg
    base_cfg, _ = chronos_config_from_checkpoint(args.checkpoint)
    tok = Tokenizer.from_file(os.path.join(resolve_checkpoint_tokenizer(args.checkpoint, fallback=False),
                                           "tokenizer.json"))
    eos = tok.token_to_id(args.eos)
    texts = load_prompts(args.prompts, args.field, args.n)
    prompts = [encode(tok, x, args.format) for x in texts]
    limits = [min(args.max_new, int(1.6 * (len(p) - 2)) + 16) for p in prompts]
    idx = list(range(len(prompts)))
    if args.only:
        idx = [int(x) for x in args.only.split(",")]
        prompts, limits = [prompts[i] for i in idx], [limits[i] for i in idx]
    opts = {"draft_depth": args.draft_depth, "gamma": args.gamma, "gamma_max": 8, "adaptive_gamma": True}
    report = {"label": "measured", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": _versions(),
              "checkpoint": {"path": _display_path(args.checkpoint), "sha256": _sha256(args.checkpoint)},
              "settings": {**vars(args), **{k: _display_path(getattr(args, k)) for k in ("checkpoint", "prompts", "out")
                                            if getattr(args, k)},
                           "speculative": opts, "eos_id": eos, "decoding": "greedy, batch 1",
                           "n_prompts": len(prompts), "prompt_tokens_mean": round(statistics.mean(map(len, prompts)), 1),
                           "order": "plain/speculative alternate per prompt"},
              "model": {k: getattr(base_cfg, k, None) for k in ("hidden_size", "num_hidden_layers", "recurrent_depth",
                        "num_experts", "num_experts_per_tok", "recurrent_attention_mode", "vocab_size")},
              "results": []}
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        backend, mode = name.split("-")
        workdir = tempfile.mkdtemp(prefix=f"chronos_bench_{name}_")
        cfg = _clone_model_cfg(base_cfg)
        try:
            if backend == "mlx":
                decode, info, close = mlx_runner(cfg, args.checkpoint, lazy=mode == "lazy", ratio=args.ratio,
                                                 device=args.mlx_device, workdir=workdir, dtype=args.mlx_dtype)
                device = args.mlx_device
            else:
                decode, info, close = torch_runner(cfg, args.checkpoint, lazy=mode == "lazy", ratio=args.ratio,
                                                   device=args.torch_device, workdir=workdir,
                                                   miss_policy=args.miss_policy)
                device = args.torch_device
            try:
                for i in range(min(args.warmup, len(prompts))):          # compile / page-in, not measured
                    decode(prompts[i], limits[i], False, opts, eos)
                    decode(prompts[i], limits[i], True, opts, eos)
                rows = []
                for i, (p, lim) in enumerate(zip(prompts, limits)):
                    order = (False, True) if i % 2 == 0 else (True, False)
                    got = {}
                    for spec in order:
                        got[spec] = decode(p, lim, spec, opts, eos)
                    (pt, ps, pst), (st, ss, sst) = got[False], got[True]
                    sp = sst.get("speculative") or {}
                    rows.append({"plain_tokens": len(pt), "spec_tokens": len(st), "plain_s": ps, "spec_s": ss,
                                 "identical": pt == st, "index": idx[i],
                                 "first_diff": next((k for k, (x, y) in enumerate(zip(pt, st)) if x != y),
                                                    None if len(pt) == len(st) else min(len(pt), len(st))),
                                 "accepted": sp.get("accepted", 0),
                                 "proposed": sp.get("proposed", 0), "verify_calls": sp.get("verify_calls", 0),
                                 "draft_calls": sp.get("draft_calls", 0),
                                 "plain_loads": int(pst.get("sync_ssd_loads", 0) or 0),
                                 "spec_loads": int(sst.get("sync_ssd_loads", 0) or 0)})
            finally:
                close()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        tp, ts = sum(r["plain_tokens"] for r in rows), sum(r["spec_tokens"] for r in rows)
        sp_, ss_ = sum(r["plain_s"] for r in rows), sum(r["spec_s"] for r in rows)
        acc, prop = sum(r["accepted"] for r in rows), sum(r["proposed"] for r in rows)
        res = {"config": name, "device": device, **info, "prompts": len(rows),
               "identical": sum(r["identical"] for r in rows),
               "plain_tokens": tp, "spec_tokens": ts, "plain_s": round(sp_, 3), "spec_s": round(ss_, 3),
               "plain_tok_s": round(tp / sp_, 2), "spec_tok_s": round(ts / ss_, 2),
               "speedup": round((ts / ss_) / (tp / sp_), 3),
               "speedup_median_per_prompt": round(statistics.median(r["plain_s"] / r["spec_s"] for r in rows), 3),
               "acceptance_rate": round(acc / prop, 4) if prop else None,
               "tokens_per_verify": round(ts / max(1, sum(r["verify_calls"] for r in rows)), 3),
               "plain_ssd_loads": sum(r["plain_loads"] for r in rows),
               "spec_ssd_loads": sum(r["spec_loads"] for r in rows), "rows": rows}
        report["results"].append(res)
        print(f"{name:11s} {device:4s} identical {res['identical']}/{len(rows)}  plain {res['plain_tok_s']:8.2f} tok/s  "
              f"spec {res['spec_tok_s']:8.2f} tok/s  x{res['speedup']:.3f}  acc {res['acceptance_rate']}  "
              f"tok/verify {res['tokens_per_verify']}  loads {res['plain_ssd_loads']}/{res['spec_ssd_loads']}",
              flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
