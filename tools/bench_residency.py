"""Lazy-expert residency benchmark: LRU window vs next-use planning, plain and speculative.

One MLX lazy engine (SSD expert pages under the Inference-tab budget) decodes
the same prompts greedily in up to five modes, alternating per prompt:

* ``plain``        — plain decoding, LRU + protected LookaheadRouter window;
* ``planned``      — plain decoding, next-use eviction + planned prefetch;
* ``spec-window``  — self-speculative, LRU + last forward's routes + lookahead;
* ``spec-planned`` — self-speculative, next-use eviction planned from the
  draft queue, pre-verify prefetch (``chronos.residency_planner``).

Each prompt starts from the same cold store (the residency state is reset
between modes), so SSD loads compare like with like.  Every number in the
report is measured.

    python tools/bench_residency.py --checkpoint path.pth --prompts flores.jsonl --device cpu --out r.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from tools.bench_speculative import _display_path, _sha256, _versions, encode, load_prompts  # noqa: E402

MODES = ("plain", "planned", "spec-window", "spec-planned")


def _reset(engine, warm: int) -> None:
    """Cold, identical starting residency for every (prompt, mode)."""
    store = engine.store
    store.set_protected_experts([])
    store.set_eviction_ranker(None)
    store.replace_live_experts_with_placeholders()
    with store._lock:
        store._warm.clear()
        store._warm_lru.clear()
        store._loaded_clusters.clear()
        store._stats.clear()                      # per-run store counters (cluster/prefetch loads)
    store.warm_up(list(range(min(int(store.num_experts), warm))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--field", default="en")
    ap.add_argument("--format", default="tags:<|en|>,<|zh|>")
    ap.add_argument("--eos", default="<|endoftext|>")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--ratio", type=float, default=1.0, help="lazy expert budget (Inference-tab RAM load ratio)")
    ap.add_argument("--draft-depth", type=int, default=1)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    import mlx.core as mx
    from tokenizers import Tokenizer
    from chronos.model.checkpoint import chronos_config_from_checkpoint, resolve_checkpoint_tokenizer
    from tools.bench_speculative import mlx_runner
    from ui.tabs.inference_tab import _clone_model_cfg

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cfg, _ = chronos_config_from_checkpoint(args.checkpoint)
    cfg = _clone_model_cfg(cfg)
    tok = Tokenizer.from_file(os.path.join(resolve_checkpoint_tokenizer(args.checkpoint, fallback=False),
                                           "tokenizer.json"))
    eos = tok.token_to_id(args.eos)
    prompts = [encode(tok, x, args.format) for x in load_prompts(args.prompts, args.field, args.n)]
    limits = [min(args.max_new, int(1.6 * (len(p) - 2)) + 16) for p in prompts]
    workdir = tempfile.mkdtemp(prefix="chronos_residency_")
    rows = []
    try:
        _decode, info, close = mlx_runner(cfg, args.checkpoint, lazy=True, ratio=args.ratio, device=args.device,
                                          workdir=workdir, dtype=args.dtype)
        engine = close.__self__                       # mlx_runner returns the engine's bound stop()
        warm = int(info["budget"]["effective_vram_expert_budget"])
        try:
            for i, (prompt, limit) in enumerate(zip(prompts, limits)):
                row = {"index": i}
                for mode in modes:
                    _reset(engine, warm)
                    ids = mx.array([prompt], dtype=mx.int32)
                    kw = dict(residency="planned" if mode.endswith("planned") else "window",
                              residency_decay=args.decay)
                    if mode.startswith("spec"):
                        kw.update(speculative=True, draft_depth=args.draft_depth, gamma=args.gamma, gamma_max=8)
                    t0 = time.perf_counter()
                    toks = list(engine.generate(ids, max_new_tokens=limit, temperature=0.0, top_p=1.0,
                                                eos_token_id=eos, **kw))
                    dt = time.perf_counter() - t0
                    st = engine.last_stats
                    row[mode] = {"tokens": toks, "s": dt, "ssd": int(st.get("sync_ssd_loads", 0)),
                                 "hits": int(st.get("cache_hits", 0)), "misses": int(st.get("cache_misses", 0)),
                                 "cluster_loads": int(st.get("cluster_loads", 0)),
                                 "prefetch_loads": int(st.get("prefetch_loads", 0)),
                                 "planned_evictions": int(st.get("planned_evictions", 0)),
                                 "eviction_policy": st.get("eviction_policy"),
                                 "verify_ready_rate": (st.get("residency") or {}).get("verify_ready_rate")}
                rows.append(row)
                print(f"[{i + 1}/{len(prompts)}] " + "  ".join(
                    f"{m} ssd {row[m]['ssd']:4d}" for m in modes), flush=True)
        finally:
            close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    summary = {}
    for mode in modes:
        toks = sum(len(r[mode]["tokens"]) for r in rows)
        secs = sum(r[mode]["s"] for r in rows)
        hits, misses = sum(r[mode]["hits"] for r in rows), sum(r[mode]["misses"] for r in rows)
        summary[mode] = {
            "tokens": toks, "seconds": round(secs, 3), "tok_s": round(toks / max(secs, 1e-9), 2),
            "sync_ssd_loads": sum(r[mode]["ssd"] for r in rows),
            "ssd_loads_per_token": round(sum(r[mode]["ssd"] for r in rows) / max(toks, 1), 4),
            "selection_miss_rate": round(misses / max(hits + misses, 1), 4),
            "prefetch_loads": sum(r[mode]["prefetch_loads"] for r in rows),
            "identical_to_plain": sum(r[mode]["tokens"] == r["plain"]["tokens"] for r in rows) if "plain" in modes
            else None,
        }
        if mode == "spec-planned":
            rates = [r[mode]["verify_ready_rate"] for r in rows if r[mode]["verify_ready_rate"] is not None]
            summary[mode]["verify_ready_rate_mean"] = round(statistics.mean(rates), 4) if rates else None
    for mode in modes:
        s = summary[mode]
        print(f"{mode:12s} tok/s {s['tok_s']:8.2f}  ssd loads {s['sync_ssd_loads']:6d} "
              f"({s['ssd_loads_per_token']:.3f}/tok)  miss rate {s['selection_miss_rate']:.4f}  "
              f"identical {s['identical_to_plain']}/{len(rows)}", flush=True)
    if args.out:
        report = {"label": "measured", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": _versions(),
                  "checkpoint": {"path": _display_path(args.checkpoint), "sha256": _sha256(args.checkpoint)},
                  "settings": {**vars(args), "checkpoint": _display_path(args.checkpoint),
                               "prompts": _display_path(args.prompts),
                               "out": _display_path(args.out), "eos_id": eos, "decoding": "greedy, batch 1",
                               "order": "modes alternate per prompt; residency reset to the same cold start"},
                  "budget": info.get("budget"), "dtype": str(info.get("dtype")), "summary": summary,
                  "rows": [{"index": r["index"], **{m: {k: v for k, v in r[m].items() if k != "tokens"}
                                                    | {"n_tokens": len(r[m]["tokens"])} for m in modes}}
                           for r in rows]}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
