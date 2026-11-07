"""Cluster-locality training A/B: does the constraint keep a stream inside one expert cluster?

Two tiny Chronos models start from the same initialisation and train on the
same synthetic multi-topic streams (MLX, the production training loss),
one with ``lambda_cluster_locality = 0`` and one with it on.  Each topic
uses its own sub-vocabulary and Markov chain, so a stream has a latent
domain the router can learn to specialise on.  Reported, all measured:

* held-out CE (the constraint must not cost quality);
* cluster window entropy / clusters touched per window / marginal entropy
  over every dispatch, recurrent iterations included;
* topic → cluster purity (how much of a topic's routing mass lands in its
  dominant cluster) — the specialisation the constraint should produce;
* lazy greedy decoding with experts paged per cluster (one SSD read per
  cluster file): cluster reads and expert misses per token under the same
  budget, LRU and next-use planned residency.

    python tools/bench_cluster_locality.py --steps 300 --out r.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

SPECIAL = 4                               # ids 0..3 reserved


def make_topics(num_topics: int, sub_vocab: int, seed: int):
    """Per topic: disjoint token range and a sparse random transition table."""
    import numpy as np
    rng = np.random.default_rng(seed)
    topics = []
    for t in range(num_topics):
        lo = SPECIAL + t * sub_vocab
        succ = rng.integers(0, sub_vocab, size=(sub_vocab, 3))      # 3 likely successors per token
        topics.append((lo, succ))
    return topics


def sample_stream(topics, length: int, segment: int, rng):
    """Concatenated single-topic segments of ``segment`` tokens (topic per segment)."""
    ids, tags = [], []
    while len(ids) < length:
        t = int(rng.integers(len(topics)))
        lo, succ = topics[t]
        cur = int(rng.integers(succ.shape[0]))
        for _ in range(segment):
            ids.append(lo + cur)
            tags.append(t)
            cur = int(succ[cur, rng.integers(3)]) if rng.random() < 0.9 else int(rng.integers(succ.shape[0]))
    return ids[:length], tags[:length]


def tiny_config(args, lam: float):
    from chronos.model.config import ChronosConfig
    return ChronosConfig(
        hidden_size=args.hidden, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=SPECIAL + args.topics * args.sub_vocab, num_experts=args.experts, num_experts_per_tok=2,
        num_shared_experts=1, moe_intermediate_size=args.hidden, max_position_embeddings=args.seq * 2,
        recurrent_depth=3, prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
        recurrent_attention_mode="shared_first", sliding_window_size=args.seq,
        lambda_cluster_locality=lam, cluster_locality_clusters=args.clusters,
        cluster_locality_window=args.segment if args.window is None else args.window, lookahead_steps=2)


def build_pair(args):
    """Two MLX models with identical initial weights (one Torch init, converted twice)."""
    import torch
    from chronos.mlx.model import ChronosMLXModel
    from chronos.model.model_chronos import ChronosForCausalLM
    models = {}
    for arm, lam in (("baseline", 0.0), ("cluster", args.lam)):
        cfg = tiny_config(args, lam)
        torch.manual_seed(args.seed)
        pt = ChronosForCausalLM(cfg)
        models[arm] = (ChronosMLXModel.from_chronos_pytorch(pt, cfg), cfg)
    return models


def train(model, cfg, batches, args, log):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from chronos.mlx.training.trainer import _ce_stage_loss
    model.train()
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    grad_fn = nn.value_and_grad(model, lambda m, x: _ce_stage_loss(m, x, x))
    curve = []
    for step, batch in enumerate(batches):
        x = mx.array(batch, dtype=mx.int32)
        loss, grads = grad_fn(model, x)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        terms = {k: float(v) for k, v in (getattr(model, "_last_loss_components", {}) or {}).items()}
        if step % max(1, args.steps // 10) == 0 or step == len(batches) - 1:
            curve.append({"step": step, "loss": round(float(loss), 4),
                          "ce": round(terms.get("primary_loss", float("nan")), 4),
                          "cluster_term": round(terms.get("cluster_locality_loss", 0.0), 5)})
            log(f"  step {step:4d} loss {float(loss):.4f} ce {terms.get('primary_loss', 0):.4f} "
                f"cluster {terms.get('cluster_locality_loss', 0.0):.5f}")
    model.eval()
    return curve


def routing_metrics(model, cfg, eval_ids, eval_tags, args):
    """Held-out CE and cluster metrics over every dispatch (recurrent iterations included)."""
    import mlx.core as mx
    import numpy as np
    from chronos.expert_clusters import expert_to_cluster, training_clusters
    from chronos.mlx.training.trainer import _collect_router_probs, _masked_lm_loss
    x = mx.array(eval_ids, dtype=mx.int32)
    logits, _la, _c = model(x, logits_dtype=mx.float32)
    ce = float(_masked_lm_loss(logits, x))
    probs = np.array(_collect_router_probs(model).astype(mx.float32))          # [B, S, V, E]
    B, S, V, E = probs.shape
    assign = np.array(expert_to_cluster(training_clusters(cfg), E))
    C = int(assign.max()) + 1
    member = np.eye(C)[assign]                                                  # [E, C]
    w = args.segment
    n = S // w
    mass = (probs @ member)[:, : n * w].reshape(B, n, w, V, C).mean(axis=(2, 3)).reshape(-1, C)
    ent = lambda p: -(np.clip(p, 1e-12, None) * np.log(np.clip(p, 1e-12, None))).sum(-1)
    top = np.argsort(-probs, axis=-1)[..., :cfg.num_experts_per_tok]           # [B, S, V, k]
    touched = np.zeros((B, n, C), dtype=bool)
    for b in range(B):
        for j in range(n):
            touched[b, j, np.unique(assign[top[b, j * w:(j + 1) * w]])] = True
    tags = np.array(eval_tags)                                                  # [B, S]
    cmass = (probs @ member).mean(axis=2)                                       # [B, S, C]
    purity = []
    for t in range(args.topics):
        sel = cmass[tags == t]
        if len(sel):
            purity.append(float(sel.mean(axis=0).max()))
    usage = np.bincount(top.reshape(-1), minlength=E) / top.size
    return {
        "heldout_ce": round(ce, 4),
        "cluster_window_entropy": round(float(ent(mass).mean() / math.log(C)), 4),
        "cluster_marginal_entropy": round(float(ent(mass.mean(0)) / math.log(C)), 4),
        "clusters_per_window": round(float(touched.sum(-1).mean()), 3),
        "topic_cluster_purity": round(float(np.mean(purity)), 4),
        "expert_usage_entropy": round(float(ent(usage) / math.log(E)), 4),
        "num_clusters": C,
    }


def lazy_decode(model, cfg, prompts, args, workdir):
    """Greedy lazy decoding, experts paged per training cluster; LRU vs planned residency."""
    import mlx.core as mx
    from chronos.expert_clusters import training_clusters
    from chronos.mlx.inference import ChronosMLXInferenceEngine
    cfg.recommended_resident_experts = args.hot
    cfg.recommended_ram_experts = args.warm
    engine = ChronosMLXInferenceEngine(model, cfg, ssd_dir=os.path.join(workdir, "experts"))
    clusters = training_clusters(cfg)
    out = {}
    try:
        engine.setup_with_clusters(clusters, warm_expert_ids=[])
        store = engine.store
        outputs = {}
        for residency in ("window", "planned"):
            tot = {"tokens": 0, "sync_ssd_loads": 0, "cluster_loads": 0, "seconds": 0.0}
            outputs[residency] = []
            for p in prompts:
                store.set_protected_experts([])
                store.set_eviction_ranker(None)
                store.replace_live_experts_with_placeholders()
                with store._lock:
                    store._warm.clear()
                    store._warm_lru.clear()
                    store._loaded_clusters.clear()
                    store._stats.clear()
                t0 = time.perf_counter()
                toks = list(engine.generate(mx.array([p], dtype=mx.int32), max_new_tokens=args.decode,
                                            temperature=0.0, top_p=1.0, residency=residency))
                tot["seconds"] += time.perf_counter() - t0
                st = engine.last_stats
                outputs[residency].append(toks)
                tot["tokens"] += len(toks)
                tot["sync_ssd_loads"] += int(st.get("sync_ssd_loads", 0))
                tot["cluster_loads"] += int(st.get("cluster_loads", 0))
            n = max(tot["tokens"], 1)
            out[residency] = {**tot, "seconds": round(tot["seconds"], 3),
                              "expert_misses_per_token": round(tot["sync_ssd_loads"] / n, 4),
                              "cluster_reads_per_token": round(tot["cluster_loads"] / n, 4)}
        out["planned_identical_to_window"] = outputs["planned"] == outputs["window"]
    finally:
        engine.stop()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topics", type=int, default=4)
    ap.add_argument("--sub-vocab", type=int, default=24)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--clusters", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--segment", type=int, default=32, help="topic segment length of the synthetic streams")
    ap.add_argument("--window", type=int, default=None,
                    help="cluster_locality_window (default: --segment; 0 = whole sequence)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=0.3, help="lambda_cluster_locality of the constrained arm")
    ap.add_argument("--hot", type=int, default=4, help="hot expert pages (lazy decode budget)")
    ap.add_argument("--warm", type=int, default=8, help="RAM expert pages (lazy decode budget)")
    ap.add_argument("--decode", type=int, default=48)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import mlx.core as mx
    import numpy as np
    log = lambda m: print(m, flush=True)
    topics = make_topics(args.topics, args.sub_vocab, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    batches = [[sample_stream(topics, args.seq, args.segment, rng)[0] for _ in range(args.batch)]
               for _ in range(args.steps)]
    held = [sample_stream(topics, args.seq, args.segment, np.random.default_rng(10_000 + i)) for i in range(16)]
    eval_ids, eval_tags = [h[0] for h in held], [h[1] for h in held]
    prompts = []
    for i in range(args.prompts):                          # one-topic prompts, round-robin over topics
        lo, succ = topics[i % args.topics]
        r = np.random.default_rng(20_000 + i)
        cur, p = int(r.integers(succ.shape[0])), []
        for _ in range(12):
            p.append(lo + cur)
            cur = int(succ[cur, r.integers(3)])
        prompts.append(p)

    report = {"label": "measured", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "settings": vars(args),
              "data": "synthetic: per-topic disjoint sub-vocabulary + sparse Markov chain; "
                      f"segments of {args.segment} tokens, one topic each", "arms": {}}
    with mx.stream(mx.cpu):
        for arm, (model, cfg) in build_pair(args).items():
            log(f"[{arm}] lambda_cluster_locality={cfg.lambda_cluster_locality}")
            mx.random.seed(args.seed)
            t0 = time.perf_counter()
            curve = train(model, cfg, batches, args, log)
            train_s = time.perf_counter() - t0
            metrics = routing_metrics(model, cfg, eval_ids, eval_tags, args)
            log(f"[{arm}] " + "  ".join(f"{k} {v}" for k, v in metrics.items()))
            workdir = tempfile.mkdtemp(prefix=f"chronos_cluster_{arm}_")
            try:
                decode = lazy_decode(model, cfg, prompts, args, workdir)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            for res in ("window", "planned"):
                d = decode[res]
                log(f"[{arm}] lazy {res:7s} expert misses/token {d['expert_misses_per_token']:.3f}  "
                    f"cluster reads/token {d['cluster_reads_per_token']:.3f}  tokens {d['tokens']}")
            report["arms"][arm] = {"lambda_cluster_locality": cfg.lambda_cluster_locality,
                                   "train_seconds": round(train_s, 2), "curve": curve,
                                   "routing": metrics, "lazy_decode": decode}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
