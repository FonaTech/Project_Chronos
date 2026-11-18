"""Greedy-decoding repetition probe for an arm's final ``.pth``.

Loads a Torch state dict through ``chronos.model.checkpoint``, converts it to
MLX with ``ChronosMLXModel.from_chronos_pytorch`` (so experts are included),
and greedy-decodes each prompt in ``prompts.jsonl``. The 15M arm averages in
``../README.md`` were produced with this script at ``--max-tokens 60`` against
each arm's ``run_<arm>/arm_<arm>.pth`` and the training tokenizer under
``../run_<arm>/arm_<arm>.config.json`` → ``tokenizer`` sidecar.

Example::

    python probes/probe_arm.py \
        --pth run_C_periter/arm_C_periter.pth \
        --tokenizer <path to tokenizer.json> \
        --prompts probes/prompts.jsonl --max-tokens 60 \
        --out probes/C_periter.json
"""
import argparse
import collections
import json
import sys

import mlx.core as mx
import numpy as np
import torch
from tokenizers import Tokenizer


def load_from_pth(pth_path):
    from chronos.model.checkpoint import chronos_config_from_checkpoint
    from chronos.mlx.model import ChronosMLXModel
    from chronos.model.model_chronos import ChronosForCausalLM
    cfg, _ = chronos_config_from_checkpoint(pth_path, require_unsniffable=False)
    sd = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    pt = ChronosForCausalLM(cfg)
    missing, unexpected = pt.load_state_dict(sd, strict=False)
    if missing:
        print("[warn] missing", len(missing), "ex:", missing[:3], file=sys.stderr)
    if unexpected:
        print("[warn] unexpected", len(unexpected), "ex:", unexpected[:3], file=sys.stderr)
    pt.eval()
    mlx_model = ChronosMLXModel.from_chronos_pytorch(pt, cfg, include_experts=True)
    mlx_model.eval()
    return mlx_model, cfg


def greedy_sample(model, prompt_ids, max_tokens, eos_id):
    ids = mx.array([prompt_ids], dtype=mx.int32)
    gen = []
    for _ in range(max_tokens):
        logits, *_ = model(ids, logits_dtype=mx.float32)
        nxt = int(mx.argmax(logits[0, -1]).item())
        gen.append(nxt)
        if nxt == eos_id:
            break
        ids = mx.concatenate([ids, mx.array([[nxt]], dtype=mx.int32)], axis=1)
    return gen


def stats(gen, k=4):
    n = len(gen)
    if n == 0:
        return dict(n=0, unique_ratio=0, max_run=0, max_4gram=0)
    uniq = len(set(gen))
    run = mr = 1
    for a, b in zip(gen, gen[1:]):
        run = run + 1 if a == b else 1
        mr = max(mr, run)
    ng = (collections.Counter(tuple(gen[i:i + k]) for i in range(n - k + 1))
          if n >= k else {})
    return dict(n=n, unique_ratio=round(uniq / n, 3), max_run=mr,
                max_4gram=max(ng.values()) if ng else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    model, cfg = load_from_pth(a.pth)
    eos = int(getattr(cfg, "eos_token_id", 0) or 0)
    rows = [json.loads(line) for line in open(a.prompts) if line.strip()]
    items = []
    for r in rows:
        g = greedy_sample(model, tok.encode(r["prompt"]).ids, a.max_tokens, eos)
        items.append(dict(id=r.get("id"), prompt=r["prompt"],
                          gen=tok.decode(g), **stats(g)))
    avg = {k: round(float(np.mean([it[k] for it in items])), 3)
           for k in ("unique_ratio", "max_run", "max_4gram", "n")}
    print("PTH", a.pth)
    print("AVG", json.dumps(avg))
    json.dump(dict(pth=a.pth, items=items, avg=avg),
              open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
