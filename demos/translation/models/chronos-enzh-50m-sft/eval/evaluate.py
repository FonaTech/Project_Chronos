#!/usr/bin/env python3
"""Score chronos-enzh-50m-sft on English -> Chinese pairs (sacrebleu==2.6.0).

    python eval/evaluate.py    # the author-written self-test
    python eval/evaluate.py --pairs flores.jsonl    # {"en": ..., "zh": ...} per line

BLEU uses ``tokenize="zh"``; chrF is sacrebleu's default.  Decoding is the
shipped greedy setting.  ``--no-limit`` also scores sentences above
``max_source_tokens`` (they are refused by default and reported by count).
``--split`` scores the driver's default path (sentence split, then join).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import translate as T  # noqa: E402

BUCKETS = ((0, 32), (33, 64), (65, 96), (97, 128), (129, 10**9))


def load_pairs(args) -> list[dict]:
    if args.pairs:
        with open(args.pairs, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        return [{"id": str(i), "cat": "pairs", "src": r["en"], "ref": r["zh"]} for i, r in enumerate(rows)]
    with open(os.path.join(HERE, "selftest.json"), encoding="utf-8") as fh:
        return json.load(fh)["en_zh"]


def score(rows: list[dict]) -> dict:
    import sacrebleu
    done = [r for r in rows if "hyp" in r]
    if not done:
        return {"n": 0}
    hyps, refs = [r["hyp"] for r in done], [[r["ref"] for r in done]]
    return {"n": len(done), "bleu": round(sacrebleu.corpus_bleu(hyps, refs, tokenize="zh").score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, refs).score, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", help='JSONL with {"en": ..., "zh": ...} per line (default: selftest.json)')
    ap.add_argument("--n", type=int, default=0, help="first N pairs only")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-limit", action="store_true", help="also translate sentences above max_source_tokens")
    ap.add_argument("--split", action="store_true", help="split each item into sentences first (translate.py default)")
    ap.add_argument("--out", help="write every source/reference/hypothesis here (JSON)")
    args = ap.parse_args(argv)
    rows = load_pairs(args)[: args.n or None]
    tr = T.Translator(backend=args.backend, device=args.device)
    if args.no_limit:
        tr.generation["max_source_tokens"] = 10**9
    t0 = time.perf_counter()
    for r in rows:
        r["src_tokens"] = len(tr.tokenizer.encode(r["src"], add_special_tokens=False).ids)
        try:
            r["hyp"] = tr.translate(r["src"], split=True) if args.split else tr.translate_sentence(r["src"])
        except T.BoundaryError as exc:
            r["refused"] = str(exc)
    report = {"backend": tr.backend, "device": tr.device, "dtype": tr.dtype, "pairs": os.path.basename(args.pairs) if args.pairs else "selftest.json", "split": args.split,
        "seconds": round(time.perf_counter() - t0, 1), "refused": sum("refused" in r for r in rows),
        "all": score(rows), "by_source_tokens": {}, "by_category": {}}
    for lo, hi in BUCKETS:
        part = [r for r in rows if lo <= r["src_tokens"] <= hi]
        if part:
            report["by_source_tokens"][f"{lo}-{hi if hi < 10**9 else 'inf'}"] = score(part) | {"total": len(part)}
    for cat in sorted({r["cat"] for r in rows}):
        report["by_category"][cat] = score([r for r in rows if r["cat"] == cat])
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "rows": rows}, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
