"""Why input is split into sentences: k consecutive sentences as one input, whole vs split.

usage: python eval/boundary_long_input.py <pairs.jsonl> <out.json>

120 blocks of k = 1, 2, 3, 4, 6 consecutive pairs; the length limit is lifted for
"whole", and "split" translates each sentence of the block on its own.
"""
import json
import os
import sys

import sacrebleu

pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flores, out_path = sys.argv[1:3]
sys.path.insert(0, pkg)
import translate as T  # noqa: E402

tr = T.Translator(pkg, backend="mlx")
tr.generation["max_source_tokens"] = 10**9
rows = [json.loads(line) for line in open(flores, encoding="utf-8")]
ntok = lambda s: len(tr.tokenizer.encode(s, add_special_tokens=False).ids)  # noqa: E731
res = {}
for k in (1, 2, 3, 4, 6):
    blocks = [rows[i:i + k] for i in range(0, 120 * k, k)][:120]
    src = [" ".join(r["en"] for r in b) for b in blocks]
    ref = ["".join(r["zh"] for r in b) for b in blocks]
    whole = [tr.translate_sentence(s) for s in src]
    split = ["".join(tr.translate_sentence(r["en"]) for r in b) for b in blocks]
    toks = sorted(ntok(s) for s in src)
    res[k] = {"blocks": len(blocks), "src_tokens_median": toks[len(toks) // 2], "src_tokens_max": toks[-1],
        "whole": {"bleu": round(sacrebleu.corpus_bleu(whole, [ref], tokenize="zh").score, 2),
            "chrf": round(sacrebleu.corpus_chrf(whole, [ref]).score, 2),
            "hyp_ref_len_ratio": round(sum(map(len, whole)) / sum(map(len, ref)), 3)},
        "split": {"bleu": round(sacrebleu.corpus_bleu(split, [ref], tokenize="zh").score, 2),
            "chrf": round(sacrebleu.corpus_chrf(split, [ref]).score, 2),
            "hyp_ref_len_ratio": round(sum(map(len, split)) / sum(map(len, ref)), 3)},
        "sample_whole": whole[0][:300]}
    print(k, json.dumps({x: res[k][x] for x in ("src_tokens_median", "src_tokens_max", "whole", "split")}), flush=True)
json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
