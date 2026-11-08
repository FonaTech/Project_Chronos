"""Why zh->en is disabled: score the raw model on zh->en (the driver's refusal is bypassed).

usage: python eval/boundary_zh_en.py <pairs.jsonl> <n> <out.json> [other.safetensors]

Two prompt formats: the SFT chat template with a "translate into English" instruction,
and the pretraining tags <|zh|>...<|en|>.  A second weights file (e.g. the pretrain
checkpoint) is the control.  BLEU uses sacrebleu's 13a tokenizer for English.
"""
import json
import os
import sys
import time

import sacrebleu

pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flores, n, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, pkg)
import translate as T  # noqa: E402

other = sys.argv[4] if len(sys.argv) > 4 else None
tr = T.Translator(pkg, backend="torch", device="cpu", weights=other, verify=other is None)
tok = tr.tokenizer
tags = {t: tok.token_to_id(t) for t in ("<|zh|>", "<|en|>", "<|im_end|>")}
tr.stop_ids |= {v for v in tags.values() if v is not None}
rows = [json.loads(line) for line in open(flores, encoding="utf-8")][:n]
self_rows = json.load(open(os.path.join(pkg, "eval", "selftest.json"), encoding="utf-8"))["zh_en_probe"]


def chat(src):
    text = f"<|im_start|>user\n翻译成英文，只输出译文：\n{src}<|im_end|>\n<|im_start|>assistant\n"
    return tok.encode(text, add_special_tokens=False).ids


def tagged(src):
    return [tags["<|zh|>"]] + tok.encode(src, add_special_tokens=False).ids + [tags["<|en|>"]]


def run(fmt, srcs):
    hyps = []
    for s in srcs:
        n_src = len(tok.encode(s, add_special_tokens=False).ids)
        ids = tr._greedy_torch(fmt(s), min(160, int(1.6 * n_src) + 16))
        hyps.append(tok.decode(ids, skip_special_tokens=True).strip())
    return hyps


res = {"tags_in_vocab": tags, "n": len(rows)}
for name, fmt in (("chat_zh_en_instruction", chat), ("pretrain_tags", tagged)):
    t0 = time.perf_counter()
    hyps = run(fmt, [r["zh"] for r in rows])
    refs = [[r["en"] for r in rows]]
    res[name] = {"bleu_13a": round(sacrebleu.corpus_bleu(hyps, refs).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, refs).score, 2), "seconds": round(time.perf_counter() - t0, 1),
        "flores_samples": [{"src": r["zh"], "ref": r["en"], "hyp": h} for r, h in list(zip(rows, hyps))[:5]],
        "selftest": [{"src": r["src"], "ref": r["ref"], "hyp": h}
            for r, h in zip(self_rows, run(fmt, [r["src"] for r in self_rows]))]}
json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) or "bleu_13a" not in v else
    {x: v[x] for x in ("bleu_13a", "chrf", "seconds")}) for k, v in res.items()}, ensure_ascii=False))
