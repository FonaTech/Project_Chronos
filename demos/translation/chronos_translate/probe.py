"""Greedy en↔zh translation probe for a model loaded from a run (FLORES+ dev, BLEU / chrF).

Pretrain checkpoints are probed in the ``<|en|>…<|zh|>`` tag format they
were trained on (both directions); SFT checkpoints use the chat prompt.  Used
by ``TranslationLongRun`` at each pretrain validation and by
``chronos-translate inspect`` (offline, on the MLX CPU device).
"""
from __future__ import annotations

import os
import time

LANG_TAGS = ("<|en|>", "<|zh|>")


def translation_probe(spec: dict, model, phase_name: str, n: int) -> dict:
    import sacrebleu
    from tokenizers import Tokenizer

    from chronos.mlx.training.inspect_checkpoint import greedy

    from .corpus.evalsets import load_eval_sets
    from .corpus.shards import SFT_PROMPT_EN_ZH
    tok = Tokenizer.from_file(os.path.join(spec["tokenizer_dir"], "tokenizer.json"))
    tid = tok.token_to_id
    stop = {tid(t) for t in ("<|endoftext|>", "<|im_end|>", *LANG_TAGS)}
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
                tags = LANG_TAGS if name == "en_zh" else LANG_TAGS[::-1]
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


def main(argv=None) -> int:
    """``chronos-translate inspect``: the kernel inspector plus this probe."""
    import argparse
    import json

    from chronos.mlx.training.inspect_checkpoint import inspect_run
    p = argparse.ArgumentParser(description="Inspect a run checkpoint (CPU) with the en↔zh FLORES+ dev probe")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--samples", type=int, default=24, help="FLORES+ dev sentences per direction (0 = skip)")
    p.add_argument("--batches", type=int, default=8, help="validation micro-batches")
    p.add_argument("--export", default=None, help="write a serveable .pth here")
    p.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    a = p.parse_args(argv)
    report = inspect_run(a.run_dir, samples=a.samples, n_batches=a.batches, export_path=a.export,
                         device=a.device, probes={"translation": translation_probe})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
