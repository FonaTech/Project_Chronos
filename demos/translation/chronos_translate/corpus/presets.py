"""Corpus presets and the ``python -m chronos_translate.corpus.presets`` CLI.

A preset is only a list of ``PairSource`` objects plus evaluation sets; the
pipeline itself is task-agnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .clean import CleanConfig
from .evalsets import load_eval_sets
from .pipeline import PairSource, build_corpus


def translation_enzh_sources(raw_root: str) -> list[PairSource]:
    sources: list[PairSource] = []
    # Priority: curated/natural sources first so duplicates keep their copy.
    opus = os.path.join(raw_root, "Helsinki-NLP__opus-100", "en-zh", "train-00000-of-00001.parquet")
    if os.path.exists(opus):
        sources.append(PairSource("opus100", opus, "pairs_parquet", ("translation.zh", "translation.en"),
                                  priority=0, weight=1.5, rows_per_unit=250_000))
    wmt_dir = os.path.join(raw_root, "wmt__wmt19__zh-en", "zh-en")
    if os.path.isdir(wmt_dir):
        for name in sorted(os.listdir(wmt_dir)):
            if name.startswith("train-") and name.endswith(".parquet"):
                sources.append(PairSource(f"wmt19_{name.split('-')[1]}", os.path.join(wmt_dir, name),
                                          "pairs_parquet", ("translation.zh", "translation.en"),
                                          priority=1, weight=1.0, rows_per_unit=1_000_000))
    iic = os.path.join(raw_root, "iic__WMT-Chinese-to-English-Machine-Translation-Training-Corpus",
                       "wmt_zh_en_training_corpus.csv")
    if os.path.exists(iic):
        sources.append(PairSource("iic_wmt", iic, "pairs_csv", ("0", "1"),
                                  clean=CleanConfig(unsegment_zh=True, detokenize_en=True),
                                  priority=2, weight=1.0, rows_per_unit=1_000_000))
    return sources


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the cleaned, deduplicated pair corpus")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--valid-per-mille", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None, help="source-name prefixes to include")
    args = parser.parse_args(argv)
    sources = translation_enzh_sources(args.raw_root)
    if args.only:
        sources = [s for s in sources if any(s.name.startswith(p) for p in args.only)]
    evals = load_eval_sets(args.raw_root)
    print(f"[corpus] sources={[s.name for s in sources]} eval={ {k: len(v) for k, v in evals.items()} }", flush=True)
    manifest = build_corpus(sources, evals, args.out, workers=args.workers,
                            valid_per_mille=args.valid_per_mille,
                            log=lambda m: print(m, flush=True))
    print(json.dumps({k: manifest[k] for k in ("rows_after_clean", "duplicates", "eval_leak_removed",
                                               "final_train", "final_valid")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
