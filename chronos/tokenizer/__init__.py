"""Chronos Tokenizer Builder.

``build_tokenizer(paths, out_dir, vocab_size)`` extracts text from any user-
selected datasets, trains the Chronos BPE (script-aware pre-tokenization,
tempered language balance, byte fallback), evaluates it and writes both the
native ``chronos_tokenizer.json`` and an HF-compatible ``tokenizer.json``.
"""
from __future__ import annotations

import json
import os
import time

from .evaluate import evaluate_tokenizer
from .extract import ExtractConfig, extract_texts
from .model import DEFAULT_SPECIAL_TOKENS, ChronosTokenizer, special_tokens_with
from .trainer import BPETrainerConfig, train_bpe

__all__ = ["build_tokenizer", "ChronosTokenizer", "BPETrainerConfig", "ExtractConfig", "special_tokens_with"]


def build_tokenizer(paths: list[str], out_dir: str, *, vocab_size: int = 16384,
                    extract: ExtractConfig | None = None, trainer: BPETrainerConfig | None = None,
                    special_tokens=DEFAULT_SPECIAL_TOKENS, heldout_fraction: float = 0.02,
                    log=print) -> ChronosTokenizer:
    t0 = time.time()
    extract = extract or ExtractConfig()
    result = extract_texts(paths, extract, log=log)
    train_split, heldout = {}, {}
    for bucket, texts in result.texts_by_bucket.items():
        cut = max(1, int(len(texts) * heldout_fraction))
        heldout[bucket], train_split[bucket] = texts[:cut], texts[cut:]
    config = trainer or BPETrainerConfig(vocab_size=vocab_size, special_tokens=tuple(special_tokens))
    config.vocab_size = vocab_size
    config.special_tokens = tuple(special_tokens)
    log(f"[tokenizer] training vocab={vocab_size} on "
        f"{ {b: len(t) for b, t in train_split.items()} } texts ({time.time() - t0:.0f}s)")
    trained = train_bpe(train_split, config,
                        progress=lambda n, total: log(f"[tokenizer] vocab {n}/{total}") if n % 2000 < 4 else None)
    tok = ChronosTokenizer.from_trained(trained, special_tokens)
    report = evaluate_tokenizer(tok, heldout)
    tok.metadata.update({
        "trainer": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()
                    if k != "special_tokens"},
        "train_stats": trained.stats,
        "heldout_eval": report,
        "sources": result.per_source,
        "elapsed_s": round(time.time() - t0, 1),
    })
    tok.save(out_dir)
    log(f"[tokenizer] saved {out_dir}: {json.dumps(report, ensure_ascii=False)}")
    return tok


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Train a Chronos tokenizer from selected datasets")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--per-source", type=int, default=400_000)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--extra-special", nargs="*", default=(), metavar="TOKEN",
                        help="task control tokens placed at ids 3, 4, … (e.g. language tags)")
    args = parser.parse_args(argv)
    build_tokenizer(
        args.paths, args.out, vocab_size=args.vocab_size,
        extract=ExtractConfig(per_source_texts=args.per_source, fields=args.fields),
        trainer=BPETrainerConfig(vocab_size=args.vocab_size, language_temperature=args.temperature),
        special_tokens=special_tokens_with(args.extra_special),
        log=lambda m: print(m, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
