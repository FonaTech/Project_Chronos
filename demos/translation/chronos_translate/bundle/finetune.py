"""Recipient-side data preparation and fine-tune launch for a bundle."""
from __future__ import annotations

import json
import os

from .builder import load_bundle_manifest

KINDS = {".jsonl": "pairs_jsonl", ".json": "pairs_jsonl", ".csv": "pairs_csv", ".tsv": "pairs_tsv",
         ".parquet": "pairs_parquet"}


def _ensure_valid_split(pairs_dir: str, min_valid: int, log=print) -> dict:
    """Small user sets can hash every row into train; move ``min_valid`` rows to valid."""
    import pyarrow.parquet as pq
    names = sorted(os.listdir(pairs_dir))
    train = [n for n in names if n.startswith("train.") and n.endswith(".parquet")]
    valid = [n for n in names if n.startswith("valid.") and n.endswith(".parquet")]
    rows = {"train": sum(pq.ParquetFile(os.path.join(pairs_dir, n)).metadata.num_rows for n in train),
            "valid": sum(pq.ParquetFile(os.path.join(pairs_dir, n)).metadata.num_rows for n in valid)}
    if rows["valid"] >= min_valid or not train:
        return rows
    need = min(min_valid - rows["valid"], max(1, rows["train"] // 10))
    name = train[-1]
    table = pq.read_table(os.path.join(pairs_dir, name))
    need = min(need, max(0, table.num_rows - 1))
    if need <= 0:
        return rows
    keep, move = table.slice(0, table.num_rows - need), table.slice(table.num_rows - need)
    pq.write_table(move, os.path.join(pairs_dir, "valid." + name[len("train."):]), compression="zstd")
    pq.write_table(keep, os.path.join(pairs_dir, name), compression="zstd")
    log(f"[prepare] moved {need} pairs to the validation split (set too small for the hash split)")
    return {"train": rows["train"] - need, "valid": rows["valid"] + need}


def prepare_data(bundle_dir: str, inputs: list[str], out_dir: str, *, zh_field: str = "zh", en_field: str = "en",
                 unsegment_zh: bool = False, detokenize_en: bool = False, workers: int = 4,
                 valid_per_mille: int | None = None, min_free_gb: float = 1.0, log=print) -> dict:
    """User parallel data → cleaned pairs → MLX shards tokenized with the bundle tokenizer.

    ``inputs`` are files (``.jsonl``/``.csv``/``.tsv``/``.parquet``) with a
    Chinese and an English column.  FLORES+ (shipped in the bundle) is removed
    from the data so the bundled evaluation stays held out."""
    from chronos_translate.corpus.clean import CleanConfig
    from chronos_translate.corpus.evalsets import load_eval_sets
    from chronos_translate.corpus.pipeline import PairSource, build_corpus
    from chronos_translate.corpus.readers import count_rows
    from chronos_translate.corpus.shards import build_shards
    bundle_dir, out_dir = os.path.abspath(bundle_dir), os.path.abspath(out_dir)
    manifest = load_bundle_manifest(bundle_dir)
    if os.path.exists(os.path.join(out_dir, "shards", "shards_manifest.json")):
        raise FileExistsError(f"{out_dir} already holds prepared shards; pick another --out")
    sources, total = [], 0
    for i, path in enumerate(inputs):
        path = os.path.abspath(os.path.expanduser(path))
        kind = KINDS.get(os.path.splitext(path)[1].lower())
        if kind is None:
            raise ValueError(f"{path}: unsupported file type (use {sorted(KINDS)})")
        n = count_rows(path, kind)
        total += n
        clean = CleanConfig(unsegment_zh=unsegment_zh, detokenize_en=detokenize_en)
        sources.append(PairSource(name=f"user{i:02d}", path=path, kind=kind, fields=(zh_field, en_field),
                                  clean=clean, priority=i, rows_per_unit=200_000))
    if valid_per_mille is None:      # ≈1 % on small sets, the recipe's 0.2 % on large ones
        valid_per_mille = 10 if total < 1_000_000 else 2
    eval_pairs = load_eval_sets(os.path.join(bundle_dir, "eval"), ["flores_dev", "flores_devtest"])
    corpus_dir = os.path.join(out_dir, "corpus")
    corpus = build_corpus(sources, eval_pairs, corpus_dir, workers=workers, valid_per_mille=valid_per_mille,
                          min_free_gb=min_free_gb, log=log)
    split_rows = _ensure_valid_split(os.path.join(corpus_dir, "pairs"), min_valid=min(200, max(4, total // 20)),
                                     log=log)
    tok_dir = os.path.join(bundle_dir, manifest["tokenizer"]["dir"])
    shards = build_shards(os.path.join(corpus_dir, "pairs"), tok_dir, os.path.join(out_dir, "shards"),
                          workers=workers, log=log)
    report = {"inputs": inputs, "rows_in": total, "split_rows": split_rows, "valid_per_mille": valid_per_mille,
              "corpus": {k: corpus.get(k) for k in ("rows_after_clean", "duplicates", "eval_leak_removed",
                                                     "final_train", "final_valid", "clean_reasons")},
              "shards": {k: v for k, v in shards.items() if k.endswith(("_tokens", "_records"))}}
    with open(os.path.join(out_dir, "prepare_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log(f"[prepare] {json.dumps(report['shards'])}")
    return report


def _check_shards(shards_dir: str, phases: list[dict]) -> None:
    path = os.path.join(shards_dir, "shards_manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{shards_dir} has no shards_manifest.json; run `chronos bundle prepare-data` first")
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    for phase in phases:
        if phase["name"] == "pretrain":
            need = int(phase.get("seq_len", 512)) + 2
            for split in ("train", "valid"):
                if int(m.get(f"pretrain_{split}_tokens", 0)) < need:
                    raise ValueError(f"pretrain/{split} has {m.get(f'pretrain_{split}_tokens', 0)} tokens, fewer "
                                     f"than one {need}-token window; use --mode sft or add data")
        else:
            for split in ("train", "valid"):
                if int(m.get(f"sft_{split}_records", 0)) < 1:
                    raise ValueError(f"sft/{split} is empty; add data")


def write_finetune_spec(bundle_dir: str, shards_dir: str, run_dir: str, *, tier: str = "quick", mode: str = "sft",
                        budget_hours: float | None = None, seed: int | None = None, cluster_locality: bool = True,
                        lambda_cluster: float | None = None, cluster_window: int | None = None,
                        clusters: int | None = None) -> str:
    """Write ``<run_dir>/finetune_spec.json`` for ``chronos-translate run --spec``.

    The expert-cluster locality constraint is on by default; see
    ``recommend.cluster_locality_config`` for the defaults and their sources."""
    from .recommend import longrun_spec
    bundle_dir = os.path.abspath(bundle_dir)
    with open(os.path.join(bundle_dir, "recipes", "source_run_spec.json"), encoding="utf-8") as fh:
        recipe = json.load(fh)
    spec = longrun_spec(bundle_dir, recipe, run_dir=run_dir, shards_dir=shards_dir, tier=tier, mode=mode,
                        budget_hours=budget_hours, seed=seed, cluster_locality=cluster_locality,
                        lambda_cluster=lambda_cluster, cluster_window=cluster_window, clusters=clusters)
    _check_shards(spec["shards_dir"], spec["phases"])
    os.makedirs(spec["run_dir"], exist_ok=True)
    path = os.path.join(spec["run_dir"], "finetune_spec.json")
    if os.path.exists(os.path.join(spec["run_dir"], "checkpoints", "latest.state.pkl")):
        # the run resumes from latest.state.pkl; a different spec would mix two runs.
        raise FileExistsError(f"{spec['run_dir']} already holds a run; resume it with "
                              f"`chronos-translate run --spec {path}` or choose a new --run-dir")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
    return path


def launch(spec_path: str, budget_hours: float | None = None) -> int:
    from ..run import main as run_main   # TranslationRunSpec: bleu_samples / probe_samples
    argv = ["--spec", spec_path] + (["--budget-hours", str(budget_hours)] if budget_hours is not None else [])
    return run_main(argv)
