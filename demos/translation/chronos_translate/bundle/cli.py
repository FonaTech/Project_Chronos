"""``chronos-translate bundle {build|pack|check|probe|translate|eval|prepare-data|finetune|serve}``."""
from __future__ import annotations

import argparse
import json
import os
import sys


def _bundle_dir(value: str | None) -> str:
    """Default: the bundle this file was installed from, else the current directory."""
    return os.path.abspath(value or os.environ.get("CHRONOS_BUNDLE") or ".")


def _add_lazy_args(parser) -> None:
    """SSD-lazy expert options (MLX backend); defaults are the engine defaults."""
    from chronos.lazy_experts import (
        DEFAULT_DECODE_WAIT_S, DEFAULT_RAM_RATIO, DEFAULT_RESIDENCY, DEFAULT_RESIDENCY_DECAY,
        DEFAULT_VERIFY_WAIT_S, RESIDENCY_MODES,
    )
    g = parser.add_argument_group("lazy experts (MLX)")
    g.add_argument("--lazy-experts", action="store_true",
                   help="keep only the trunk in memory and page routed experts from an SSD cache")
    g.add_argument("--paging", default="auto", choices=["auto", "cluster", "expert"],
                   help="page layout; auto = by training cluster if the model was trained with cluster "
                        "locality, else per expert (default: auto)")
    g.add_argument("--residency", default=DEFAULT_RESIDENCY, choices=list(RESIDENCY_MODES),
                   help=f"planned = next-use eviction, window = LRU + protected window (default: {DEFAULT_RESIDENCY})")
    g.add_argument("--decay", type=float, default=DEFAULT_RESIDENCY_DECAY,
                   help=f"planned residency: weight of older forwards, [0, 1) (default: {DEFAULT_RESIDENCY_DECAY})")
    g.add_argument("--ram-ratio", type=float, default=DEFAULT_RAM_RATIO,
                   help=f"expert pages kept loaded, × the routed working set (default: {DEFAULT_RAM_RATIO})")
    g.add_argument("--decode-wait", type=float, default=DEFAULT_DECODE_WAIT_S,
                   help=f"seconds a decode step waits for prefetched pages (default: {DEFAULT_DECODE_WAIT_S})")
    g.add_argument("--verify-wait", type=float, default=DEFAULT_VERIFY_WAIT_S,
                   help=f"seconds a speculative verify waits for prefetched pages (default: {DEFAULT_VERIFY_WAIT_S})")


def _lazy_options(a):
    from chronos.lazy_experts import LazyOptions
    return LazyOptions(paging=a.paging, residency=a.residency, residency_decay=a.decay, ram_ratio=a.ram_ratio,
                       decode_wait_s=a.decode_wait, verify_wait_s=a.verify_wait).validated()


def _bundle_model(bundle: str, a):
    from .runtime import BundleModel
    backend = "mlx" if a.lazy_experts else a.backend
    return BundleModel(bundle, device=a.device, dtype=a.dtype, backend=backend, lazy_experts=a.lazy_experts,
                       lazy_options=_lazy_options(a) if a.lazy_experts else None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="chronos-translate bundle", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="assemble a bundle from a run directory (read-only on the run)")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--out", default=None)
    b.add_argument("--phase", default="pretrain", choices=["pretrain", "sft"])
    b.add_argument("--which", default="best", choices=["best", "final"])
    b.add_argument("--name", default=None)
    b.add_argument("--status", default="interim", choices=["interim", "final"])
    b.add_argument("--wheel", action="append", default=None,
                   help="a wheel to ship under wheels/ (repeat: the project_chronos and chronos_translate wheels)")
    b.add_argument("--probe-n", type=int, default=50, help="FLORES+ dev sentences for the quality probe (0 = skip)")
    b.add_argument("--pack", action="store_true", help="also write <out>.tar.gz + .sha256")

    k = sub.add_parser("pack", help="tar.gz a bundle directory")
    k.add_argument("bundle")
    k.add_argument("--archive", default=None)

    c = sub.add_parser("check", help="verify file hashes and run a CPU smoke translation")
    c.add_argument("bundle", nargs="?")
    c.add_argument("--no-smoke", action="store_true")
    c.add_argument("--device", default="cpu")

    q = sub.add_parser("probe", help="measure BLEU/chrF on FLORES+ dev and store it in bundle.json")
    q.add_argument("bundle", nargs="?")
    q.add_argument("-n", type=int, default=50)
    q.add_argument("--device", default="cpu")

    t = sub.add_parser("translate", help="translate text (argument, --file or stdin)")
    t.add_argument("text", nargs="*")
    t.add_argument("--bundle", default=None)
    t.add_argument("--file", default=None)
    t.add_argument("--direction", default="en_zh", choices=["en_zh", "zh_en"])
    t.add_argument("--repetition-penalty", type=float, default=None,
                   help="default 1.2; 1.0 with --speculative (required for exact speculative greedy)")
    t.add_argument("--max-new-tokens", type=int, default=256)
    t.add_argument("--backend", default="torch", choices=["torch", "mlx"])
    t.add_argument("--device", default="auto", help="torch: auto|cpu|mps|cuda; mlx: auto|gpu|cpu")
    t.add_argument("--dtype", default="float32")
    t.add_argument("--speculative", action="store_true",
                   help="recurrent-depth self-speculative decoding (same greedy output, fewer full-depth forwards)")
    t.add_argument("--draft-depth", type=int, default=1)
    t.add_argument("--gamma", type=int, default=4, help="initial draft length (adapts between 1 and 8)")
    _add_lazy_args(t)

    e = sub.add_parser("eval", help="BLEU/chrF on the bundled FLORES+ split")
    e.add_argument("--bundle", default=None)
    e.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    e.add_argument("-n", type=int, default=0, help="first n sentences (0 = all)")
    e.add_argument("--repetition-penalty", type=float, default=1.2)
    e.add_argument("--backend", default="torch", choices=["torch", "mlx"])
    e.add_argument("--device", default="auto")
    e.add_argument("--dtype", default="float32")
    e.add_argument("--out", default=None, help="write the report JSON here")
    _add_lazy_args(e)

    d = sub.add_parser("prepare-data", help="your parallel data → shards tokenized with the bundle tokenizer")
    d.add_argument("inputs", nargs="+", help=".jsonl/.csv/.tsv/.parquet files with zh and en columns")
    d.add_argument("--bundle", default=None)
    d.add_argument("--out", required=True)
    d.add_argument("--zh-field", default="zh")
    d.add_argument("--en-field", default="en")
    d.add_argument("--unsegment-zh", action="store_true", help="remove word-segmentation spaces between CJK chars")
    d.add_argument("--detokenize-en", action="store_true", help="undo Moses tokenization on the English side")
    d.add_argument("--workers", type=int, default=4)

    f = sub.add_parser("finetune", help="write a run spec that starts from the bundle weights (MLX)")
    f.add_argument("--bundle", default=None)
    f.add_argument("--data", required=True, help="the --out of prepare-data (or a shards dir)")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--tier", default="quick", choices=["smoke", "quick", "standard", "full"])
    f.add_argument("--mode", default="sft", choices=["sft", "pretrain", "pretrain+sft"])
    f.add_argument("--hours", type=float, default=None, help="override the tier's wall-clock budget")
    f.add_argument("--seed", type=int, default=None)
    f.add_argument("--dry-run", action="store_true", help="write the spec, do not start training")
    from chronos.expert_clusters import DEFAULT_LAMBDA
    f.add_argument("--cluster-locality", action=argparse.BooleanOptionalAction, default=True,
                   help="train with the expert-cluster locality constraint so lazy inference can page "
                        "by cluster (default: on)")
    f.add_argument("--lambda-cluster", type=float, default=None,
                   help=f"constraint strength (default: {DEFAULT_LAMBDA}, from a measured sweep)")
    f.add_argument("--cluster-window", type=int, default=None,
                   help="tokens per locality window (default: nearest power of two to the median SFT "
                        "record length of --data, clamped to [16, seq_len])")
    f.add_argument("--clusters", type=int, default=None, help="number of clusters (default: ceil(E / 2k))")

    s = sub.add_parser("serve", help="Ollama/OpenAI-protocol server over the bundle's model/ (loopback only)")
    s.add_argument("--bundle", default=None)
    s.add_argument("rest", nargs=argparse.REMAINDER, help="extra `chronos serve` options")

    a = p.parse_args(argv)
    return _run(a)


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _run(a) -> int:
    from . import builder
    if a.cmd == "build":
        out = builder.build_bundle(a.run_dir, a.out, phase=a.phase, which=a.which, name=a.name, wheel=a.wheel,
                                   status=a.status)
        if a.probe_n:
            _print(builder.record_quality_probe(out, a.probe_n))
        report = builder.check_bundle(out)
        _print(report)
        if a.pack:
            builder.pack_bundle(out)
        return 0 if report["ok"] else 1
    if a.cmd == "pack":
        builder.pack_bundle(a.bundle, a.archive)
        return 0
    if a.cmd == "check":
        report = builder.check_bundle(_bundle_dir(a.bundle), smoke=not a.no_smoke, device=a.device)
        _print(report)
        return 0 if report["ok"] else 1
    if a.cmd == "probe":
        _print(builder.record_quality_probe(_bundle_dir(a.bundle), a.n, device=a.device))
        return 0
    if a.cmd == "translate":
        lines = list(a.text) and [" ".join(a.text)]
        if a.file:
            with open(a.file, encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
        elif not lines:
            lines = [ln.strip() for ln in sys.stdin if ln.strip()]
        rp = a.repetition_penalty if a.repetition_penalty is not None else (1.0 if a.speculative else 1.2)
        bm = _bundle_model(_bundle_dir(a.bundle), a)
        try:
            for hyp in bm.translate(lines, a.direction, max_new_tokens=a.max_new_tokens, repetition_penalty=rp,
                                    speculative=a.speculative, draft_depth=a.draft_depth, gamma=a.gamma):
                print(hyp, flush=True)
        finally:
            bm.close()
        if bm.last_stats.get("lazy"):
            print(json.dumps({"lazy": bm.last_stats["lazy"]}, ensure_ascii=False), file=sys.stderr)
        if bm.last_stats.get("speculative"):
            print(f"[speculative] {json.dumps(bm.last_stats['speculative'])}", file=sys.stderr)
        return 0
    if a.cmd == "eval":
        from .runtime import eval_pairs, score
        bundle = _bundle_dir(a.bundle)
        pairs = eval_pairs(bundle, a.split)
        pairs = pairs[: a.n] if a.n else pairs
        bm = _bundle_model(bundle, a)
        try:
            report = score(bm, pairs, repetition_penalty=a.repetition_penalty)
        finally:
            bm.close()
        if bm.last_stats.get("lazy"):
            report["lazy"] = bm.last_stats["lazy"]
        report["set"] = f"flores_plus_{a.split}"
        _print(report)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
        return 0
    if a.cmd == "prepare-data":
        from .finetune import prepare_data
        prepare_data(_bundle_dir(a.bundle), a.inputs, a.out, zh_field=a.zh_field, en_field=a.en_field,
                     unsegment_zh=a.unsegment_zh, detokenize_en=a.detokenize_en, workers=a.workers)
        return 0
    if a.cmd == "finetune":
        from .finetune import launch, write_finetune_spec
        data = os.path.abspath(a.data)
        shards = os.path.join(data, "shards") if os.path.isdir(os.path.join(data, "shards")) else data
        spec = write_finetune_spec(_bundle_dir(a.bundle), shards, a.run_dir, tier=a.tier, mode=a.mode,
                                   budget_hours=a.hours, seed=a.seed, cluster_locality=a.cluster_locality,
                                   lambda_cluster=a.lambda_cluster, cluster_window=a.cluster_window,
                                   clusters=a.clusters)
        print(f"[finetune] spec: {spec}")
        if a.dry_run:
            print(f"[finetune] start it with: chronos-translate run --spec {spec}")
            return 0
        return launch(spec)
    if a.cmd == "serve":
        from chronos.serving.ollama_server import main as serve_main
        from .builder import load_bundle_manifest
        bundle = _bundle_dir(a.bundle)
        load_bundle_manifest(bundle)
        rest = [r for r in a.rest if r != "--"]
        return serve_main(["--models-dir", os.path.join(bundle, "model"), *rest])
    raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
