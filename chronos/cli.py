"""
chronos/cli.py — Project Chronos unified CLI

Commands:
    chronos train     — pretrain / fine-tune a Chronos model
    chronos eval      — Phase 1 lookahead accuracy validation
    chronos benchmark — Phase 3 end-to-end benchmark
    chronos diagnose  — checkpoint topology + offload/predictive-routing report
    chronos export    — export full checkpoints or expert SSD cluster layout
    chronos mac       — Apple Silicon backend diagnostics
    chronos migrate   — explicitly migrate a v1 checkpoint to v2
    chronos serve     — Ollama/OpenAI-compatible server (native runtime; optional /monitor page)
    chronos tokenizer — train a Chronos tokenizer from selected datasets
    chronos corpus    — download / clean / dedup / shard training corpora
    chronos longrun   — time-budgeted resumable MLX training run (alias: run)
    chronos monitor   — read-only training monitor page + JSON API
    chronos lazy-probe — lazy-expert efficiency / hit rate / fidelity of a checkpoint or a run
    chronos webui     — Web UI (alias: ui)
"""
import sys
import os

import argparse
import json

if __package__ in {None, ""}:
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_pkg_dir)
    # Running `python chronos/cli.py` puts `chronos/` itself on sys.path.
    # That makes the local `chronos/mlx` package shadow Apple's top-level
    # `mlx` package. Keep only the repo root.
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _pkg_dir]
    sys.path.insert(0, _repo_root)


def cmd_train(args):
    import train_chronos
    sys.argv = ['train_chronos'] + args
    train_chronos.main()


def cmd_eval(args):
    from chronos.eval.io_profiler import main
    sys.argv = ['io_profiler'] + args
    main()


def cmd_benchmark(args):
    from chronos.eval.benchmark import main
    sys.argv = ['benchmark'] + args
    main()


def cmd_diagnose(args):
    from diagnose_checkpoint import main
    sys.argv = ['diagnose_checkpoint'] + args
    main()


def cmd_export(args):
    # Backward compatibility: the original `chronos export` command generated
    # an expert cluster layout and required --data_path. New full-checkpoint
    # deployment export does not.
    if "--data_path" in args:
        from chronos.io.cluster_layout import main
        sys.argv = ['cluster_layout'] + args
        main()
    else:
        from chronos.export import main
        main(args)


def cmd_mac(args):
    from chronos.backend.mac_diagnostics import mac_backend_diagnostics

    configure = "--no-configure-threads" not in args
    print(json.dumps(mac_backend_diagnostics(configure_threads=configure), indent=2, ensure_ascii=False))


def cmd_migrate(args):
    from chronos.migrate import main
    main(args)


def _module_main(module: str, rest):
    import importlib
    raise SystemExit(importlib.import_module(module).main(rest))


def _cmd_corpus(rest):
    """chronos corpus {download|build|shards} ..."""
    if not rest or rest[0] not in {"download", "build", "shards"}:
        raise SystemExit("usage: chronos corpus {download|build|shards} [options]")
    module = {"download": "chronos.data.corpus.sources", "build": "chronos.data.corpus.presets",
              "shards": "chronos.data.corpus.shards"}[rest[0]]
    _module_main(module, rest[1:])


def main():
    parser = argparse.ArgumentParser(
        prog='chronos',
        description='Project Chronos — On-device low-latency lookahead dual-layer MoE inference',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('train',     help='Train a Chronos model',           add_help=False)
    sub.add_parser('eval',      help='Phase 1 lookahead accuracy eval',  add_help=False)
    sub.add_parser('benchmark', help='Phase 3 end-to-end benchmark',     add_help=False)
    sub.add_parser('diagnose',  help='Checkpoint/offload diagnostic',    add_help=False)
    sub.add_parser('export',    help='Export checkpoint or expert cluster layout', add_help=False)
    sub.add_parser('mac',       help='Apple Silicon backend diagnostics', add_help=False)
    sub.add_parser('migrate',   help='Explicitly migrate a v1 checkpoint to v2', add_help=False)
    sub.add_parser('serve',     help='Ollama/OpenAI-compatible server (native Chronos runtime)', add_help=False)
    sub.add_parser('tokenizer', help='Train a Chronos tokenizer from selected datasets', add_help=False)
    sub.add_parser('corpus',    help='Download/clean/dedup/shard training corpora', add_help=False)
    sub.add_parser('longrun',   help='Time-budgeted resumable MLX training run', aliases=['run'], add_help=False)
    sub.add_parser('monitor',   help='Read-only training monitor page + JSON API', add_help=False)
    sub.add_parser('bundle',    help='Out-of-the-box model bundle: build/check/translate/finetune/serve', add_help=False)
    sub.add_parser('lazy-probe', help='Lazy-expert efficiency/hit-rate/fidelity probe (checkpoint or --watch run)',
                   add_help=False)
    sub.add_parser('webui',     help='Web UI (FastAPI; --legacy for the Gradio console)', aliases=['ui'],
                   add_help=False)

    known, rest = parser.parse_known_args()
    longrun = lambda rest: _module_main('chronos.mlx.training.longrun', rest)  # noqa: E731
    dispatch = {
        'train':     cmd_train,
        'eval':      cmd_eval,
        'benchmark': cmd_benchmark,
        'diagnose':  cmd_diagnose,
        'export':    cmd_export,
        'mac':       cmd_mac,
        'migrate':   cmd_migrate,
        'serve':     lambda rest: _module_main('chronos.serving.ollama_server', rest),
        'tokenizer': lambda rest: _module_main('chronos.tokenizer', rest),
        'corpus':    _cmd_corpus,
        'longrun':   longrun,
        'run':       longrun,
        'monitor':   lambda rest: _module_main('chronos.monitor.server', rest),
        'bundle':    lambda rest: _module_main('chronos.bundle.cli', rest),
        'lazy-probe': lambda rest: _module_main('chronos.workbench.lazy_probe', rest),
        'webui':     lambda rest: _module_main('chronos.webui.server', rest),
        'ui':        lambda rest: _module_main('chronos.webui.server', rest),
    }
    dispatch[known.command](rest)


if __name__ == '__main__':
    main()
