"""``chronos-translate``: the en↔zh translation demo's command line.

    chronos-translate corpus download|build|shards|docmix …   data pipeline
    chronos-translate run --spec recipes/enzh_50m_72h.json     time-budgeted MLX run
    chronos-translate doc-run --spec <doc-mix recipe>          doc-mix variant
    chronos-translate inspect --run-dir runs/…                 offline FLORES+ probe
    chronos-translate bundle build|check|translate|…           model bundles

Kernel commands (``chronos serve``, ``chronos monitor``, ``chronos tokenizer``,
…) stay in the ``chronos`` CLI.
"""
from __future__ import annotations

import importlib
import sys

CORPUS = {
    "download": "chronos_translate.corpus.sources",
    "build": "chronos_translate.corpus.presets",
    "shards": "chronos_translate.corpus.shards",
    "docmix": "chronos_translate.corpus.docmix",
}
COMMANDS = {
    "run": "chronos_translate.run",
    "doc-run": "chronos_translate.doc_run",
    "inspect": "chronos_translate.probe",
    "bundle": "chronos_translate.bundle.cli",
}


def _usage() -> str:
    return (__doc__ or "").strip() + "\n"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0 if argv else 2
    if argv[0] in {"-V", "--version"}:
        from . import __version__
        print(__version__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "corpus":
        if not rest or rest[0] not in CORPUS:
            print("usage: chronos-translate corpus {" + "|".join(CORPUS) + "} [options]", file=sys.stderr)
            return 2
        module, rest = CORPUS[rest[0]], rest[1:]
    elif cmd in COMMANDS:
        module = COMMANDS[cmd]
    else:
        print(f"unknown command {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    return int(importlib.import_module(module).main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
