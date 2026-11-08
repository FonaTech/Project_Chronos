"""Opt-in loader for an external minimind checkout (comparison tools only).

Chronos itself does not use minimind since 0.3.2.  The historical
``tools/compare_minimind_chronos*.py`` harnesses still compare against it, so
they need a checkout the user provides: ``--minimind-dir`` or
``CHRONOS_MINIMIND_DIR``.  Nothing is cloned or downloaded here.
"""
from __future__ import annotations

import os
import sys


def minimind_dir_from_argv(argv: list[str] | None = None) -> str | None:
    """Pop ``--minimind-dir PATH`` / ``--minimind-dir=PATH`` from ``argv``."""
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(list(argv)):
        if arg == "--minimind-dir" and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if arg.startswith("--minimind-dir="):
            del argv[i]
            return arg.split("=", 1)[1]
    return None


def require_minimind(path: str | None = None) -> str:
    """Put a user-provided minimind checkout on ``sys.path`` and return it."""
    path = path or minimind_dir_from_argv() or os.environ.get("CHRONOS_MINIMIND_DIR", "").strip()
    if not path:
        raise SystemExit(
            "This comparison needs an external minimind checkout as the baseline.\n"
            "Chronos no longer bundles or downloads it (since 0.3.2). Clone it yourself, e.g.\n"
            "  git clone https://github.com/jingyaogong/minimind /path/to/minimind\n"
            "and pass --minimind-dir /path/to/minimind (or set CHRONOS_MINIMIND_DIR).\n"
            "For a baseline with no external code, run benchmark_compare.py.")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(os.path.join(path, "model", "model_minimind.py")):
        raise SystemExit(f"{path} is not a minimind checkout (model/model_minimind.py is missing).")
    if path not in sys.path:
        sys.path.append(path)       # after Chronos, so it never shadows chronos.* modules
    return path
