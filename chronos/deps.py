"""Compatibility shim (0.3.2+): Chronos has no external kernel dependency.

Before 0.3.2 this module git-cloned minimind from an unpinned ``master`` at
import time and inserted it at the front of ``sys.path``.  The kernel pieces
Chronos used now live in :mod:`chronos.model.kernel`; nothing is downloaded
and ``sys.path`` is left alone.  ``import chronos.deps`` keeps working for old
scripts, and :func:`get_tokenizer_path` returns the bundled legacy tokenizer.
"""
from __future__ import annotations

import os
from pathlib import Path

LEGACY_TOKENIZER_DIR = Path(__file__).resolve().parent / "assets" / "tokenizers" / "legacy_6400"


def get_tokenizer_path() -> str:
    """Default tokenizer for checkpoints that never recorded one.

    ``CHRONOS_DEFAULT_TOKENIZER`` overrides it; otherwise the bundled legacy
    6400-token vocabulary (data files only, Apache-2.0, see its NOTICE) is
    used.  New models record their own tokenizer in the checkpoint sidecar.
    """
    override = os.environ.get("CHRONOS_DEFAULT_TOKENIZER", "").strip()
    if override:
        p = Path(override).expanduser()
        if not (p / "tokenizer.json").is_file():
            raise FileNotFoundError(f"CHRONOS_DEFAULT_TOKENIZER={override!r} has no tokenizer.json")
        return str(p)
    return str(LEGACY_TOKENIZER_DIR)


def ensure_minimind():
    """Removed in 0.3.2: Chronos no longer needs minimind.  Returns ``None``."""
    import warnings
    warnings.warn("chronos.deps.ensure_minimind() is a no-op since 0.3.2; Chronos ships its own kernel "
                  "(chronos.model.kernel).", DeprecationWarning, stacklevel=2)
    return None
