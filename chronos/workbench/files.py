"""Server-side file picking and dataset merging for the Web UI (no UI dependency).

The Web UI runs on the user's machine and every path it sends is a server path, so the picker lists
directories through :func:`list_dir` instead of a browser upload. ``kind`` filters what a field
accepts: ``dir`` (folders only), ``data`` (jsonl / json / parquet / csv / tsv / txt), ``checkpoint``
(.pth / .safetensors / .npz / .gguf), ``config`` (.json) or ``any``.

:func:`merge_datasets` streams several dataset files or folders into one JSONL file, the format every
Chronos training loader reads. Records are copied as they are (plain-text lines become
``{"text": ...}``); nothing is loaded whole into memory.

:func:`upload_path` places files a browser on another machine uploads (LAN access) under
``<workdir>/uploads/<datasets|checkpoints|configs|folders|files>/<batch>/``, keeping a folder's
relative layout and refusing traversal, hidden names and executable or script types.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

DATA_SUFFIXES = (".parquet", ".jsonl", ".json", ".csv", ".tsv", ".txt")
KINDS = {
    "dir": (),
    "data": DATA_SUFFIXES,
    "checkpoint": (".pth", ".pt", ".safetensors", ".npz", ".gguf"),
    "config": (".json",),
    "any": None,
}
MAX_ENTRIES = 2000


def _entry(path: str, name: str, is_dir: bool) -> dict:
    try:
        st = os.stat(path)
        size, mtime = (None if is_dir else st.st_size), st.st_mtime
    except OSError:
        size, mtime = None, None
    return {"name": name, "path": path, "dir": is_dir, "size": size, "mtime": mtime}


def list_dir(path: str, kind: str = "any", show_hidden: bool = False) -> dict:
    """Folders and matching files directly under ``path`` (a file path lists its parent).

    Returns ``{"path", "parent", "entries", "truncated", "roots"}``; raises ``FileNotFoundError``
    when nothing usable exists there and ``ValueError`` for an unknown ``kind``.
    """
    if kind not in KINDS:
        raise ValueError("kind")
    path = os.path.abspath(os.path.expanduser(path or "~"))
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            raise FileNotFoundError(path)
        path = parent
    suffixes = KINDS[kind]
    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for e in it:
                if not show_hidden and e.name.startswith("."):
                    continue
                try:
                    is_dir = e.is_dir()
                except OSError:
                    continue
                if is_dir:
                    dirs.append(_entry(e.path, e.name, True))
                elif suffixes is None or (suffixes and e.name.lower().endswith(suffixes)):
                    files.append(_entry(e.path, e.name, False))
    except PermissionError:
        pass
    key = lambda x: x["name"].lower()  # noqa: E731
    entries = sorted(dirs, key=key) + sorted(files, key=key)
    parent = os.path.dirname(path)
    return {"path": path, "parent": parent if parent != path else None,
            "entries": entries[:MAX_ENTRIES], "truncated": len(entries) > MAX_ENTRIES, "roots": _roots()}


def _roots() -> list[dict]:
    """Shortcut places; ``id`` lets the page show the name in the user's language."""
    home = os.path.expanduser("~")
    out = [{"id": "home", "name": "~", "path": home}]
    for name in ("Downloads", "Desktop", "Documents"):
        p = os.path.join(home, name)
        if os.path.isdir(p):
            out.append({"id": name.lower(), "name": name, "path": p})
    if os.path.isdir("/Volumes"):
        out.append({"id": "volumes", "name": "/Volumes", "path": "/Volumes"})
    out.append({"id": "root", "name": "/", "path": os.path.abspath(os.sep)})
    return out


def parse_paths(text) -> list[str]:
    """Paths from a list or from text with one path per line (``~`` expanded, blanks and repeats dropped)."""
    items = text if isinstance(text, (list, tuple)) else str(text or "").splitlines()
    out = []
    for item in items:
        p = os.path.expanduser(str(item).strip().strip('"').strip("'"))
        if p and p not in out:
            out.append(p)
    return out


def data_files(paths: list[str]) -> list[str]:
    """Readable data files behind files, folders (recursive) or globs, in a stable order."""
    from chronos.tokenizer.extract import expand
    out = []
    for p in paths:
        for f in ([p] if os.path.isfile(p) else expand([p])):
            if f.lower().endswith(DATA_SUFFIXES) and f not in out:
                out.append(f)
    return out


def merge_datasets(paths: list[str], out_dir: str, log=None) -> str:
    """One JSONL training file for the given files / folders.

    A single ``.jsonl`` source is used in place. Otherwise every record of every source is streamed
    into ``out_dir/merged_<hash>.jsonl`` (the hash covers paths, sizes and mtimes, so an unchanged
    selection reuses the earlier merge). A ``.sources.json`` manifest records what went in.
    Raises ``FileNotFoundError`` when no readable data file is found.
    """
    from chronos.tokenizer.extract import iter_records
    files = data_files(paths)
    if not files:
        raise FileNotFoundError("no_files")
    if len(files) == 1 and files[0].lower().endswith(".jsonl"):
        return files[0]
    sig = hashlib.sha256()
    for f in files:
        st = os.stat(f)
        sig.update(f"{os.path.abspath(f)}\0{st.st_size}\0{int(st.st_mtime)}\n".encode())
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"merged_{sig.hexdigest()[:12]}.jsonl")
    if os.path.isfile(dst) and os.path.isfile(dst + ".sources.json"):
        if log:
            log(f"[data] reusing merged dataset {dst}")
        return dst
    counts, t0 = [], time.monotonic()
    tmp = dst + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        for f in files:
            n = 0
            for rec in iter_records(f):
                fh.write(json.dumps(rec if isinstance(rec, dict) else {"text": str(rec)}, ensure_ascii=False,
                                    default=str) + "\n")
                n += 1
            counts.append({"path": os.path.abspath(f), "records": n})
            if log:
                log(f"[data] merged {n:,} records from {f}")
    if not sum(c["records"] for c in counts):
        os.remove(tmp)
        raise FileNotFoundError("no_files")
    os.replace(tmp, dst)
    with open(dst + ".sources.json", "w", encoding="utf-8") as fh:
        json.dump({"sources": counts, "records": sum(c["records"] for c in counts),
                   "seconds": round(time.monotonic() - t0, 1)}, fh, indent=1, ensure_ascii=False)
    if log:
        log(f"[data] {sum(c['records'] for c in counts):,} records from {len(files)} files → {dst}")
    return dst


# ── uploads from a browser on another machine ──────────────────────────
UPLOAD_DIRS = {"data": "datasets", "checkpoint": "checkpoints", "config": "configs", "dir": "folders", "any": "files"}
BLOCKED_SUFFIXES = (".py", ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bat", ".cmd", ".com", ".sh", ".command",
                    ".ps1", ".app", ".pkg", ".dmg", ".jar")
_BATCH = re.compile(r"^[A-Za-z0-9_-]{6,48}$")
MAX_DEPTH = 8


def upload_parts(name: str) -> list[str] | None:
    """Safe relative path components of an uploaded file name (``folder/sub/x.jsonl``), else ``None``.

    Rejects absolute paths, ``..``, hidden components, control characters and over-long names.
    """
    if not isinstance(name, str) or not name or len(name) > 1024 or "\x00" in name:
        return None
    parts = [p for p in name.replace("\\", "/").split("/") if p != ""]
    if not parts or len(parts) > MAX_DEPTH or name.startswith("/"):
        return None
    for p in parts:
        if p in (".", "..") or p.startswith(".") or len(p) > 200 or any(ord(c) < 32 for c in p):
            return None
    return parts


def upload_accepts(kind: str, filename: str) -> bool:
    """Whether a field of ``kind`` takes an uploaded file with this name."""
    low = filename.lower()
    if low.endswith(BLOCKED_SUFFIXES):
        return False
    if kind == "data":
        return low.endswith(DATA_SUFFIXES)
    if kind == "checkpoint":                      # the ``.config.json`` sidecar travels with the weights
        return low.endswith(KINDS["checkpoint"] + (".json",))
    if kind == "config":
        return low.endswith(".json")
    return kind in ("dir", "any")


def upload_path(workdir: str, kind: str, batch: str, name: str) -> str:
    """Where an upload lands: ``<workdir>/uploads/<datasets|checkpoints|…>/<batch>/<name>``.

    Raises ``ValueError`` naming the bad part (``kind``, ``batch``, ``name`` or ``type``).
    """
    if kind not in UPLOAD_DIRS:
        raise ValueError("kind")
    if not _BATCH.match(batch or ""):
        raise ValueError("batch")
    parts = upload_parts(name)
    if parts is None:
        raise ValueError("name")
    if not upload_accepts(kind, parts[-1]):
        raise ValueError("type")
    root = os.path.abspath(os.path.join(workdir, "uploads", UPLOAD_DIRS[kind], batch))
    dst = os.path.abspath(os.path.join(root, *parts))
    if os.path.commonpath([root, dst]) != root:
        raise ValueError("name")
    return dst


def upload_root(workdir: str, kind: str, batch: str) -> str:
    return os.path.abspath(os.path.join(workdir, "uploads", UPLOAD_DIRS[kind], batch))
