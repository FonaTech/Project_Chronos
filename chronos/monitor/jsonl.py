"""Incremental JSON-lines reader.

A training run appends to its JSONL files for days; re-parsing a growing file
on every refresh would cost O(file) per poll.  ``JsonLines`` remembers the
(inode, byte offset) it reached and parses only the new bytes.  A trailing
line without ``\\n`` is an in-flight write and is left for the next poll; a
replaced or truncated file (new inode / smaller size) is re-read from zero.
"""
from __future__ import annotations

import json
import os
import threading


class JsonLines:
    def __init__(self, path: str, max_rows: int | None = None):
        self.path = path
        self.max_rows = max_rows          # keep only the newest rows (None = all)
        self.rows: list[dict] = []
        self.bad_lines = 0
        self._key = None                  # (st_dev, st_ino)
        self._offset = 0
        self._lock = threading.Lock()

    def read(self) -> list[dict]:
        with self._lock:
            try:
                st = os.stat(self.path)
            except OSError:
                self.rows, self._key, self._offset = [], None, 0
                return self.rows
            key = (st.st_dev, st.st_ino)
            if key != self._key or st.st_size < self._offset:
                self.rows, self._key, self._offset, self.bad_lines = [], key, 0, 0
            if st.st_size > self._offset:
                with open(self.path, "rb") as fh:
                    fh.seek(self._offset)
                    chunk = fh.read(st.st_size - self._offset)
                end = chunk.rfind(b"\n")
                if end >= 0:
                    self._offset += end + 1
                    for line in chunk[:end].splitlines():
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError:
                            self.bad_lines += 1
                            continue
                        if isinstance(row, dict):
                            self.rows.append(row)
                    if self.max_rows is not None and len(self.rows) > self.max_rows:
                        del self.rows[: len(self.rows) - self.max_rows]
            return self.rows


def read_json(path: str):
    """Whole-file JSON (small files: specs, status, inspections); None if absent/partial."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def tail_text(path: str, max_bytes: int = 16000) -> str:
    """Last ``max_bytes`` of a text log, starting at a line boundary."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                data = fh.read()
                cut = data.find(b"\n")
                data = data[cut + 1:] if cut >= 0 else data
            else:
                data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")
