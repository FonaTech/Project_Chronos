"""Verified, resumable dataset downloads.

Every source is pinned to a revision, written into its own directory and
checked against the size and SHA-256 that the hosting API reports.  Results
are recorded in ``download_manifest.json`` next to the files so later stages
can prove which bytes they consumed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field

CHUNK = 1 << 20


@dataclass
class RemoteFile:
    url: str
    rel_path: str
    size: int | None = None
    sha256: str | None = None


@dataclass
class SourceSpec:
    name: str
    dest_dir: str
    files: list[RemoteFile] = field(default_factory=list)
    revision: str | None = None
    license: str | None = None
    note: str = ""


def _get_json(url: str, timeout: float = 60.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "chronos-corpus/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hf_dataset_files(repo: str, subdir: str, revision: str | None = None) -> tuple[str, list[RemoteFile]]:
    """List files under ``subdir`` of a Hugging Face dataset, pinned to a commit."""
    meta = _get_json(f"https://huggingface.co/api/datasets/{repo}" + (f"/revision/{revision}" if revision else ""))
    sha = meta["sha"]
    tree = _get_json(f"https://huggingface.co/api/datasets/{repo}/tree/{sha}/{urllib.parse.quote(subdir)}")
    files = []
    for item in tree:
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        files.append(RemoteFile(
            url=f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{urllib.parse.quote(item['path'])}",
            rel_path=item["path"],
            size=int(item.get("size") or 0) or None,
            sha256=lfs.get("oid"),
        ))
    return sha, files


def modelscope_dataset_files(repo: str, root: str, match=None) -> list[RemoteFile]:
    """List files from a ModelScope dataset tree; ``match(path)`` filters them."""
    query = urllib.parse.urlencode({"Revision": "master", "Root": root, "Recursive": "true", "PageSize": 5000})
    data = _get_json(f"https://modelscope.cn/api/v1/datasets/{repo}/repo/tree?{query}")
    files = []
    for item in (data.get("Data") or {}).get("Files") or []:
        if item.get("Type") != "blob" or (match and not match(item["Path"])):
            continue
        rev = item.get("Revision") or "master"
        files.append(RemoteFile(
            url=f"https://modelscope.cn/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(item['Path'])}",
            rel_path=item["Path"],
            size=int(item.get("Size") or 0) or None,
            sha256=item.get("Sha256") or None,
        ))
    return files


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def free_bytes(path: str) -> int:
    probe = path
    while not os.path.exists(probe):
        probe = os.path.dirname(probe) or "."
    return shutil.disk_usage(probe).free


def download_file(remote: RemoteFile, dest_dir: str, *, min_free_bytes: int = 8 << 30,
                  retries: int = 30, progress=None) -> dict:
    """Download one file with HTTP range resume, then verify size and SHA-256."""
    target = os.path.join(dest_dir, remote.rel_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    record = asdict(remote) | {"local": target}
    if os.path.isfile(target) and remote.sha256 and os.path.getsize(target) == (remote.size or -1):
        if sha256_file(target) == remote.sha256:
            return record | {"status": "verified_existing"}
    needed = (remote.size or 0) + min_free_bytes
    if free_bytes(dest_dir) < needed:
        raise OSError(f"insufficient disk for {remote.rel_path}: need {needed} bytes free")
    partial = target + ".part"
    for attempt in range(retries):
        offset = os.path.getsize(partial) if os.path.exists(partial) else 0
        if remote.size and offset >= remote.size:
            break
        headers = {"User-Agent": "chronos-corpus/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            req = urllib.request.Request(remote.url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                # A server that ignores Range restarts from byte zero.
                mode = "ab" if offset and resp.status == 206 else "wb"
                done = offset if mode == "ab" else 0
                with open(partial, mode) as out:
                    for block in iter(lambda: resp.read(CHUNK), b""):
                        out.write(block)
                        done += len(block)
                        if progress:
                            progress(remote.rel_path, done, remote.size)
        except (OSError, urllib.error.URLError):  # network hiccup: resume below
            pass
        # CDNs may close a stream early without raising; resume until the
        # advertised size is reached (unknown size: accept one clean pass).
        current = os.path.getsize(partial) if os.path.exists(partial) else 0
        if (remote.size and current >= remote.size) or (not remote.size and current > 0):
            break
        if attempt == retries - 1:
            raise IOError(f"download incomplete for {remote.rel_path}: {current} of {remote.size} bytes")
        time.sleep(min(2 ** attempt, 30))
    size = os.path.getsize(partial)
    if remote.size and size != remote.size:
        raise IOError(f"size mismatch for {remote.rel_path}: {size} != {remote.size}")
    digest = sha256_file(partial)
    if remote.sha256 and digest != remote.sha256:
        os.remove(partial)
        raise IOError(f"sha256 mismatch for {remote.rel_path}")
    os.replace(partial, target)
    return record | {"bytes": size, "sha256": digest,
                     "status": "downloaded_verified" if remote.sha256 else "downloaded_unverified_hash",
                     "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def download_source(spec: SourceSpec, *, progress=None) -> dict:
    os.makedirs(spec.dest_dir, exist_ok=True)
    records = [download_file(f, spec.dest_dir, progress=progress) for f in spec.files]
    manifest = {"name": spec.name, "revision": spec.revision, "license": spec.license,
                "note": spec.note, "files": records}
    with open(os.path.join(spec.dest_dir, "download_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
