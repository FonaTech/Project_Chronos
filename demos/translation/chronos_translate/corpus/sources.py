"""Source registry for corpus presets.

A preset names the remote sources it needs; each source resolves to a pinned
``SourceSpec``.  Local, user-provided datasets are described by
``LocalSource`` and never re-downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from .download import SourceSpec, download_source, hf_dataset_files, modelscope_dataset_files


@dataclass
class LocalSource:
    name: str
    path: str
    kind: str            # "pairs_csv" | "pairs_parquet" | "pairs_jsonl" | "text_jsonl"
    role: str = "train"  # "train" | "eval"
    weight: float = 1.0
    fields: tuple[str, str] = ("zh", "en")


def _hf(name: str, repo: str, subdir: str, dest: str, revision: str | None = None,
        license: str | None = None, note: str = "") -> SourceSpec:
    sha, files = hf_dataset_files(repo, subdir, revision)
    return SourceSpec(name=name, dest_dir=dest, files=files, revision=sha, license=license, note=note)


def _ms(name: str, repo: str, root: str, dest: str, match, license: str | None = None,
        note: str = "") -> SourceSpec:
    files = modelscope_dataset_files(repo, root, match)
    revs = sorted({f.url.split("/resolve/")[1].split("/")[0] for f in files})
    return SourceSpec(name=name, dest_dir=dest, files=files, revision=",".join(revs), license=license, note=note)


def translation_enzh_remote(raw_root: str) -> dict:
    """Remote sources for the en→zh validation demo (lazy: resolved on demand)."""
    return {
        "wmt19_zh_en": lambda: _hf(
            "wmt19_zh_en", "wmt/wmt19", "zh-en", os.path.join(raw_root, "wmt__wmt19__zh-en"),
            license="unknown (dataset card); research use only",
            note="news-commentary, UN, CWMT, wikititles mixture"),
        "flores_plus_enzh": lambda: _ms(
            "flores_plus_enzh", "facebook/flores", "data/pair",
            os.path.join(raw_root, "facebook__flores__eng-zho"),
            match=lambda p: p.startswith("data/pair/eng_Latn-zho_Hans/"),
            license="CC BY-SA 4.0", note="evaluation only: dev + devtest"),
        "wmt22_test_enzh": lambda: _hf(
            "wmt22_test_enzh", "haoranxu/WMT22-Test", "en-zh",
            os.path.join(raw_root, "haoranxu__WMT22-Test__en-zh"),
            license="unknown (dataset card); evaluation only"),
        "wmt24pp_enzh": lambda: _ms(
            "wmt24pp_enzh", "google/wmt24pp", "", os.path.join(raw_root, "google__wmt24pp__en-zh_CN"),
            match=lambda p: p in {"en-zh_CN.jsonl", "README.md"},
            license="see README", note="evaluation only"),
    }


def translation_enzh_local(data_root: str) -> list[LocalSource]:
    raw = os.path.join(data_root, "raw")
    return [
        LocalSource("iic_wmt_zh_en", os.path.join(
            raw, "iic__WMT-Chinese-to-English-Machine-Translation-Training-Corpus",
            "wmt_zh_en_training_corpus.csv"), "pairs_csv", fields=("0", "1")),
        LocalSource("opus100_en_zh", os.path.join(raw, "Helsinki-NLP__opus-100", "en-zh",
                                                  "train-00000-of-00001.parquet"), "pairs_parquet",
                    fields=("translation.zh", "translation.en")),
        LocalSource("opus100_en_zh_test", os.path.join(raw, "Helsinki-NLP__opus-100", "en-zh",
                                                       "test-00000-of-00001.parquet"), "pairs_parquet",
                    role="eval", fields=("translation.zh", "translation.en")),
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download pinned corpus sources")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    remote = translation_enzh_remote(args.raw_root)
    names = args.only or list(remote)
    for name in names:
        spec = remote[name]()
        total = sum(f.size or 0 for f in spec.files)
        print(f"[{name}] rev={spec.revision} files={len(spec.files)} bytes={total}", flush=True)
        if args.list:
            continue
        last = {}

        def progress(path, done, size, _last=last):
            step = done // (256 << 20)
            if _last.get(path) != step:
                _last[path] = step
                print(f"  {path}: {done/1e6:.0f}/{(size or 0)/1e6:.0f} MB", flush=True)

        manifest = download_source(spec, progress=progress)
        print(json.dumps({"name": name, "status": [f["status"] for f in manifest["files"]]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
