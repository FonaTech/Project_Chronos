"""Build and verify a self-contained Chronos model bundle (layout in ``__init__``)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time

BUNDLE_FORMAT = "chronos-bundle/1"
MANIFEST = "bundle.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chronos_tokenizer.json", "tokenizer_report.json")
FLORES_REL = os.path.join("facebook__flores__eng-zho", "data", "pair", "eng_Latn-zho_Hans")
FLORES_FILES = ("dev-00000-of-00001.parquet", "devtest-00000-of-00001.parquet")
# Never shipped: they hold absolute local paths, optimizer state or caches.
_SKIP_NAMES = {"__pycache__", ".DS_Store"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bundle_manifest(bundle_dir: str) -> dict:
    path = os.path.join(bundle_dir, MANIFEST)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{bundle_dir!r} is not a Chronos bundle (no {MANIFEST})")
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format {manifest.get('format')!r}; expected {BUNDLE_FORMAT}")
    return manifest


def _copy(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copytree(src: str, dst: str) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*_SKIP_NAMES, "*.pyc"), dirs_exist_ok=True)


def _file_table(root: str) -> dict:
    """``relative path → {bytes, sha256}`` for every file except the manifest."""
    table = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_NAMES)
        for name in sorted(files):
            if name in _SKIP_NAMES:
                continue
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel == MANIFEST:
                continue
            table[rel] = {"bytes": os.path.getsize(full), "sha256": sha256_file(full)}
    return table


def _sanitize(value, roots: list[str]):
    """Replace local absolute prefixes (repo, home) so a bundle leaks no paths."""
    if isinstance(value, str):
        for root, label in roots:
            if value.startswith(root):
                return label + value[len(root):]
        return value
    if isinstance(value, list):
        return [_sanitize(v, roots) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize(v, roots) for k, v in value.items()}
    return value


def _path_roots(run_dir: str) -> list[tuple[str, str]]:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    return [(repo, "$CHRONOS_REPO"), (os.path.expanduser("~"), "~")]


def _metrics(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                       # a line being written right now
    return out


def run_provenance(run_dir: str, phase: str, which: str) -> dict:
    """Training state behind ``<prefix>_<phase>[.best].pth`` from ``metrics.jsonl``.

    ``best``: the validation with the lowest valid CE (the run exports
    ``.best`` right after it); ``final``: the last validation."""
    records = _metrics(run_dir)
    vals = [r for r in records if r.get("event") == "validation" and r.get("phase") == phase]
    if not vals:
        return {"phase": phase, "note": "no validation record found"}
    val = min(vals, key=lambda r: r["valid_ce"]) if which == "best" else vals[-1]
    steps = [r for r in records if r.get("phase") == phase and "tokens_seen" in r and r.get("step", 0) <= val["step"]]
    last = steps[-1] if steps else {}
    return {"phase": phase, "step": val["step"], "planned_steps_at_export": last.get("total"),
            "time": val.get("time"), "valid_ce": round(float(val["valid_ce"]), 4),
            "valid_ppl": round(float(val.get("valid_ppl", 0.0)), 3), "tokens_seen": last.get("tokens_seen"),
            "label": "measured (held-out shard validation, 8 fixed batches)",
            "validation_history": [{"step": r["step"], "valid_ce": round(float(r["valid_ce"]), 4)} for r in vals]}


def _stable_copy(src: str, dst: str, retries: int = 3) -> str:
    """Copy ``src`` and prove it did not change mid-copy (a live run may re-export it)."""
    for _ in range(retries):
        before = os.stat(src)
        _copy(src, dst)
        after = os.stat(src)
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return sha256_file(dst)
        time.sleep(2.0)
    raise RuntimeError(f"{src} kept changing while being copied")


def _planner_preset(vocab_size: int, hidden_size: int) -> dict | None:
    """The v3 preset with the same topology, if one exists (``ui/presets_v3.json``)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "ui", "presets_v3.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        presets = json.load(fh).get("presets", {})
    for preset in presets.values():
        if preset.get("vocab_size") == vocab_size and preset.get("hidden_size") == hidden_size:
            return preset
    return None


def build_bundle(run_dir: str, out_dir: str | None = None, *, phase: str = "pretrain", which: str = "best",
                 name: str | None = None, minimind_dir: str | None = None, wheel=None,
                 include_sft_valid: bool = True, status: str = "interim", log=print) -> str:
    """Assemble a bundle from a run directory (read-only on the run).

    ``wheel``: one path or a list of paths (the kernel and the
    ``chronos_translate`` wheels) copied to ``wheels/`` for ``bin/setup.sh``.
    ``minimind_dir`` is ignored since 0.3.2 (the kernel is self-contained)."""
    from chronos.model.checkpoint import checkpoint_config_path, tokenizer_metadata
    from . import docs, recommend
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "run_spec.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    prefix = spec.get("export_prefix", "chronos_enzh")
    stem = f"{prefix}_{phase}" + (".best" if which == "best" else "")
    src_ckpt = os.path.join(run_dir, stem + ".pth")
    if not os.path.isfile(src_ckpt):
        raise FileNotFoundError(f"no exported checkpoint {src_ckpt}")
    prov = run_provenance(run_dir, phase, which)
    name = name or f"chronos-enzh-50m-{phase}-s{prov.get('step', 'x')}"
    out_dir = os.path.abspath(out_dir or os.path.join(os.path.dirname(os.path.dirname(run_dir)), "dist", "bundles",
                                                      name))
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"{out_dir} exists and is not empty; pick another --out")
    os.makedirs(out_dir, exist_ok=True)
    model_name = name.replace(".", "_")
    ckpt_rel = f"model/{model_name}.pth"
    log(f"[bundle] weights {src_ckpt} → {ckpt_rel}")
    ckpt_sha = _stable_copy(src_ckpt, os.path.join(out_dir, ckpt_rel))
    # Sidecar: same config, tokenizer path made relative (resolve_checkpoint_tokenizer
    # finds <ckpt dir>/tokenizer first; "tokenizer" is also correct relative to it).
    with open(checkpoint_config_path(src_ckpt), encoding="utf-8") as fh:
        sidecar = json.load(fh)
    tok_src = spec["tokenizer_dir"]
    tok_dst = os.path.join(out_dir, "model", "tokenizer")
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tok_src, f)):
            _copy(os.path.join(tok_src, f), os.path.join(tok_dst, f))
    tok_meta = tokenizer_metadata(tok_dst)
    tok_meta["path"] = "tokenizer"
    sidecar.update(checkpoint=os.path.basename(ckpt_rel), tokenizer=tok_meta)
    sidecar["extra"] = dict(sidecar.get("extra") or {}, bundle=name, source_checkpoint_sha256=ckpt_sha)
    with open(checkpoint_config_path(os.path.join(out_dir, ckpt_rel)), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
    # Evaluation data in the raw layout ``load_eval_sets`` reads.
    raw_root = (spec.get("extra") or {}).get("raw_root")
    flores = []
    for f in FLORES_FILES:
        src = os.path.join(raw_root or "", FLORES_REL, f)
        if raw_root and os.path.exists(src):
            _copy(src, os.path.join(out_dir, "eval", FLORES_REL, f))
            flores.append(f)
    if raw_root and os.path.exists(os.path.join(raw_root, "facebook__flores__eng-zho", "download_manifest.json")):
        with open(os.path.join(raw_root, "facebook__flores__eng-zho", "download_manifest.json"), encoding="utf-8") as fh:
            dl = json.load(fh)
        for item in dl.get("files", []):
            item.pop("local", None)
        recommend.write_json(os.path.join(out_dir, "eval", "facebook__flores__eng-zho", "download_manifest.json"), dl)
    if include_sft_valid:
        valid_src = os.path.join(spec["shards_dir"], "sft", "valid")
        if os.path.isdir(valid_src):
            _copytree(valid_src, os.path.join(out_dir, "data", "sft_valid"))
    del minimind_dir                      # kept for callers of the pre-0.3.2 signature
    for whl in ([wheel] if isinstance(wheel, str) else list(wheel or ())):
        _copy(whl, os.path.join(out_dir, "wheels", os.path.basename(whl)))
    # Recipes: the exact spec that produced these weights (paths made relative).
    roots = _path_roots(run_dir)
    source_spec = _sanitize(spec, roots)
    recommend.write_json(os.path.join(out_dir, "recipes", "source_run_spec.json"), source_spec)
    preset = _planner_preset(int(sidecar["config"].get("vocab_size", 0)), int(sidecar["config"].get("hidden_size", 0)))
    recs = recommend.build_recommendations(spec, preset)
    recommend.write_json(os.path.join(out_dir, "recommended.json"), recs)
    docs.write_examples(os.path.join(out_dir, "examples"))
    manifest = {
        "format": BUNDLE_FORMAT, "name": name, "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_format": "pretrain_tags" if phase == "pretrain" else "chat_sft",
        "model": {"checkpoint": ckpt_rel, "config": checkpoint_config_path(ckpt_rel), "stage": phase,
                  "which": which, "sha256": ckpt_sha, "params": (spec.get("extra") or {}).get("params"),
                  "dtype_on_disk": "see config; weights saved by the MLX exporter",
                  "architecture": {k: sidecar["config"].get(k) for k in (
                      "architecture_version", "hidden_size", "num_hidden_layers", "prelude_layers", "coda_layers",
                      "recurrent_depth", "recurrent_attention_mode", "num_experts", "num_experts_per_tok",
                      "num_shared_experts", "vocab_size", "max_position_embeddings")}},
        "tokenizer": {"dir": "model/tokenizer", **{k: v for k, v in tok_meta.items() if k != "path"}},
        "training": prov,
        "eval": {"flores_plus": flores, "licence": "CC BY-SA 4.0", "root": "eval"},
        "chronos_version": _chronos_version(),
        "chronos_translate_version": _extension_version(),
    }
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    docs.write_docs(out_dir, manifest, recs)
    manifest["files"] = _file_table(out_dir)
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log(f"[bundle] wrote {out_dir} ({len(manifest['files'])} files)")
    return out_dir


def _chronos_version() -> str:
    import chronos
    return getattr(chronos, "__version__", "unknown")


def _extension_version() -> str:
    from .. import __version__
    return __version__


def _rewrite_manifest(bundle_dir: str, manifest: dict) -> None:
    manifest["files"] = _file_table(bundle_dir)
    with open(os.path.join(bundle_dir, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def record_quality_probe(bundle_dir: str, n: int = 50, *, repetition_penalties=(1.0, 1.2),
                         device: str = "cpu", log=print) -> dict:
    """Measure BLEU/chrF of the bundled weights on FLORES+ dev and store it in the manifest."""
    from .runtime import BundleModel, eval_pairs, score
    manifest = load_bundle_manifest(bundle_dir)
    pairs = eval_pairs(bundle_dir, "dev")[:n]
    if not pairs:
        raise FileNotFoundError("bundle has no FLORES+ dev file under eval/")
    bm = BundleModel(bundle_dir, device=device)
    results = {}
    for rp in repetition_penalties:
        log(f"[bundle] probe: {len(pairs)} FLORES+ dev sentences, repetition_penalty={rp}")
        results[f"rp_{rp}"] = score(bm, pairs, repetition_penalty=rp)
    manifest["quality"] = {"set": "flores_plus_dev", "first_n": len(pairs),
                           "label": "measured on the bundled checkpoint (sacrebleu 2.6.0; BLEU tokenize=zh for →zh, "
                                    "13a for →en; corpus-level)",
                           "results": results}
    from . import docs
    docs.write_docs(bundle_dir, manifest)
    _rewrite_manifest(bundle_dir, manifest)
    return manifest["quality"]


def check_bundle(bundle_dir: str, *, smoke: bool = True, device: str = "cpu", log=print) -> dict:
    """Verify every file hash, then (``smoke``) load the model and translate one sentence."""
    bundle_dir = os.path.abspath(bundle_dir)
    manifest = load_bundle_manifest(bundle_dir)
    bad, missing = [], []
    for rel, meta in (manifest.get("files") or {}).items():
        path = os.path.join(bundle_dir, rel)
        if not os.path.isfile(path):
            missing.append(rel)
        elif os.path.getsize(path) != meta["bytes"] or sha256_file(path) != meta["sha256"]:
            bad.append(rel)
    report = {"bundle": manifest["name"], "files": len(manifest.get("files") or {}), "missing": missing,
              "hash_mismatch": bad, "ok": not bad and not missing}
    log(f"[check] {report['files']} files, missing={len(missing)}, hash_mismatch={len(bad)}")
    if smoke and report["ok"]:
        from .runtime import BundleModel, eval_pairs
        t0 = time.monotonic()
        bm = BundleModel(bundle_dir, device=device)
        pairs = eval_pairs(bundle_dir, "dev")[:1] or [("The weather is nice today.", "")]
        hyp = bm.translate([pairs[0][0]], "en_zh", max_new_tokens=64)[0]
        report["smoke"] = {"src": pairs[0][0], "hyp": hyp, "non_empty": bool(hyp.strip()),
                           "seconds": round(time.monotonic() - t0, 1), "device": bm.device}
        report["ok"] = report["ok"] and bool(hyp.strip())
        log(f"[check] smoke en→zh: {hyp!r}")
    return report


def pack_bundle(bundle_dir: str, archive: str | None = None, log=print) -> str:
    """``tar.gz`` the bundle (top-level folder = bundle name) and write ``<archive>.sha256``."""
    bundle_dir = os.path.abspath(bundle_dir)
    archive = os.path.abspath(archive or bundle_dir.rstrip(os.sep) + ".tar.gz")
    if os.path.exists(archive):
        raise FileExistsError(f"{archive} exists")
    base = os.path.basename(bundle_dir.rstrip(os.sep))

    def _filter(info):
        return None if os.path.basename(info.name) in _SKIP_NAMES or info.name.endswith(".pyc") else info

    tmp = archive + ".partial"
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        tar.add(bundle_dir, arcname=base, filter=_filter)
    os.replace(tmp, archive)
    digest = sha256_file(archive)
    with open(archive + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  {os.path.basename(archive)}\n")
    log(f"[bundle] {archive} ({os.path.getsize(archive) / 2**20:.1f} MiB) sha256={digest}")
    return archive
