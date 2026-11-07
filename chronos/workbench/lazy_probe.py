"""Lazy-expert validation probe: efficiency, hit rate and fidelity of a checkpoint.

At a validation point the trainer has just written a checkpoint.  This probe
loads that checkpoint the way inference would (``chronos.lazy_experts``):
trunk in memory, routed experts paged from an SSD cache, and decodes a few
fixed prompts under every requested setting.  Each setting is compared with a
full-memory reference decode of the same prompts on the same backend.

Measured per variant (labelled *measured*, conditions in ``conditions``):

* efficiency – decode tokens/s, prefill/decode seconds, on-demand loads and
  their load time, prefetch wait, cluster loads, peak RSS;
* hit rate   – resident (selection) hit rate, lookahead prediction hit rate,
  cold-miss prediction hit rate;
* fidelity   – greedy token agreement with the reference (exact match,
  matching prefix, position-wise agreement).

Settings: backend (mlx / torch), paging (auto / cluster / expert), residency
(planned / window), RAM ratio, Torch miss policy, residency decay, prefetch
waits and self-speculative decoding.  A sweep is the Cartesian product of the
lists given, deduplicated and capped at ``max_variants`` (the cap is reported,
never silent).
"""
from __future__ import annotations

import itertools
import os
import time
from dataclasses import asdict, dataclass, field

# Prompts used when the checkpoint has no tokenizer: fixed id sequences, so
# results from different validation points are comparable.
_SYNTHETIC_PROMPTS = ((5, 17, 33, 2, 61, 9, 40, 12), (7, 3, 29, 44, 18, 6, 51, 23, 11, 2))
DEFAULT_PROMPTS = (
    "Summarize in one sentence: the weather is nice today and the park is busy.",
    "Explain what a mixture-of-experts layer does in one sentence.",
)


@dataclass
class ProbeSweep:
    """What to measure; every list is one sweep axis."""

    backends: list[str] = field(default_factory=lambda: ["auto"])
    paging: list[str] = field(default_factory=lambda: ["auto", "expert"])   # deduped after resolving
    residency: list[str] = field(default_factory=lambda: ["planned"])
    ram_ratios: list[float] = field(default_factory=lambda: [0.5, 1.0])
    miss_policies: list[str] = field(default_factory=lambda: ["sync_on_demand"])   # torch only
    residency_decay: list[float] = field(default_factory=lambda: [0.7])           # mlx planned only
    decode_wait_s: list[float] = field(default_factory=lambda: [0.012])            # mlx only
    verify_wait_s: list[float] = field(default_factory=lambda: [0.02])             # mlx speculative only
    speculative: list[bool] = field(default_factory=lambda: [False])
    draft_depth: int = 1
    gamma: int = 4
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    max_new_tokens: int = 32
    max_variants: int = 12
    dtype: str = "float32"
    cache_root: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "ProbeSweep":
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        out = cls(**{k: v for k, v in data.items() if k in known})
        for name in ("backends", "paging", "residency", "ram_ratios", "miss_policies", "residency_decay",
                     "decode_wait_s", "verify_wait_s", "speculative", "prompts"):
            value = getattr(out, name)
            if not isinstance(value, (list, tuple)):
                setattr(out, name, [value])
        return out.validated()

    def validated(self) -> "ProbeSweep":
        from chronos.expert_clusters import PAGING_MODES
        from chronos.lazy_experts import RESIDENCY_MODES
        checks = (("paging", self.paging, PAGING_MODES), ("residency", self.residency, RESIDENCY_MODES),
                  ("miss_policies", self.miss_policies, ("on_demand", "sync_on_demand", "fallback_diagnostic")),
                  ("backends", self.backends, ("auto", "mlx", "cpu", "mps", "cuda", "xpu")))
        for name, values, allowed in checks:
            bad = [v for v in values if v not in allowed]
            if bad or not values:
                raise ValueError(f"{name}: {bad or 'empty'} not in {allowed}")
        if any(float(r) <= 0 for r in self.ram_ratios) or not self.ram_ratios:
            raise ValueError("ram_ratios must be > 0")
        if any(not 0.0 <= float(d) < 1.0 for d in self.residency_decay):
            raise ValueError("residency_decay must be in [0, 1)")
        if any(float(w) < 0 for w in (*self.decode_wait_s, *self.verify_wait_s)):
            raise ValueError("waits must be >= 0")
        if not 1 <= int(self.max_new_tokens) <= 512 or not 1 <= int(self.max_variants) <= 256:
            raise ValueError("max_new_tokens in [1, 512], max_variants in [1, 256]")
        if int(self.draft_depth) < 1 or not 1 <= int(self.gamma) <= 8:
            raise ValueError("draft_depth >= 1, gamma in [1, 8]")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_backend(requested: str) -> str:
    from chronos.backend import select
    return select(None if requested == "auto" else requested)


def expand_variants(sweep: ProbeSweep, resolve_paging=None) -> tuple[list[dict], int]:
    """Deduplicated settings; axes that do not apply to a backend collapse.

    ``resolve_paging(p)`` maps a paging request to the layout it produces for
    the checkpoint (``auto`` → ``cluster`` or ``expert``); settings that
    resolve to the same layout are measured once.  Returns
    ``(variants, total_before_cap)``."""
    resolve_paging = resolve_paging or (lambda p: p)
    seen, out = set(), []
    for backend in dict.fromkeys(resolve_backend(b) for b in sweep.backends):
        mlx = backend == "mlx"
        for paging, residency, ratio, policy, decay, dwait, vwait, spec in itertools.product(
                sweep.paging, sweep.residency if mlx else ["-"], sweep.ram_ratios,
                ["-"] if mlx else sweep.miss_policies, sweep.residency_decay, sweep.decode_wait_s,
                sweep.verify_wait_s, sweep.speculative):
            v = {"backend": backend, "paging": resolve_paging(paging), "ram_ratio": round(float(ratio), 4),
                 "speculative": bool(spec)}
            if mlx:
                v.update(residency=residency, decode_wait_s=float(dwait))
                if residency == "planned":
                    v["residency_decay"] = float(decay)
                if spec:
                    v["verify_wait_s"] = float(vwait)
            else:
                v["miss_policy"] = policy
            key = tuple(sorted(v.items()))
            if key not in seen:
                seen.add(key)
                out.append(v)
    return out[: int(sweep.max_variants)], len(out)


def encode_prompts(model_path: str, sweep: ProbeSweep, tokenizer_dir: str | None = None,
                   vocab_size: int | None = None) -> tuple[list[list[int]], str]:
    """Prompt ids from the checkpoint's tokenizer, else the fixed synthetic ids.

    A tokenizer whose ids do not fit the model's vocabulary (a fallback
    tokenizer for a checkpoint that recorded none) is not used."""
    try:
        from transformers import AutoTokenizer
        from chronos.workbench.inference import _encode_prompt, _inference_tokenizer_dir
        tok = AutoTokenizer.from_pretrained(tokenizer_dir or _inference_tokenizer_dir({}, model_path))
        ids = [_encode_prompt(tok, p, False)[0] for p in sweep.prompts if str(p).strip()]
        if ids and (not vocab_size or max(max(x) for x in ids if x) < int(vocab_size)):
            return ids, "tokenizer"
    except Exception:
        pass
    limit = int(vocab_size or 64)
    return [[t % limit for t in p] for p in _SYNTHETIC_PROMPTS], "synthetic"


def agreement(tokens: list[int], ref: list[int]) -> dict:
    """Greedy fidelity of one decode against the full-memory reference."""
    prefix = 0
    for a, b in zip(tokens, ref):
        if a != b:
            break
        prefix += 1
    n = max(len(ref), 1)
    same = sum(1 for a, b in zip(tokens, ref) if a == b)
    return {"exact": tokens == ref, "prefix": prefix, "positionwise": same / n, "ref_len": len(ref)}


def _decode(backend: str, cfg, path: str, ids: list[int], sweep: ProbeSweep, variant: dict | None):
    """One greedy decode; ``variant`` None = full-memory reference."""
    from chronos.workbench.inference import (
        _clone_model_cfg, _run_mlx_full_dram, _run_mlx_lazy_inference, _run_torch_full_dram,
        _run_torch_inference, _speculative_options,
    )
    spec = _speculative_options(bool(variant and variant["speculative"]), sweep.draft_depth, sweep.gamma)
    run_cfg, n = _clone_model_cfg(cfg), int(sweep.max_new_tokens)
    if backend == "mlx":
        if variant is None:
            return _run_mlx_full_dram(run_cfg, path, ids, n, 0.0, do_sample=False, speculative=None)
        return _run_mlx_lazy_inference(
            run_cfg, path, ids, n, 0.0, do_sample=False, ram_load_ratio=variant["ram_ratio"], quality_safe=True,
            speculative=spec, residency=variant["residency"], paging=variant["paging"],
            residency_decay=variant.get("residency_decay"), decode_wait_s=variant.get("decode_wait_s"),
            verify_wait_s=variant.get("verify_wait_s"))
    mask = [1] * len(ids)
    if variant is None:
        return _run_torch_full_dram(backend, run_cfg, path, ids, mask, n, 1.0, None, do_sample=False)
    return _run_torch_inference(backend, run_cfg, path, ids, mask, n, 1.0, None, do_sample=False,
                                ram_load_ratio=variant["ram_ratio"], miss_policy=variant["miss_policy"],
                                speculative=spec, paging=variant["paging"])


# Stats summed over prompts, and rates recomputed from their counts.
_SUMS = ("on_demand_loads", "on_demand_load_time_s", "prefetch_wait_time_s", "sync_ssd_loads", "cluster_loads",
         "cache_hits", "cache_misses", "prediction_hits", "prediction_total", "cold_prediction_hits",
         "cold_prediction_total", "prefill_time_s", "decode_time_s")


def _merge(stats: list[dict], tokens: int, elapsed: float) -> dict:
    out = {k: round(sum(float(s.get(k, 0) or 0) for s in stats), 4) for k in _SUMS}
    sel = out["cache_hits"] + out["cache_misses"]
    out["resident_hit_rate"] = round(out["cache_hits"] / sel, 4) if sel else (
        round(sum(float(s.get("resident_hit_rate", 0) or 0) for s in stats) / max(len(stats), 1), 4))
    if out["prediction_total"]:
        out["prediction_hit_rate"] = round(out["prediction_hits"] / out["prediction_total"], 4)
    elif any("prediction_total" in s for s in stats):
        out["prediction_hit_rate"] = None            # this path made no lookahead predictions
    else:
        out["prediction_hit_rate"] = round(sum(float(s.get("prediction_hit_rate", 0) or 0) for s in stats)
                                           / max(len(stats), 1), 4)
    if out["cold_prediction_total"]:
        out["cold_miss_predict_hit_rate"] = round(out["cold_prediction_hits"] / out["cold_prediction_total"], 4)
    out["tokens"] = tokens
    out["tokens_per_s"] = round(tokens / elapsed, 2) if elapsed > 0 else 0.0
    out["decode_tokens_per_s"] = round(tokens / out["decode_time_s"], 2) if out["decode_time_s"] > 0 else None
    last = stats[-1] if stats else {}
    for key in ("paging_resolved", "num_pages", "effective_expert_budget", "ideal_active_experts",
                "eviction_policy", "num_clusters", "storage_format"):
        if key in last:
            out[key] = last[key]
    peaks = [float(s.get("rss_after_decode_gb") or 0) for s in stats]
    out["peak_rss_gb"] = round(max(peaks), 3) if peaks else None
    spec = [s.get("speculative") for s in stats if isinstance(s.get("speculative"), dict)]
    if spec:
        acc = [float(x.get("acceptance_rate", 0) or 0) for x in spec]
        out["spec_acceptance_rate"] = round(sum(acc) / len(acc), 4)
    return out


def variant_label(v: dict) -> str:
    parts = [v["backend"], v["paging"], f"r{v['ram_ratio']:g}"]
    if v["backend"] == "mlx":
        parts.append(v["residency"])
        if "residency_decay" in v:
            parts.append(f"d{v['residency_decay']:g}")
        parts.append(f"w{v['decode_wait_s'] * 1000:g}ms")
    else:
        parts.append(v["miss_policy"])
    if v["speculative"]:
        parts.append("spec")
    return "/".join(parts)


def run_probe(model_path: str, sweep: ProbeSweep | dict | None = None, *, tokenizer_dir: str | None = None,
              log=None) -> dict:
    """Measure every variant of ``sweep`` on ``model_path``; see the module docstring.

    Lazy variants run first and the full-memory references last, so process
    RSS after a lazy decode is not inflated by a fully loaded model."""
    from chronos.lazy_experts import page_layout
    from chronos.workbench.inference import _build_model_cfg
    log = log or (lambda _m: None)
    sweep = sweep if isinstance(sweep, ProbeSweep) else ProbeSweep.from_dict(sweep)
    cfg, _sniffed, _note = _build_model_cfg({"dtype": sweep.dtype}, model_path)
    if not getattr(cfg, "use_moe", True) or int(getattr(cfg, "num_experts", 0) or 0) < 2:
        return {"skipped": "dense model: no routed experts to page", "checkpoint": model_path}
    variants, total = expand_variants(sweep, lambda p: page_layout(cfg, p)["resolved"])
    prompts, prompt_source = encode_prompts(model_path, sweep, tokenizer_dir, int(cfg.vocab_size))
    t_start = time.monotonic()
    results = []
    for v in variants:
        label = variant_label(v)
        log(f"[lazy-probe] {label}")
        stats, toks, t0 = [], [], time.monotonic()
        try:
            for ids in prompts:
                out, st = _decode(v["backend"], cfg, model_path, ids, sweep, v)
                toks.append(list(out))
                stats.append(st)
        except Exception as exc:                 # one bad setting must not hide the others
            results.append({"label": label, "settings": v, "error": f"{type(exc).__name__}: {exc}"[:400]})
            continue
        results.append({"label": label, "settings": v, "tokens": toks,
                        "measured": _merge(stats, sum(len(t) for t in toks), time.monotonic() - t0)})
    references = {}
    for backend in dict.fromkeys(r["settings"]["backend"] for r in results if "tokens" in r):
        try:
            refs, t0 = [], time.monotonic()
            ref_stats = []
            for ids in prompts:
                out, st = _decode(backend, cfg, model_path, ids, sweep, None)
                refs.append(list(out))
                ref_stats.append(st)
            references[backend] = {"tokens": refs,
                                   "measured": _merge(ref_stats, sum(len(t) for t in refs), time.monotonic() - t0)}
        except Exception as exc:
            references[backend] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
    for r in results:
        ref = references.get(r["settings"]["backend"], {})
        if "tokens" not in r or "tokens" not in ref:
            continue
        per = [agreement(a, b) for a, b in zip(r.pop("tokens"), ref["tokens"])]
        r["fidelity"] = {"exact_match_rate": round(sum(p["exact"] for p in per) / len(per), 4),
                         "positionwise_agreement": round(sum(p["positionwise"] for p in per) / len(per), 4),
                         "min_matching_prefix": min(p["prefix"] for p in per),
                         "reference_tokens": sum(p["ref_len"] for p in per)}
        full_tps = ref["measured"].get("tokens_per_s") or 0
        if full_tps:
            r["measured"]["speed_vs_full"] = round(r["measured"]["tokens_per_s"] / full_tps, 4)
    for ref in references.values():
        ref.pop("tokens", None)
    layout = page_layout(cfg)
    return {
        "checkpoint": os.path.abspath(model_path),
        "trained_with_clusters": layout["trained_with_clusters"], "auto_paging": layout["resolved"],
        "num_experts": int(cfg.num_experts), "top_k": int(cfg.num_experts_per_tok),
        "variants": results, "references": references,
        "truncated": total > len(variants), "variants_requested": total,
        "elapsed_s": round(time.monotonic() - t_start, 2),
        "conditions": {
            "label": "measured", "decode": "greedy", "max_new_tokens": int(sweep.max_new_tokens),
            "prompts": len(prompts), "prompt_source": prompt_source, "dtype": sweep.dtype,
            "reference": "full-memory decode of the same prompts on the same backend",
            "rss_note": "process RSS after each decode; variants share one process in the listed order",
        },
        "sweep": sweep.as_dict(),
    }


def _sidecar(path: str) -> str:
    return os.path.splitext(path)[0] + ".config.json"


def newest_checkpoint(candidates) -> str | None:
    """The most recently written existing ``.pth`` among ``candidates``."""
    found = [c for c in candidates if c and os.path.isfile(c)]
    return max(found, key=os.path.getmtime) if found else None


def _settled(path: str, wait_s: float = 1.0, tries: int = 10) -> bool:
    """True once the checkpoint's size and mtime are unchanged across ``wait_s``.

    ``save_state_dict_with_config`` writes the ``.pth`` first and its
    ``.config.json`` sidecar second, so a sidecar that is older than the
    weights (or still missing) is waited for as well; checkpoints saved
    without any sidecar are accepted once stable."""
    def sig():
        st = os.stat(path)
        side = _sidecar(path)
        fresh = os.path.isfile(side) and os.path.getmtime(side) >= st.st_mtime
        return st.st_size, st.st_mtime_ns, fresh
    try:
        last = sig()
        for attempt in range(tries):
            time.sleep(wait_s)
            now = sig()
            if now == last and (now[2] or attempt >= 2):
                return True
            last = now
    except OSError:
        return False
    return False


class ProbeScheduler:
    """Runs one probe at a time in a low-priority child process.

    ``request`` copies the checkpoint (and its config sidecar) to
    ``<run_dir>/lazy_probe/step_<n>/`` first, so training can overwrite the
    original while the probe reads the snapshot; the snapshot and the probe's
    expert cache are deleted when the probe ends.  Results are appended to
    ``<run_dir>/lazy_probe.jsonl``; ``poll`` returns the finished summaries
    (one line each).  A request while a probe runs is recorded as
    skipped (validation never waits for the probe).
    """

    def __init__(self, run_dir: str, sweep: dict | None = None, *, tokenizer_dir: str | None = None,
                 nice: int = 10, log=None):
        self.run_dir = os.path.abspath(run_dir)
        self.sweep = ProbeSweep.from_dict(sweep).as_dict()          # validate up front
        self.tokenizer_dir = tokenizer_dir or None
        self.nice, self.log = int(nice), (log or (lambda _m: None))
        self.proc = None
        self.current: dict | None = None
        self.results_path = os.path.join(self.run_dir, "lazy_probe.jsonl")
        os.makedirs(self.run_dir, exist_ok=True)

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def skip(self, *, phase: str, step: int, reason: str) -> dict:
        """Record a validation that will not be probed (e.g. superseded by a newer one)."""
        return self._record({"phase": phase, "step": step, "skipped": reason})

    def request(self, checkpoint: str | None, *, phase: str, step: int) -> dict:
        import json
        import shutil
        import subprocess
        import sys
        if not checkpoint or not os.path.isfile(checkpoint):
            return self._record({"phase": phase, "step": step, "skipped": "no checkpoint on disk yet"})
        if self.busy():
            return self._record({"phase": phase, "step": step, "skipped": "previous probe still running"})
        if not _settled(checkpoint):
            return self._record({"phase": phase, "step": step, "skipped": "checkpoint still being written"})
        work = os.path.join(self.run_dir, "lazy_probe", f"{phase}_step_{int(step):08d}")
        os.makedirs(work, exist_ok=True)
        snap = os.path.join(work, os.path.basename(checkpoint))
        shutil.copy2(checkpoint, snap)
        if os.path.isfile(_sidecar(checkpoint)):
            shutil.copy2(_sidecar(checkpoint), _sidecar(snap))
        sweep_file = os.path.join(work, "sweep.json")
        with open(sweep_file, "w", encoding="utf-8") as fh:
            json.dump(self.sweep, fh)
        argv = [sys.executable, "-m", "chronos.workbench.lazy_probe", "--checkpoint", snap,
                "--sweep-file", sweep_file, "--out", os.path.join(work, "result.json")]
        if self.tokenizer_dir:
            argv += ["--tokenizer-dir", self.tokenizer_dir]
        preexec = (lambda: os.nice(self.nice)) if hasattr(os, "nice") and self.nice > 0 else None
        self.log_file = open(os.path.join(work, "probe.log"), "w", encoding="utf-8")
        self.proc = subprocess.Popen(argv, stdout=self.log_file, stderr=subprocess.STDOUT,
                                     preexec_fn=preexec, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.current = {"phase": phase, "step": int(step), "checkpoint": os.path.abspath(checkpoint),
                        "work": work, "t0": time.monotonic()}
        self.log(f"[lazy-probe] started for {phase} step {step} ({os.path.basename(checkpoint)})")
        return {"phase": phase, "step": step, "started": True}

    def poll(self) -> list[dict]:
        """Entries of probes that finished since the last call (at most one)."""
        import json
        import shutil
        if self.proc is None or self.proc.poll() is None:
            return []
        cur, code = self.current or {}, self.proc.returncode
        self.proc, self.current = None, None
        self.log_file.close()
        result, out = None, os.path.join(cur.get("work", ""), "result.json")
        try:
            with open(out, encoding="utf-8") as fh:
                result = json.load(fh)
        except (OSError, ValueError):
            pass
        entry = {"phase": cur.get("phase"), "step": cur.get("step"), "checkpoint": cur.get("checkpoint"),
                 "wall_s": round(time.monotonic() - cur.get("t0", time.monotonic()), 1)}
        if result is None:
            try:
                with open(os.path.join(cur["work"], "probe.log"), encoding="utf-8") as fh:
                    tail = fh.read()[-600:]
            except (OSError, KeyError):
                tail = ""
            entry["error"] = f"probe exited {code}: {tail.strip()}"[:800]
        else:
            entry["result"] = result
        shutil.rmtree(cur.get("work", ""), ignore_errors=True)      # our snapshot + cache only
        self._record(entry)
        return [entry]

    def wait(self, timeout_s: float = 600.0) -> list[dict]:
        t0 = time.monotonic()
        while self.busy() and time.monotonic() - t0 < timeout_s:
            time.sleep(0.5)
        if self.busy():
            self.stop()
        return self.poll()

    def stop(self) -> None:
        if self.busy():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def _record(self, entry: dict) -> dict:
        import json
        from datetime import datetime
        with open(self.results_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **entry},
                                ensure_ascii=False, default=str) + "\n")
        return entry


def watch(run_dir: str, sweep: dict | None = None, *, out_dir: str | None = None, tokenizer_dir: str | None = None,
          poll_s: float = 5.0, once: bool = False, log=print) -> int:
    """Probe any trainer's run directory from outside the training process.

    Tails ``<run_dir>/metrics.jsonl``; after each ``validation`` event it
    probes the newest ``.pth`` under ``run_dir`` written since the previous
    probe (``chronos longrun`` exports ``*.best.pth`` at an improving
    validation; the Web UI trainer saves before validating).  Results go to
    ``<out_dir>/lazy_probe.jsonl`` (default: the run directory).  The run is
    only read.  ``once``: probe the newest checkpoint now and exit.
    """
    import glob
    import json
    run_dir = os.path.abspath(run_dir)
    sched = ProbeScheduler(out_dir or run_dir, sweep, tokenizer_dir=tokenizer_dir, log=log)

    def newest():
        pths = [p for p in glob.glob(os.path.join(run_dir, "**", "*.pth"), recursive=True)
                if os.sep + "lazy_probe" + os.sep not in p]
        return newest_checkpoint(pths)

    if once:
        sched.request(newest(), phase="manual", step=0)
        for entry in sched.wait(timeout_s=24 * 3600):
            log(json.dumps({k: v for k, v in entry.items() if k != "result"}, ensure_ascii=False))
        return 0
    metrics, offset, probed = os.path.join(run_dir, "metrics.jsonl"), 0, 0.0
    if os.path.exists(metrics):
        offset = os.path.getsize(metrics)                 # only validations from now on
    try:
        while True:
            sched.poll()
            if os.path.exists(metrics) and os.path.getsize(metrics) > offset:
                with open(metrics, encoding="utf-8") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                cut = chunk.rfind("\n") + 1                # keep a partial last line for later
                offset += len(chunk[:cut].encode("utf-8"))
                for line in chunk[:cut].splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("event") != "validation":
                        continue
                    ckpt = newest()
                    if ckpt and os.path.getmtime(ckpt) > probed:
                        probed = os.path.getmtime(ckpt)
                        sched.request(ckpt, phase=str(row.get("phase", "")), step=int(row.get("step", 0) or 0))
            time.sleep(poll_s)
    except KeyboardInterrupt:
        sched.stop()
    return 0


def main(argv=None) -> int:
    import argparse
    import json
    import shutil
    import tempfile
    p = argparse.ArgumentParser(prog="chronos lazy-probe", description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="probe this checkpoint once")
    src.add_argument("--watch", metavar="RUN_DIR", help="probe after every validation of a run directory")
    p.add_argument("--out-dir", default=None, help="--watch: where lazy_probe.jsonl goes (default: the run dir)")
    p.add_argument("--sweep", default=None, help="JSON object with ProbeSweep fields")
    p.add_argument("--sweep-file", default=None)
    p.add_argument("--tokenizer-dir", default=None)
    p.add_argument("--out", default=None, help="write the result JSON here (default: stdout)")
    p.add_argument("--keep-cache", action="store_true", help="keep the scratch expert cache")
    a = p.parse_args(argv)
    sweep = json.loads(a.sweep) if a.sweep else {}
    if a.sweep_file:
        with open(a.sweep_file, encoding="utf-8") as fh:
            sweep = json.load(fh)
    if a.watch:
        return watch(a.watch, sweep, out_dir=a.out_dir, tokenizer_dir=a.tokenizer_dir)
    scratch = tempfile.mkdtemp(prefix="chronos_probe_cache_")
    os.environ["CHRONOS_EXPERT_CACHE_ROOT"] = scratch
    try:
        result = run_probe(a.checkpoint, sweep, tokenizer_dir=a.tokenizer_dir,
                           log=lambda m: print(m, flush=True))
    finally:
        if not a.keep_cache:
            shutil.rmtree(scratch, ignore_errors=True)
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if a.out:
        with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(a.out + ".tmp", a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
