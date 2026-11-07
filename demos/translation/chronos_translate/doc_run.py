"""Time-budgeted MLX run on doc-mix shards (``chronos_translate.corpus.docmix``).

Same loop, checkpoints, telemetry and resume as the kernel ``LongRun`` (via
``TranslationLongRun``); this subclass changes only what the doc-mix data needs:

* SFT batches come from ``DocSFTRecords`` (target record drawn ∝ length, so
  short sentences and 4k-token documents get the same passes per record)
  and carry both directions (``.dir``: 0 = en→zh, 1 = zh→en).
* Translation eval covers both directions at sentence level (FLORES+
  devtest) and document level (FLORES+ devtest articles, WMT24++ documents
  that fit the context).  ``bleu_zh`` / ``chrf`` keep the en→zh sentence
  scores under the live run's key names, so monitors and comparisons keep working.
* ``micro_tokens`` is fitted at launch to the GPU's recommended working set
  (largest power-of-two fraction that runs one full-length step without an
  allocation failure and under the memory target); ``tokens_per_step`` is
  kept by raising gradient accumulation, so the optimisation recipe is unchanged.

Paths in the recipe may be relative to the recipe file, so a package can be
copied to another machine and launched in place.
"""
from __future__ import annotations

import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chronos.mlx.training.longrun import PhaseSpec, PretrainSource, SFTSource
from chronos.mlx.training.longrun import load_spec as _load_spec
from chronos.mlx.training.longrun import main as _longrun_main

from .run import EXTRA_PATH_KEYS, TranslationLongRun, TranslationRunSpec

DIR_NAMES = ("en_zh", "zh_en")


class _PhaseDone(Exception):
    """Raised from ``_emit`` when a phase reaches ``extra.target_passes``."""


class DocSFTSource(SFTSource):
    """``SFTSource`` over doc-mix shards (records ∝ length, both directions)."""

    def __init__(self, shards_dir, seed):  # noqa: D107 - same interface as SFTSource
        from chronos_translate.corpus.docmix import DocSFTRecords
        self.records = DocSFTRecords(os.path.join(shards_dir, "sft", "train"), seed=seed)
        self.valid = DocSFTRecords(os.path.join(shards_dir, "sft", "valid"), seed=seed + 1)

    def seen(self) -> dict:
        s = self.records.seen
        n = max(1, s["records"])
        tot = max(1, s["tokens"])
        return {"sft_drawn_records": s["records"], "sft_drawn_tokens": s["tokens"],
                "sft_drawn_zh_en_share": round(s["dir"][1] / n, 4),
                "sft_drawn_len_share": {k: round(v / tot, 4) for k, v in sorted(s["len_bins"].items())},
                "sft_passes": round(s["tokens"] / max(1, self.records.total_tokens()), 4)}


def _docmix_manifest(shards_dir: str) -> dict:
    path = os.path.join(shards_dir, "docmix_manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{shards_dir} is not a doc-mix shard folder (no docmix_manifest.json)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _gpu_info() -> dict:
    """Metal device info (unified memory: the GPU working set bounds CPU-stream runs too)."""
    if not hasattr(mx, "device_info"):
        return {}
    try:
        return dict(mx.device_info(mx.gpu))
    except Exception:  # noqa: BLE001 - no Metal device
        return dict(mx.device_info())


class DocLongRun(TranslationLongRun):
    def __init__(self, spec: TranslationRunSpec, log=print):
        self.docmix = _docmix_manifest(spec.shards_dir)
        self.ctx_len = int(self.docmix["spec"]["ctx_len"])
        if int(spec.model_config.get("max_position_embeddings", self.ctx_len)) < self.ctx_len:
            raise ValueError("model_config.max_position_embeddings is below the shards' ctx_len")
        super().__init__(spec, log=log)
        self._data_tokens: dict = {}
        self._projected: set = set()

    # ---- data ----------------------------------------------------------------
    def _source(self, phase: PhaseSpec):
        if phase.name == "pretrain":
            src = PretrainSource(self.spec.shards_dir, phase.seq_len, self.spec.seed)
            self._data_tokens[phase.name] = int(src.stream.total_tokens)
        else:
            src = DocSFTSource(self.spec.shards_dir, self.spec.seed)
            self._data_tokens[phase.name] = int(src.records.supervised_tokens())
        return src

    # ---- memory fit ------------------------------------------------------------
    def _probe_peak(self, grad_fn, rows: int, seq: int) -> float:
        """Peak bytes of two accumulated micro-steps on ``rows`` full-length rows."""
        from chronos.mlx.training.trainer import _add_grads
        rng = np.random.default_rng(0)
        ids = mx.array(rng.integers(8, int(self.config.vocab_size), size=(rows, seq)), dtype=mx.int32)
        mx.clear_cache()
        mx.reset_peak_memory()
        _, g1 = grad_fn(self.model, ids, ids)
        mx.eval(g1)
        _, g2 = grad_fn(self.model, ids, ids)
        acc = _add_grads(g1, g2)
        mx.eval(acc)
        peak = float(mx.get_peak_memory())
        del g1, g2, acc
        mx.clear_cache()
        return peak

    def _fit_micro_tokens(self, phase: PhaseSpec) -> int:
        """Largest ``micro_tokens`` ≤ the recipe's whose step fits the memory target.

        Peak memory is measured at 1 and 2 full-context rows and extrapolated
        linearly (activations scale with rows); the chosen size is then run
        once to confirm.  The result is stored in the run state, so a resume
        keeps the same micro-batches.
        """
        fitted = self.state.setdefault("fitted_micro_tokens", {})
        if phase.name in fitted:
            return int(fitted[phase.name])
        extra = self.spec.extra or {}
        seq = max(int(phase.seq_len), self.ctx_len) if phase.name != "pretrain" else int(phase.seq_len)
        want_rows = max(1, int(phase.micro_tokens) // seq)
        info = _gpu_info()
        limit = float(info.get("max_recommended_working_set_size") or info.get("memory_size") or 0)
        if extra.get("micro_fit", "auto") == "off" or not limit:
            fitted[phase.name] = int(phase.micro_tokens)
            return int(phase.micro_tokens)
        target = float(extra.get("micro_fit_fraction", 0.7)) * limit
        grad_fn = nn.value_and_grad(self.model, self._loss_fn)
        p1 = self._probe_peak(grad_fn, 1, seq)
        p2 = self._probe_peak(grad_fn, 2, seq) if want_rows >= 2 else p1
        per_row = max(p2 - p1, 1.0)
        rows = 1
        while rows * 2 <= want_rows and p1 + (rows * 2 - 1) * per_row <= target:
            rows *= 2
        peak = self._probe_peak(grad_fn, rows, seq) if rows > 2 else (p2 if rows == 2 else p1)
        while rows > 1 and peak > target:
            rows //= 2
            peak = self._probe_peak(grad_fn, rows, seq)
        micro = rows * seq if rows < want_rows else int(phase.micro_tokens)
        fitted[phase.name] = micro
        report = {"phase": phase.name, "seq": seq, "rows": rows, "micro_tokens": micro,
                  "recipe_micro_tokens": int(phase.micro_tokens), "peak_gb": round(peak / 1e9, 2),
                  "target_gb": round(target / 1e9, 2), "limit_gb": round(limit / 1e9, 2),
                  "device": info.get("device_name") or info.get("architecture")}
        self.state.setdefault("micro_fit", {})[phase.name] = report
        self.log(f"[doc] micro-batch fit: {json.dumps(report, ensure_ascii=False)}")
        if peak > target:
            self.log(f"[doc] warning: one {seq}-token row needs {peak / 1e9:.1f} GB, above the "
                     f"{target / 1e9:.1f} GB target; training continues at 1 row per micro-batch")
        return micro

    # ---- training ------------------------------------------------------------
    def run_phase(self, pi: int, deadline_s: float, max_steps: int | None = None):
        from dataclasses import replace
        original = self.spec.phases[pi]
        micro = self._fit_micro_tokens(original)
        phase = replace(original, micro_tokens=micro)
        accum = max(1, phase.tokens_per_step // micro)
        data = self._data_tokens.get(phase.name)
        self.log(f"[doc] {phase.name}: micro_tokens={micro} × accum {accum} = {micro * accum} tokens/step "
                 f"(recipe {original.tokens_per_step}); ctx={self.ctx_len}")
        if self.state["phase_step"] == 0:
            self.state.setdefault("phase_start_tokens", {})[phase.name] = int(self.state["tokens_seen"])
        self._phase_name = phase.name
        self.spec.phases[pi] = phase
        try:
            super().run_phase(pi, deadline_s, max_steps=max_steps)
        except _PhaseDone:
            # Same closing steps as LongRun.run_phase after its loop.
            source = self._active_source
            self.log(f"[doc] {phase.name}: reached {self._passes(phase.name):.2f} passes "
                     f"(target {self._target(phase.name)}) at step {self.state['phase_step']}")
            self.state["rng"][phase.name] = source.rng_state()
            self._beat("evaluating")
            self._periodic_eval(source, phase)
            self.export_checkpoint(f"{self.spec.export_prefix}_{phase.name}", stage=phase.name)
            self._emit({"phase": phase.name, "event": "phase_end", "step": self.state["phase_step"],
                        "tokens_seen": self.state["tokens_seen"], "reason": "target_passes"})
        finally:
            self.spec.phases[pi] = original
            self._phase_name = None
        if data and max_steps is None:
            self.log(f"[doc] {phase.name}: {self._passes(phase.name):.2f} passes over {data:,} "
                     f"{'tokens' if phase.name == 'pretrain' else 'supervised tokens'}")

    def _target(self, name: str) -> float | None:
        t = ((self.spec.extra or {}).get("target_passes") or {}).get(name)
        return float(t) if t else None

    def _emit(self, record: dict):
        """Pass accounting on training rows; ends the phase at ``target_passes``.

        Once the cumulative rate (tokens per wall second, evals included) is
        known, the time schedule's phase budget shrinks to the predicted time of
        the target, so warmup + cosine end where the data passes do.  Time left
        over moves to the next phase (its deadline is fixed by the recipe).
        """
        name = getattr(self, "_phase_name", None)
        target = self._target(name) if name and "event" not in record and record.get("phase") == name else None
        done = False
        if target and self._data_tokens.get(name):
            st = self.state
            data = self._data_tokens[name]
            seen = int(st["tokens_seen"]) - int((st.get("phase_start_tokens") or {}).get(name, 0))
            wall = float(st.get("phase_wall_s") or 0.0)
            done = seen >= target * data
            if wall >= 60 and seen > 0 and self.spec.schedule == "time" and st.get("phase_budget_s"):
                need = wall * target * data / seen
                orig = st.setdefault("phase_budget_orig_s", {}).setdefault(name, float(st["phase_budget_s"]))
                st["phase_budget_s"] = max(wall, min(orig, need))
                step = int(record.get("step") or st["phase_step"])
                st["phase_total"] = record["total"] = max(step, int(step * st["phase_budget_s"] / wall))
            record["passes"] = round(seen / data, 4)
            if name not in self._projected and wall >= 600:
                self._projected.add(name)
                rate = seen / wall
                budget = (st.get("phase_budget_orig_s") or {}).get(name, st.get("phase_budget_s") or wall)
                self.log(f"[doc] {name}: {rate:,.0f} tok/s over {wall / 60:.0f} min → "
                         f"{rate * budget / data:.2f} passes fit the phase budget; target {target} "
                         f"({'reached early' if rate * budget >= target * data else 'not reached in time'})")
        super()._emit(record)
        if done:
            raise _PhaseDone

    def _passes(self, name: str) -> float:
        start = (self.state.get("phase_start_tokens") or {}).get(name, 0)
        return (int(self.state["tokens_seen"]) - int(start)) / max(1, self._data_tokens.get(name) or 1)

    def _periodic_eval(self, source, phase):
        rec = {"phase": phase.name, "step": self.state["phase_step"], "event": "data_passes",
               "passes": round(self._passes(phase.name), 4)}
        if isinstance(source, DocSFTSource):
            rec.update(source.seen())
        self._emit(rec)
        super()._periodic_eval(source, phase)

    # ---- translation ---------------------------------------------------------
    def translate(self, sources: list[str], max_new_tokens: int | None = None,
                  direction: str = "en_zh") -> list[str]:
        """Greedy translation with the doc-mix prompt for ``direction``.

        The output limit scales with the source (zh output ≲ 1.6× en tokens,
        en ≲ 2× zh tokens) and is capped by the context left after the prompt.
        """
        from chronos_translate.corpus.shards import SFT_PROMPTS
        tok = self._tokenizer()
        end = tok.token_to_id("<|im_end|>")
        ratio = 1.6 if direction == "en_zh" else 2.0
        self.model.eval()
        out = []
        try:
            for src in sources:
                prompt = f"<|im_start|>user\n{SFT_PROMPTS[direction]}{src}<|im_end|>\n<|im_start|>assistant\n"
                ids = tok.encode(prompt, add_special_tokens=False).ids
                src_len = len(tok.encode(src, add_special_tokens=False).ids)
                room = self.ctx_len - len(ids)
                limit = min(room, int(ratio * src_len) + 16)
                if max_new_tokens:
                    limit = min(limit, int(max_new_tokens))
                out.append(tok.decode(self.generate(ids, max(1, limit), (end,)), skip_special_tokens=True)
                           if room > 0 else "")
        finally:
            self.model.train()
        return out

    def _score(self, hyps, refs, direction: str) -> tuple[float, float]:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh" if direction == "en_zh" else "13a")
        return round(bleu.score, 2), round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)

    def _fits(self, src: str, ref: str, direction: str) -> bool:
        from chronos_translate.corpus.shards import SFT_PROMPTS
        tok = self._tokenizer()
        n = len(tok.encode(SFT_PROMPTS[direction] + src, add_special_tokens=False).ids)
        return n + len(tok.encode(ref, add_special_tokens=False).ids) + 12 <= self.ctx_len

    def translation_eval(self, n: int) -> dict:
        """Both directions on FLORES+ devtest sentences and on whole documents.

        Documents are translated in one pass each (source segments joined the
        way the doc-mix records join them) and scored at document level;
        ``*_len_ratio`` = hypothesis / reference characters flags early stops
        and run-on output on long inputs.
        """
        from chronos_translate.corpus.evalsets import load_eval_documents, load_eval_sets
        raw = (self.spec.extra or {}).get("raw_root")
        if not raw:
            return {}
        t0 = time.monotonic()
        out, samples = {}, []
        pairs = load_eval_sets(raw, ["flores_devtest"]).get("flores_devtest", [])[:n]
        for direction in DIR_NAMES:
            if not pairs:
                break
            srcs = [en if direction == "en_zh" else zh for en, zh in pairs]
            refs = [zh if direction == "en_zh" else en for en, zh in pairs]
            hyps = self.translate(srcs, direction=direction)
            bleu, chrf = self._score(hyps, refs, direction)
            if direction == "en_zh":
                out.update({"flores_devtest_n": len(pairs), "bleu_zh": bleu, "chrf": chrf})
            else:
                out.update({"sent_zh_en_bleu": bleu, "sent_zh_en_chrf": chrf})
            samples += [{"direction": direction, "set": "flores_devtest", "src": s, "ref": r, "hyp": h}
                        for s, r, h in list(zip(srcs, refs, hyps))[:3]]
        extra = self.spec.extra or {}
        final = n >= int(self.spec.final_eval_samples or 0) > 0
        # Periodic evals use a fixed small subset; the final eval every document
        # that fits the context unless the recipe caps it.
        n_docs = int(extra.get("final_doc_eval_docs", 1 << 30) if final else extra.get("doc_eval_docs", 16))
        docs = load_eval_documents(raw, ["flores_devtest_docs", "wmt24pp_docs"])
        plan = (("doc", "flores_devtest_docs", DIR_NAMES), ("wmt24pp_doc", "wmt24pp_docs", ("en_zh",)))
        for prefix, name, directions in plan:
            for direction in directions:
                srcs, refs = [], []
                for d in docs.get(name, []):
                    en = " ".join(e for e, _z in d["segments"])
                    zh = "".join(z for _e, z in d["segments"])
                    src, ref = (en, zh) if direction == "en_zh" else (zh, en)
                    if self._fits(src, ref, direction):
                        srcs.append(src); refs.append(ref)
                    if len(srcs) >= n_docs:
                        break
                if not srcs:
                    continue
                hyps = self.translate(srcs, direction=direction)
                bleu, chrf = self._score(hyps, refs, direction)
                ratio = sum(map(len, hyps)) / max(1, sum(map(len, refs)))
                out.update({f"{prefix}_{direction}_bleu": bleu, f"{prefix}_{direction}_chrf": chrf,
                            f"{prefix}_{direction}_len_ratio": round(ratio, 3), f"{prefix}_{direction}_n": len(srcs)})
                samples += [{"direction": direction, "set": name, "src": s, "ref": r, "hyp": h}
                            for s, r, h in list(zip(srcs, refs, hyps))[:2]]
        self.write_samples(samples)
        if out:
            out["eval_s"] = round(time.monotonic() - t0, 1)
        return out


def load_spec(path: str, budget_hours: float | None = None) -> TranslationRunSpec:
    """Read a recipe; relative paths (and ``extra.raw_root``) resolve against its folder."""
    return _load_spec(path, budget_hours, spec_cls=TranslationRunSpec, extra_path_keys=EXTRA_PATH_KEYS)


def _smoke(spec, args) -> None:
    if args.max_steps is not None:                 # smoke: keep the closing eval to a few documents
        spec.extra = dict(spec.extra or {}, doc_eval_docs=2)


def main(argv=None) -> int:
    return _longrun_main(argv, runner_cls=DocLongRun, spec_cls=TranslationRunSpec, extra_path_keys=EXTRA_PATH_KEYS,
                         description="Time-budgeted Chronos MLX run on doc-mix shards", prepare=_smoke)


if __name__ == "__main__":
    raise SystemExit(main())
