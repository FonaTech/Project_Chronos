"""en↔zh translation run: the kernel ``LongRun`` plus translation metrics.

* every SFT validation: greedy en→zh on a FLORES+ devtest subset
  (``bleu_samples``), BLEU (``tokenize=zh``) and chrF;
* every pretrain validation: the ``<|en|>…<|zh|>`` probe on FLORES+ dev in
  both directions (``probe_samples``, keys ``probe_{en_zh,zh_en}_{bleu,chrf}``);
* end of run: en→zh on ``final_eval_samples`` FLORES+ devtest sentences.

FLORES+ is read from ``extra.raw_root`` (a ``chronos-translate corpus download``
tree).  The recipe format is the kernel's ``RunSpec`` plus the three sample
counts above, so 0.3.1a1 recipes run unchanged::

    chronos-translate run --spec demos/translation/recipes/enzh_50m_72h.json
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from chronos.mlx.training.longrun import LongRun, RunSpec
from chronos.mlx.training.longrun import main as _longrun_main

#: ``extra`` keys holding paths (resolved against the recipe's folder).
EXTRA_PATH_KEYS = ("raw_root",)


@dataclass
class TranslationRunSpec(RunSpec):
    final_eval_samples: int = 1012   # full FLORES+ devtest at the end (0 = skip)
    export_prefix: str = "chronos_enzh"
    bleu_samples: int = 200          # periodic FLORES+ devtest subset during SFT
    # Greedy FLORES+ dev probe at each pretrain validation (``<|en|>…<|zh|>``
    # format, both directions); samples go to ``samples.jsonl``.  0 = off.
    probe_samples: int = 16


class TranslationLongRun(LongRun):
    def pinned_modules(self) -> tuple[str, ...]:
        return super().pinned_modules() + ("sacrebleu", "chronos_translate.corpus.evalsets",
                                           "chronos_translate.corpus.shards", "chronos_translate.probe")

    def eval_samples_for(self, phase_name: str, *, final: bool) -> int:
        if final:
            return int(self.spec.final_eval_samples or 0)
        if phase_name == "sft":
            return int(getattr(self.spec, "bleu_samples", 0) or 0)
        if phase_name == "pretrain":
            return int(getattr(self.spec, "probe_samples", 0) or 0)
        return super().eval_samples_for(phase_name, final=final)

    def task_eval(self, phase_name: str, n: int, *, final: bool = False) -> dict:
        if final or phase_name != "pretrain":
            return self.translation_eval(n)
        return self._pretrain_probe(n)

    def _pretrain_probe(self, n: int) -> dict:
        from .probe import translation_probe
        probe = translation_probe(asdict(self.spec), self.model, "pretrain", n)
        out, rows = {}, []
        for direction in ("en_zh", "zh_en"):
            res = probe.get(direction)
            if not isinstance(res, dict):
                continue
            out[f"probe_{direction}_bleu"], out[f"probe_{direction}_chrf"] = res["bleu"], res["chrf"]
            rows += [{"direction": direction, "set": probe.get("probe_set"), **sample}
                     for sample in res.get("samples", [])[:3]]
        self.write_samples(rows)
        if out:
            out["probe_n"], out["probe_s"] = n, probe.get("probe_s")
        return out

    # ---- translation quality (greedy, KV cache, MLX GPU) ----------------------
    def translate(self, sources: list[str], max_new_tokens: int = 160) -> list[str]:
        from .corpus.shards import SFT_PROMPT_EN_ZH
        tok = self._tokenizer()
        end = tok.token_to_id("<|im_end|>")
        self.model.eval()
        out = []
        for src in sources:
            prompt = f"<|im_start|>user\n{SFT_PROMPT_EN_ZH}{src}<|im_end|>\n<|im_start|>assistant\n"
            ids = tok.encode(prompt, add_special_tokens=False).ids
            src_len = len(tok.encode(src, add_special_tokens=False).ids)
            limit = min(max_new_tokens, int(1.6 * src_len) + 16)   # zh output ≲ 1.6× en tokens
            out.append(tok.decode(self.generate(ids, limit, (end,)), skip_special_tokens=True))
        self.model.train()
        return out

    def translation_eval(self, n: int) -> dict:
        import sacrebleu

        from .corpus.evalsets import load_eval_sets
        raw = self.spec.extra.get("raw_root")
        if not raw:
            return {}
        pairs = load_eval_sets(raw, ["flores_devtest"]).get("flores_devtest", [])[:n]
        if not pairs:
            return {}
        t0 = time.monotonic()
        hyps = self.translate([en for en, _zh in pairs])
        refs = [zh for _en, zh in pairs]
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        self.write_samples([{"direction": "en_zh", "set": "flores_devtest", "src": en, "ref": zh, "hyp": hyp}
                            for (en, zh), hyp in list(zip(pairs, hyps))[:5]])
        return {"flores_devtest_n": len(pairs), "bleu_zh": round(bleu.score, 2), "chrf": round(chrf.score, 2),
                "eval_s": round(time.monotonic() - t0, 1)}


def main(argv=None) -> int:
    return _longrun_main(argv, runner_cls=TranslationLongRun, spec_cls=TranslationRunSpec,
                         extra_path_keys=EXTRA_PATH_KEYS, description="Time-budgeted en↔zh Chronos MLX run")


if __name__ == "__main__":
    raise SystemExit(main())
