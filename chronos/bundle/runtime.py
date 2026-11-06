"""Load a bundle's model and translate with it (Torch: CPU/MPS/CUDA, or MLX).

Pretrain bundles use the ``<|en|>src<|zh|>`` (or reverse) format they were
trained on; SFT bundles use the chat prompt.  Decoding is greedy with a KV
cache; ``repetition_penalty`` > 1 suppresses the loops a pretrain-only model
falls into on long inputs.  ``speculative=True`` uses recurrent-depth
self-speculative decoding (same greedy output, fewer full-depth forwards);
it needs ``repetition_penalty == 1`` because the penalty changes the target
distribution position by position.
"""
from __future__ import annotations

import os
import time

STOP_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|en|>", "<|zh|>")
LANG_TAGS = {"en_zh": ("<|en|>", "<|zh|>"), "zh_en": ("<|zh|>", "<|en|>")}


class BundleModel:
    """Weights + tokenizer of one bundle, ready for greedy translation."""

    def __init__(self, bundle_dir: str, *, device: str = "cpu", dtype: str = "float32", backend: str = "torch"):
        import torch
        from tokenizers import Tokenizer
        from chronos.model.checkpoint import chronos_config_from_checkpoint, load_checkpoint_into_model
        from chronos.model.model_chronos import ChronosForCausalLM
        from chronos.precision import place_torch_model
        from .builder import load_bundle_manifest
        self.bundle_dir = os.path.abspath(bundle_dir)
        self.manifest = load_bundle_manifest(self.bundle_dir)
        self.checkpoint = os.path.join(self.bundle_dir, self.manifest["model"]["checkpoint"])
        tok_dir = os.path.join(self.bundle_dir, self.manifest["tokenizer"]["dir"])
        self.tokenizer = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        self.config, _ = chronos_config_from_checkpoint(self.checkpoint)
        if self.tokenizer.get_vocab_size() > int(self.config.vocab_size):
            raise ValueError("bundle tokenizer is larger than the model vocabulary")
        model = ChronosForCausalLM(self.config)
        load_checkpoint_into_model(model, self.checkpoint, map_location="cpu")
        self.backend, self.last_stats = backend, {}
        if backend == "mlx":
            import mlx.core as mx
            from chronos.mlx.model import ChronosMLXModel
            from chronos.precision import resolve_mlx
            if device == "cpu":
                mx.set_default_device(mx.cpu)
            elif device not in ("auto", "gpu"):
                raise ValueError(f"MLX backend runs on 'gpu' or 'cpu', not {device!r}")
            self.precision = resolve_mlx(dtype, "cpu" if device == "cpu" else "gpu")
            self.model = ChronosMLXModel.from_chronos_pytorch(model.eval(), self.config,
                                                              dtype=self.precision.compute).eval()
            del model
            self.device = "cpu" if device == "cpu" else "gpu"
        elif backend == "torch":
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.precision = place_torch_model(model, device, dtype)
            self.model.eval()
            self.device = device
        else:
            raise ValueError(f"backend must be 'torch' or 'mlx', not {backend!r}")
        self.prompt_format = self.manifest.get("prompt_format", "pretrain_tags")
        tid = self.tokenizer.token_to_id
        self.stop_ids = {tid(t) for t in STOP_TOKENS if tid(t) is not None}

    # ---- prompts ---------------------------------------------------------------
    def directions(self) -> list[str]:
        return ["en_zh", "zh_en"] if self.prompt_format == "pretrain_tags" else ["en_zh"]

    def prompt_ids(self, src: str, direction: str = "en_zh") -> list[int]:
        tok = self.tokenizer
        if direction not in self.directions():
            raise ValueError(f"this bundle supports {self.directions()}, not {direction!r}")
        if self.prompt_format == "pretrain_tags":
            a, b = LANG_TAGS[direction]
            return [tok.token_to_id(a)] + tok.encode(src, add_special_tokens=False).ids + [tok.token_to_id(b)]
        from chronos.data.corpus.shards import SFT_PROMPT_EN_ZH
        text = f"<|im_start|>user\n{SFT_PROMPT_EN_ZH}{src}<|im_end|>\n<|im_start|>assistant\n"
        return tok.encode(text, add_special_tokens=False).ids

    # ---- decoding --------------------------------------------------------------
    def greedy(self, prompt: list[int], limit: int, repetition_penalty: float = 1.0, *, speculative: bool = False,
               draft_depth: int = 1, gamma: int = 4) -> list[int]:
        """Greedy continuation of ``prompt`` (stop tokens excluded), at most ``limit`` tokens."""
        if speculative and repetition_penalty != 1.0:
            raise ValueError("speculative decoding needs repetition_penalty=1.0 (the penalty changes the target "
                             "distribution at every position)")
        if speculative:
            out = self._speculative(prompt, limit, draft_depth, gamma)
        elif self.backend == "mlx":
            out = self._greedy_mlx(prompt, limit, repetition_penalty)
        else:
            out = self._greedy_torch(prompt, limit, repetition_penalty)
        for i, t in enumerate(out):
            if t in self.stop_ids:
                return out[:i]
        return out

    def _greedy_torch(self, prompt, limit, repetition_penalty):
        import torch
        from chronos.model.model_chronos import apply_repetition_penalty
        ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        seen = ids
        out: list[int] = []
        cache = None
        with torch.inference_mode():
            for _ in range(limit):
                res = self.model(ids, past_key_values=cache, use_cache=True)
                cache = res.past_key_values
                logits = res.logits[:, -1, :].float()
                if repetition_penalty != 1.0:
                    logits = apply_repetition_penalty(logits, seen, repetition_penalty)
                nxt = int(torch.argmax(logits, dim=-1).item())
                if nxt in self.stop_ids:
                    break
                out.append(nxt)
                ids = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                seen = torch.cat([seen, ids], dim=1)
        return out

    def _greedy_mlx(self, prompt, limit, repetition_penalty):
        import mlx.core as mx
        import numpy as np
        seen = np.zeros(int(self.config.vocab_size), dtype=bool)
        seen[prompt] = True
        out: list[int] = []
        logits, _, cache = self.model(mx.array([prompt], dtype=mx.int32))
        for _ in range(limit):
            row = logits[0, -1].astype(mx.float32)
            if repetition_penalty != 1.0:      # CTRL-style, same rule as the Torch helper
                p = float(repetition_penalty)
                row = mx.where(mx.array(seen), mx.where(row < 0, row * p, row / p), row)
            nxt = int(mx.argmax(row).item())
            if nxt in self.stop_ids:
                break
            out.append(nxt)
            seen[nxt] = True
            logits, _, cache = self.model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
        return out

    def _speculative(self, prompt, limit, draft_depth, gamma):
        eos = sorted(self.stop_ids)
        if self.backend == "mlx":
            from chronos.mlx.speculative import speculative_generate
            tokens, stats = speculative_generate(self.model, prompt, max_new_tokens=limit, draft_depth=draft_depth,
                                                 gamma=gamma, eos_token_id=eos)
        else:
            import torch
            from chronos.runtime.speculative import speculative_generate
            with torch.inference_mode():
                ids, stats = speculative_generate(self.model, torch.tensor([prompt], device=self.device),
                                                  max_new_tokens=limit, draft_depth=draft_depth, gamma=gamma,
                                                  eos_token_id=eos)
            tokens = ids[0].tolist()
        agg = self.last_stats.setdefault("speculative", {"proposed": 0, "accepted": 0, "verify_calls": 0,
                                                          "draft_calls": 0, "generated": 0})
        for k in agg:
            agg[k] += int(getattr(stats, k))
        return tokens[len(prompt):]

    def translate(self, sources: list[str], direction: str = "en_zh", *, max_new_tokens: int = 256,
                  repetition_penalty: float = 1.0, speculative: bool = False, draft_depth: int = 1,
                  gamma: int = 4) -> list[str]:
        out = []
        self.last_stats = {}
        for src in sources:
            src_len = len(self.tokenizer.encode(src, add_special_tokens=False).ids)
            limit = min(int(max_new_tokens), int(1.6 * src_len) + 16)   # output ≲ 1.6× source tokens
            gen = self.greedy(self.prompt_ids(src, direction), limit, repetition_penalty, speculative=speculative,
                              draft_depth=draft_depth, gamma=gamma)
            out.append(self.tokenizer.decode(gen, skip_special_tokens=True))
        spec = self.last_stats.get("speculative")
        if spec:
            spec["acceptance_rate"] = round(spec["accepted"] / spec["proposed"], 4) if spec["proposed"] else 0.0
            spec["tokens_per_verify"] = round(spec["generated"] / max(spec["verify_calls"], 1), 3)
        return out


def eval_pairs(bundle_dir: str, split: str = "dev") -> list[tuple[str, str]]:
    """FLORES+ (en, zh) pairs shipped in ``<bundle>/eval``."""
    from chronos.data.corpus.evalsets import load_eval_sets
    name = "flores_dev" if split == "dev" else "flores_devtest"
    return load_eval_sets(os.path.join(os.path.abspath(bundle_dir), "eval"), [name]).get(name, [])


def score(bm: BundleModel, pairs: list[tuple[str, str]], *, directions=None, repetition_penalty: float = 1.0,
          samples: int = 3) -> dict:
    """Corpus BLEU/chrF with sacrebleu (BLEU tokenize=zh for →zh, 13a for →en)."""
    import sacrebleu
    out, t0 = {}, time.monotonic()
    for direction in directions or bm.directions():
        si, ti = (0, 1) if direction == "en_zh" else (1, 0)
        hyps = bm.translate([p[si] for p in pairs], direction, repetition_penalty=repetition_penalty)
        refs = [p[ti] for p in pairs]
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh" if ti == 1 else "13a")
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        out[direction] = {"n": len(pairs), "bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2),
                          "samples": [{"src": p[si], "ref": p[ti], "hyp": h}
                                      for p, h in list(zip(pairs, hyps))[:samples]]}
    out["seconds"] = round(time.monotonic() - t0, 1)
    out["decode"] = {"greedy": True, "kv_cache": True, "repetition_penalty": repetition_penalty,
                     "max_new_tokens": "min(256, 1.6*src_tokens+16)", "device": bm.device,
                     "dtype": getattr(bm.precision, "as_dict", lambda: str(bm.precision))()}
    return out
