#!/usr/bin/env python3
"""English -> Chinese translation with chronos-enzh-50m-sft (research use only).

Needs the Chronos kernel from PyPI, pinned:
    pip install --pre "Project_Chronos==0.3.2a0"           # add [mlx] on Apple silicon
    python translate.py --download    # weights from the GitHub release
    python translate.py "The meeting was moved to Friday."

Boundary, enforced below: English -> Chinese only, one sentence at a time,
at most ``max_source_tokens`` tokens per sentence.  Chinese (or other
non-English) input raises ``BoundaryError``: Chinese -> English is disabled
because the SFT data held English -> Chinese instructions only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_NAME = "chronos-enzh-50m-sft.safetensors"
WEIGHTS_URL = ("https://github.com/FonaTech/Project_Chronos/releases/download/enzh-50m-sft-v1/"
        + WEIGHTS_NAME)
WEIGHTS_SHA256 = "57c59a141f3f75898268bbe2cda6d3218d514b229ad7c8e61fc02cc270fab4b7"
KERNEL_VERSION = "0.3.2a0"


class BoundaryError(ValueError):
    """The input is outside what the model was trained to do."""


_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_ABBREV = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|U\.K|No|Fig|Inc|Ltd|Co|[A-Z])"
       r"|^\s*\d{1,3})\.$")    # abbreviations, initials, list numbers ("2. Use CSS")
_SENT_END = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Z0-9])")
_CLOSERS = "\"'”’)]"


def _line_sentences(line: str) -> list[str]:
    out, start = [], 0
    for m in _SENT_END.finditer(line):
        piece = line[start:m.end()]
        if not _ABBREV.search(piece.rstrip(_CLOSERS)):
            out.append(piece.strip())
            start = m.end()
    if line[start:].strip():
        out.append(line[start:].strip())
    return out


def split_sentences(text: str) -> list[str]:
    """English sentences of ``text``; a newline always ends a sentence."""
    return [s for line in text.splitlines() for s in _line_sentences(line)]


def check_english(text: str) -> None:
    """Raise ``BoundaryError`` unless ``text`` is English."""
    latin = len(_LATIN.findall(text))
    cjk = len(_CJK.findall(text))
    other = sum(1 for c in text if c.isalpha()) - latin - cjk
    if cjk * 3 >= max(latin, 1):        # one CJK character carries about one word
        raise BoundaryError("input is Chinese/CJK text. This model translates English -> Chinese only; "
                            "Chinese -> English is disabled (no zh->en instructions in its SFT data).")
    if latin == 0 or other >= latin:
        raise BoundaryError("input is not English text. This model translates English -> Chinese only.")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(dest_dir: str = HERE, url: str = WEIGHTS_URL) -> str:
    """Fetch the weights from the GitHub release and check their SHA-256."""
    dest = os.path.join(dest_dir, WEIGHTS_NAME)
    if os.path.exists(dest) and sha256_of(dest) == WEIGHTS_SHA256:
        return dest
    part = dest + ".part"
    print(f"downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, part)
    got = sha256_of(part)
    if got != WEIGHTS_SHA256:
        os.remove(part)
        raise RuntimeError(f"SHA-256 mismatch: got {got}, expected {WEIGHTS_SHA256}")
    os.replace(part, dest)
    return dest


def _load_json(name: str, model_dir: str) -> dict:
    with open(os.path.join(model_dir, name), encoding="utf-8") as fh:
        return json.load(fh)


def _torch_model(config, weights: str):
    from safetensors.torch import load_file
    from chronos.model.model_chronos import ChronosForCausalLM
    model = ChronosForCausalLM(config)
    missing, unexpected = model.load_state_dict(load_file(weights), strict=False)
    if unexpected or set(missing) - {"lm_head.weight"}:
        raise RuntimeError(f"weights do not match config.json: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.tie_weights()  # lm_head shares the embedding (tie_word_embeddings)
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("lm_head is not tied to the embedding")
    return model.eval()


def loop_start(tokens: list[int], max_period: int = 8) -> int:
    """Start of a degenerate tail loop (kept: one copy), or -1.

    A period-p pattern counts as a loop after max(4, ceil(8 / p)) repeats, so a
    run of up to seven equal digits (digits are single tokens) is not a loop.
    """
    n = len(tokens)
    for p in range(1, max_period + 1):
        k = p * max(4, -(-8 // p))
        if n >= k and tokens[n - k:] == tokens[n - p:] * (k // p):
            return n - k + p
    return -1


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return "torch"
    return "mlx"


class Translator:
    """chronos-enzh-50m-sft with its trained prompt, decoding and input boundary."""

    def __init__(self, model_dir: str = HERE, *, weights: str | None = None, backend: str = "auto",
            device: str = "auto", verify: bool = True):
        from tokenizers import Tokenizer
        from chronos.model.config import ChronosConfig
        self.model_dir = os.path.abspath(model_dir)
        self.prompts = _load_json("prompts.json", self.model_dir)
        self.generation = _load_json("generation_config.json", self.model_dir)
        self.config = ChronosConfig(**_load_json("config.json", self.model_dir))
        self.tokenizer = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer", "tokenizer.json"))
        self.stop_ids = {self.tokenizer.token_to_id(t) for t in self.prompts["stop_tokens"]} - {None}
        weights = weights or os.path.join(self.model_dir, WEIGHTS_NAME)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"{weights} not found; run `python translate.py --download` first")
        if verify and sha256_of(weights) != WEIGHTS_SHA256:
            raise RuntimeError(f"{weights} is not the released file (SHA-256 mismatch); pass verify=False to override")
        self.backend = _resolve_backend(backend)
        model = _torch_model(self.config, weights)
        if self.backend == "mlx":
            import mlx.core as mx
            from chronos.mlx.model import ChronosMLXModel
            if device == "cpu":
                mx.set_default_device(mx.cpu)   # process-wide; MLX CPU runs in float32
            self.dtype = "float32" if device == "cpu" else "bfloat16"
            self.model = ChronosMLXModel.from_chronos_pytorch(model, self.config, dtype=self.dtype).eval()
            self.device = "cpu" if device == "cpu" else "gpu"
        elif self.backend == "torch":
            import torch
            from chronos.precision import place_torch_model
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.model, policy = place_torch_model(model, device, "float32" if device == "cpu" else "auto")
            self.dtype, self.device = policy.compute, device
        else:
            raise ValueError(f"backend must be auto, torch or mlx, not {backend!r}")

    def prompt_ids(self, sentence: str) -> list[int]:
        text = self.prompts["template"].format(source=sentence)
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def check(self, sentence: str) -> int:
        """Source token count; ``BoundaryError`` if ``sentence`` is outside the trained range."""
        check_english(sentence)
        n = len(self.tokenizer.encode(sentence, add_special_tokens=False).ids)
        limit = int(self.generation["max_source_tokens"])
        if n > limit:
            raise BoundaryError(f"sentence has {n} tokens, more than {limit}; split it into shorter sentences")
        return n

    def translate_sentence(self, sentence: str) -> str:
        """One English sentence -> Chinese (greedy, KV cache)."""
        n = self.check(sentence)
        limit = min(int(self.generation["max_new_tokens"]), int(1.6 * n) + 16)
        run = self._greedy_mlx if self.backend == "mlx" else self._greedy_torch
        self.warning = None
        return self.tokenizer.decode(run(self.prompt_ids(sentence), limit), skip_special_tokens=True).strip()

    def translate(self, text: str, split: bool | None = None) -> str:
        """English text -> Chinese, sentence by sentence; line breaks are kept."""
        split = self.generation["split_sentences"] if split is None else split
        lines, self.warnings = [], []
        for line in text.splitlines() or [text]:
            parts = _line_sentences(line) if split else ([line.strip()] if line.strip() else [])
            out = []
            for part in parts:
                out.append(self.translate_sentence(part))
                if self.warning:
                    self.warnings.append(f"{part[:60]!r}: {self.warning}")
            lines.append("".join(out))
        return "\n".join(lines)

    def _greedy_torch(self, prompt: list[int], limit: int) -> list[int]:
        import torch
        ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        out, cache = [], None
        with torch.inference_mode():
            for _ in range(limit):
                res = self.model(ids, past_key_values=cache, use_cache=True)
                cache = res.past_key_values
                nxt = int(torch.argmax(res.logits[0, -1].float()).item())
                if nxt in self.stop_ids:
                    break
                out.append(nxt)
                cut = loop_start(out)
                if cut >= 0:
                    return self._looped(out, cut)
                ids = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
        return out

    def _greedy_mlx(self, prompt: list[int], limit: int) -> list[int]:
        import mlx.core as mx
        out = []
        logits, _, cache = self.model(mx.array([prompt], dtype=mx.int32))
        for _ in range(limit):
            nxt = int(mx.argmax(logits[0, -1].astype(mx.float32)).item())
            if nxt in self.stop_ids:
                break
            out.append(nxt)
            cut = loop_start(out)
            if cut >= 0:
                return self._looped(out, cut)
            logits, _, cache = self.model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
        return out

    def _looped(self, out: list[int], cut: int) -> list[int]:
        self.warning = "repetition loop detected and cut; the translation is likely incomplete"
        return out[:cut]


def _kernel_warning() -> None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        got = version("Project_Chronos")
    except PackageNotFoundError:
        sys.exit(f'Project_Chronos is not installed: pip install --pre "Project_Chronos=={KERNEL_VERSION}"')
    if got != KERNEL_VERSION:
        print(f"warning: tested with Project_Chronos=={KERNEL_VERSION}, found {got}", file=sys.stderr)


def _items(args) -> list[str]:
    if args.text:
        return list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            return fh.read().splitlines()
    if sys.stdin.isatty():
        print("English sentence per line, Ctrl-D to end:", file=sys.stderr)
    return sys.stdin.read().splitlines()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="English -> Chinese with chronos-enzh-50m-sft (research use only)")
    ap.add_argument("text", nargs="*", help="English text; each argument is translated separately")
    ap.add_argument("--file", help="UTF-8 text file, one item per line (default: stdin)")
    ap.add_argument("--out", help="write translations here, one line per input line")
    ap.add_argument("--json", action="store_true", help="one JSON object per item")
    ap.add_argument("--direction", default="en_zh", help="en_zh only (zh_en is disabled)")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    ap.add_argument("--device", default="auto", help="torch: cpu/mps/cuda; mlx: gpu/cpu")
    ap.add_argument("--weights", help=f"path to {WEIGHTS_NAME} (default: next to this script)")
    ap.add_argument("--download", action="store_true", help="download the weights and exit")
    ap.add_argument("--no-split", action="store_true", help="treat each line as one sentence")
    args = ap.parse_args(argv)
    if args.download:
        print(download())
        return 0
    if args.direction != "en_zh":
        print("refused: only en_zh is supported. zh_en is disabled because the SFT data held "
            "English -> Chinese instructions only.", file=sys.stderr)
        return 2
    _kernel_warning()
    tr = Translator(weights=args.weights, backend=args.backend, device=args.device)
    refused, lines = 0, []
    for item in _items(args):
        t0 = time.perf_counter()
        try:
            hyp, err = tr.translate(item, split=not args.no_split), None
        except BoundaryError as exc:
            hyp, err, refused = "", str(exc), refused + 1
        row = {"source": item, "translation": hyp, "seconds": round(time.perf_counter() - t0, 3)}
        if err:
            row["refused"] = err
            print(f"refused: {err}", file=sys.stderr)
        elif tr.warnings:
            row["warnings"] = tr.warnings
            print("warning: " + "; ".join(tr.warnings), file=sys.stderr)
        lines.append(json.dumps(row, ensure_ascii=False) if args.json else hyp)
        if not args.out:
            print(lines[-1], flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return 2 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
