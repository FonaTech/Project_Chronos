"""Chronos tokenizer runtime: encoder/decoder plus HF-compatible export.

The pure-Python encoder is the reference implementation of the trained
merges.  ``save`` also writes a ``tokenizer.json`` (HF ``tokenizers`` BPE with
byte fallback and the same pre-tokenizer) so training/inference code that
expects ``AutoTokenizer`` works unchanged; tests assert both agree.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache

from .normalize import PRETOKEN_PATTERN, SPACE, from_model_text, pretokenize, to_model_text
from .trainer import BYTE_TOKENS, TrainedBPE

_CONTROL_PATTERN = r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b\u200c\u200e\u200f\ufeff]"
_CONTROL_RE = re.compile(_CONTROL_PATTERN)
_BYTE_RE = re.compile(r"^<0x([0-9A-F]{2})>$")
_BYTE_SPLIT_RE = re.compile(r"(<0x[0-9A-F]{2}>)")

# Plain turns render exactly as before (``<|im_start|>role\ncontent<|im_end|>\n``).
# Tools: the definitions go into one system block, an assistant turn's
# ``tool_calls`` follow its text as ``<tool_call>{json}</tool_call>`` lines, and
# ``tool`` turns carry the results.  ``chronos.data.chat_records.render_chat``
# produces the same text in Python (with per-turn spans for loss masks).
CHAT_TEMPLATE = (
    "{% if tools %}{{ '<|im_start|>system\n' }}"
    "{% if messages and messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n\n' }}{% endif %}"
    "{{ '<tools>\n' }}{% for tool in tools %}{{ tool | tojson + '\n' }}{% endfor %}"
    "{{ '</tools><|im_end|>\n' }}{% endif %}"
    "{% for message in messages %}"
    "{% if not (tools and loop.first and message['role'] == 'system') %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + (message['content'] or '') }}"
    "{% if message['role'] == 'assistant' and message.get('tool_calls') %}"
    "{% set text = message['content'] or '' %}"
    "{% for call in message['tool_calls'] %}"
    "{% set fn = call['function'] if call.get('function') else call %}"
    "{% if text or not loop.first %}{{ '\n' }}{% endif %}"
    "{{ '<tool_call>' + {'name': fn['name'], 'arguments': fn['arguments']} | tojson + '</tool_call>' }}"
    "{% endfor %}{% endif %}"
    "{{ '<|im_end|>' + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

#: Ids 0–2 are fixed (EOS / chat start / chat end); the rest of the 64
#: control slots are ``<|reserved_i|>`` until a task claims them.
CORE_SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>")
SPECIAL_SLOTS = 64


def special_tokens_with(extra=(), *, slots: int = SPECIAL_SLOTS) -> tuple[str, ...]:
    """Core control tokens, then ``extra`` task tokens, padded with reserved slots.

    ``extra`` tokens go right after ``<|im_end|>`` (ids 3, 4, …) in the given
    order; a token already in the core set keeps its core position.
    """
    head = list(CORE_SPECIAL_TOKENS[:3])
    for tok in extra:
        if tok not in head:
            head.append(tok)
    for tok in CORE_SPECIAL_TOKENS[3:]:
        if tok not in head:
            head.append(tok)
    if len(head) > slots:
        raise ValueError(f"{len(head)} special tokens exceed the {slots} control slots")
    return tuple(head) + tuple(f"<|reserved_{i}|>" for i in range(slots - len(head)))


DEFAULT_SPECIAL_TOKENS = special_tokens_with()


def tokenizer_normalize(text: str) -> str:
    """Tokenizer-time normalization: NFC + control removal only.

    Whitespace collapsing, de-segmentation and punctuation policy belong to
    corpus cleaning; doing them here would make ``encode(a + b)`` differ from
    ``encode(a) + encode(b)`` around chat-template boundaries.
    """
    import unicodedata
    return _CONTROL_RE.sub("", unicodedata.normalize("NFC", text))


class ChronosTokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]],
                 special_tokens: tuple[str, ...], metadata: dict | None = None):
        self.vocab = dict(vocab)
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: r for r, pair in enumerate(self.merges)}
        self.special_tokens = tuple(special_tokens)
        self.metadata = dict(metadata or {})
        self._byte_ids = [self.vocab[t] for t in BYTE_TOKENS]
        self._byte_of_id = {i: b for b, i in enumerate(self._byte_ids)}
        escaped = sorted((re.escape(t) for t in self.special_tokens), key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(escaped) + ")") if escaped else None
        self._encode_word = lru_cache(maxsize=1 << 18)(self._encode_word_uncached)

    @classmethod
    def from_trained(cls, trained: TrainedBPE, special_tokens, metadata=None):
        return cls(trained.vocab, trained.merges, tuple(special_tokens), metadata)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_id(self, token: str) -> int:
        return self.vocab[token]

    def _encode_word_uncached(self, word: str) -> tuple[int, ...]:
        symbols: list[str] = []
        for ch in word:
            if ch in self.vocab:
                symbols.append(ch)
            else:
                symbols.extend(BYTE_TOKENS[b] for b in ch.encode("utf-8"))
        while len(symbols) > 1:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
                break
            a, b = symbols[best], symbols[best + 1]
            merged, out, i = a + b, [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return tuple(self.vocab[s] for s in symbols)

    # ---- encode / decode -------------------------------------------------
    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        text = tokenizer_normalize(text)
        pieces = self._special_re.split(text) if (allow_special and self._special_re) else [text]
        ids: list[int] = []
        for piece in pieces:
            if not piece:
                continue
            if allow_special and piece in self.special_tokens:
                ids.append(self.vocab[piece])
                continue
            for word in pretokenize(to_model_text(piece)):
                ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids, *, skip_special: bool = False) -> str:
        # Merged tokens may contain several ``<0xNN>`` pieces (multi-byte
        # characters outside the alphabet); split every token back into raw
        # bytes and text, then decode the byte runs as UTF-8.
        data = bytearray()
        for i in ids:
            token = self.id_to_token.get(int(i), "")
            if skip_special and token in self.special_tokens:
                continue
            for piece in _BYTE_SPLIT_RE.split(token):
                if not piece:
                    continue
                match = _BYTE_RE.match(piece)
                data.extend(bytes([int(match.group(1), 16)]) if match else piece.encode("utf-8"))
        return from_model_text(data.decode("utf-8", errors="replace"))

    # ---- persistence -----------------------------------------------------
    def fingerprint(self) -> str:
        payload = json.dumps({"vocab": self.vocab, "merges": self.merges,
                              "special": self.special_tokens}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def hf_tokenizer_json(self) -> dict:
        added = [{"id": self.vocab[t], "content": t, "single_word": False, "lstrip": False,
                  "rstrip": False, "normalized": False, "special": True} for t in self.special_tokens]
        return {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added,
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "NFC"},
                {"type": "Replace", "pattern": {"Regex": _CONTROL_PATTERN}, "content": ""},
                {"type": "Replace", "pattern": {"String": " "}, "content": SPACE},
            ]},
            "pre_tokenizer": {"type": "Split", "pattern": {"Regex": PRETOKEN_PATTERN},
                              "behavior": "Isolated", "invert": False},
            "post_processor": None,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": SPACE}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ]},
            "model": {"type": "BPE", "dropout": None, "unk_token": None,
                      "continuing_subword_prefix": None, "end_of_word_suffix": None,
                      "fuse_unk": False, "byte_fallback": True, "ignore_merges": False,
                      "vocab": self.vocab, "merges": [[a, b] for a, b in self.merges]},
        }

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        meta = dict(self.metadata)
        meta.update(vocab_size=self.vocab_size, fingerprint=self.fingerprint(),
                    special_tokens=list(self.special_tokens), algorithm="chronos-bpe-v1")
        with open(os.path.join(directory, "chronos_tokenizer.json"), "w", encoding="utf-8") as fh:
            json.dump({"vocab": self.vocab, "merges": self.merges,
                       "special_tokens": list(self.special_tokens), "metadata": meta},
                      fh, ensure_ascii=False)
        with open(os.path.join(directory, "tokenizer.json"), "w", encoding="utf-8") as fh:
            json.dump(self.hf_tokenizer_json(), fh, ensure_ascii=False)
        config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<|im_start|>", "eos_token": "<|im_end|>",
            "unk_token": None, "pad_token": "<|endoftext|>",
            "model_max_length": 32768, "clean_up_tokenization_spaces": False,
            "chat_template": CHAT_TEMPLATE, "add_bos_token": False, "add_eos_token": False,
        }
        with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(directory, "special_tokens_map.json"), "w", encoding="utf-8") as fh:
            json.dump({"bos_token": "<|im_start|>", "eos_token": "<|im_end|>",
                       "pad_token": "<|endoftext|>"}, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(directory, "tokenizer_report.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        return directory

    @classmethod
    def load(cls, directory: str) -> "ChronosTokenizer":
        with open(os.path.join(directory, "chronos_tokenizer.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data["vocab"], [tuple(m) for m in data["merges"]],
                   tuple(data["special_tokens"]), data.get("metadata"))
