"""Pair cleaning: de-segmentation, detokenization, normalization and filters.

Every rule is deterministic and recorded as a reject reason so the corpus
report can account for every dropped pair.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from chronos.tokenizer.normalize import normalize_text

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN = re.compile(r"[A-Za-z]")
_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_HTML = re.compile(r"</?[a-zA-Z][^>]{0,80}>|&(?:amp|lt|gt|quot|nbsp|#\d+);")
_REPEAT = re.compile(r"(.)\1{9,}")

# Punctuation policy for Chinese text: full-width sentence punctuation after
# Han characters, matching standard simplified-Chinese typesetting.
_ZH_PUNCT = {",": "，", ";": "；", ":": "：", "?": "？", "!": "！", "(": "（", ")": "）"}
_H = r"\u3400-\u9fff\u3000-\u303f\uff00-\uffef"
# ASCII punctuation touching Han text becomes full-width; ``1,000``/``3:30``
# inside Latin spans stay ASCII.
_ZH_PUNCT_RE = re.compile(
    rf"(?<=[{_H}])([,;:?!])|([,;:?!])(?=[{_H}])"
    rf"|(?<=[{_H}])([(])|([(])(?=[{_H}])|(?<=[{_H}])([)])|([)])(?=[{_H}])"
)

# Segmenter padding around full-width punctuation, and Moses ``3 : 30``.
_FW_GAP_RE = re.compile(r"\s*([\uff08\uff09\uff0c\u3002\uff1b\uff1a\uff1f\uff01\u3001])\s*")
_EN_TIME_RE = re.compile(r"(\d)\s*([:.,])\s*(?=\d)")


@dataclass
class CleanConfig:
    unsegment_zh: bool = False         # spaces between CJK chars come from segmentation
    detokenize_en: bool = False        # Moses-tokenized English (`` don 't`` → ``don't``)
    zh_fullwidth_punct: bool = True
    min_chars: int = 1
    max_chars: int = 1024
    max_len_ratio: float = 3.0         # char-length ratio after a script-aware scale
    min_han_ratio: float = 0.3
    min_latin_ratio: float = 0.4


class PairCleaner:
    def __init__(self, config: CleanConfig):
        self.config = config
        self._detok = None
        if config.detokenize_en:
            from sacremoses import MosesDetokenizer
            self._detok = MosesDetokenizer(lang="en")

    def clean(self, zh: str, en: str) -> tuple[str, str] | str:
        """Return the cleaned pair, or a reject-reason string."""
        cfg = self.config
        zh = normalize_text(zh, unsegment_cjk=cfg.unsegment_zh)
        en = normalize_text(en)
        if self._detok is not None:
            en = self._detok.detokenize(en.split())
            if en[:1].islower():
                en = en[0].upper() + en[1:]
        if cfg.zh_fullwidth_punct:
            convert = lambda m: _ZH_PUNCT[next(g for g in m.groups() if g)]
            # Two passes: a conversion can create the Han-side context for its
            # neighbour, e.g. ``Blue),`` → ``Blue)，`` → ``Blue）,``.
            zh = _ZH_PUNCT_RE.sub(convert, _ZH_PUNCT_RE.sub(convert, zh))
            zh = _FW_GAP_RE.sub(r"\1", zh)
        if self._detok is not None:
            en = _EN_TIME_RE.sub(r"\1\2", en)
        if len(zh) < cfg.min_chars or len(en) < cfg.min_chars:
            return "empty"
        if len(zh) > cfg.max_chars or len(en) > cfg.max_chars:
            return "too_long"
        if _URL.search(zh) or _URL.search(en) or _HTML.search(zh) or _HTML.search(en):
            return "markup"
        if _REPEAT.search(zh) or _REPEAT.search(en):
            return "repetition"
        han = len(_HAN.findall(zh))
        if han / max(len(zh.replace(" ", "")), 1) < cfg.min_han_ratio:
            return "zh_script"
        latin = len(_LATIN.findall(en))
        if latin / max(len(en.replace(" ", "")), 1) < cfg.min_latin_ratio:
            return "en_script"
        # One Han character carries roughly as much content as 3-4 English
        # characters; compare on that scale.
        zh_len, en_len = len(zh) * 3.5, len(en)
        ratio = max(zh_len, en_len) / max(min(zh_len, en_len), 1)
        if ratio > cfg.max_len_ratio and max(len(zh), len(en)) > 8:
            return "length_ratio"
        if zh == en:
            return "copy"
        return zh, en


def pair_key(zh: str, en: str) -> str:
    """Normalization used for exact deduplication and leakage checks."""
    def squash(s: str) -> str:
        return re.sub(r"[\s\W_]+", "", s.lower())
    return squash(zh) + "\x1f" + squash(en)


def side_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())
