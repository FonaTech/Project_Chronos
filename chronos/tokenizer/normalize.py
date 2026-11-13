"""Text normalization and script-aware pre-tokenization for Chronos tokenizers.

The same pattern is used by the trainer, the pure-Python encoder and the
exported ``tokenizer.json`` (HF ``Split`` pre-tokenizer), so all three agree
token for token.
"""
from __future__ import annotations

import re
import unicodedata

import regex

SPACE = "\u2581"  # marks a preceding space, SentencePiece-style.

# Pre-token classes, in priority order:
#   1. runs of Han ideographs (BPE learns multi-character words inside them)
#   2. an optional ▁ followed by letters/marks of any non-Han script
#   3. a single digit (numbers are always split digit by digit)
#   4. an optional ▁ followed by one punctuation/symbol run
#   5. leftover ▁ runs and any other whitespace
PRETOKEN_PATTERN = (
    r"\p{Han}+"
    r"|" + SPACE + r"?[^\s" + SPACE + r"\p{Han}\p{P}\p{S}\p{N}]+"
    r"|\p{N}"
    r"|" + SPACE + r"?[\p{P}\p{S}]+"
    r"|" + SPACE + r"+"
    r"|\s+"
)
_PRETOKEN_RE = regex.compile(PRETOKEN_PATTERN)

_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u200b\u200c\u200e\u200f\ufeff]")
_SPACES_RE = re.compile(r"[ \t\u00a0\u3000]+")
# Spaces touching a CJK character, or padding punctuation, come from word
# segmentation; spaces between Latin tokens (``New Day Blue``) are kept.
_CJK = r"\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef"
_CJK_GAP_RE = re.compile(rf"(?<=[{_CJK}])\s+|\s+(?=[{_CJK}])")
_PUNCT_GAP_RE = re.compile(r"\s+(?=[,.;:!?%)\]}])|(?<=[(\[{])\s+")


def normalize_text(text: str, *, form: str = "NFKC", unsegment_cjk: bool = False) -> str:
    """NFKC, control-character removal, whitespace collapsing.

    ``unsegment_cjk`` removes spaces between CJK characters (for corpora that
    ship word-segmented Chinese).
    """
    if form:
        text = unicodedata.normalize(form, text)
    text = _CONTROL_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text).strip()
    if unsegment_cjk:
        text = _CJK_GAP_RE.sub("", text)
        text = _PUNCT_GAP_RE.sub("", text)
    return text


def to_model_text(text: str) -> str:
    """Make spaces visible: ``a b`` → ``a▁b`` (no leading marker)."""
    return text.replace(" ", SPACE)


def from_model_text(text: str) -> str:
    return text.replace(SPACE, " ")


def pretokenize(model_text: str) -> list[str]:
    return _PRETOKEN_RE.findall(model_text)


def looks_word_segmented_cjk(samples: list[str], threshold: float = 0.15) -> bool:
    """Heuristic detector: share of CJK-CJK boundaries separated by a space."""
    gaps = joins = 0
    pair = re.compile(rf"[{_CJK}](\s?)(?=[{_CJK}])")
    for text in samples:
        for match in pair.finditer(text):
            if match.group(1):
                gaps += 1
            else:
                joins += 1
    total = gaps + joins
    return total > 0 and gaps / total >= threshold
