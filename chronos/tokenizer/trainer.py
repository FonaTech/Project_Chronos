"""Chronos BPE trainer: script-aware, quota-balanced, byte-fallback.

Algorithm (self-contained, no external tokenizer library):

1. Pre-tokenize with ``normalize.PRETOKEN_PATTERN`` and count unique
   pre-tokens per language bucket.
2. Base alphabet: characters covering ``char_coverage`` of the corpus mass
   plus 256 ``<0xNN>`` byte tokens, so encoding is total (no UNK).
3. Merge loop with incremental pair statistics: a max-heap keyed by
   language-balanced pair counts with lazy invalidation, plus an inverted
   index pair → words.  Each merge only updates words that contain the
   pair, giving O(Σ affected-word length · log P) per merge.
4. Language balance: a pair's score is Σ_bucket count_b(pair) · w_b, where
   ``w_b = (share_b ** temperature) / share_b`` re-weights buckets toward a
   tempered distribution, so the high-resource language cannot crowd out the
   other's subwords.
5. Optional minimum frequency stops degenerate merges on tail data.
"""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .normalize import pretokenize, to_model_text

BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]


@dataclass
class BPETrainerConfig:
    vocab_size: int = 16384
    special_tokens: tuple[str, ...] = ()
    char_coverage: float = 0.9999
    # Characters beyond this share of the vocabulary fall back to bytes,
    # leaving room for multi-character merges in CJK-heavy corpora.
    max_alphabet_fraction: float = 0.4
    min_pair_frequency: int = 2
    language_temperature: float = 0.5
    max_token_chars: int = 16
    # Buckets below this mass share are pooled into ``other``; no bucket
    # weight may exceed ``max_bucket_weight`` (stray scripts must not
    # dominate merge selection).
    min_bucket_share: float = 0.01
    max_bucket_weight: float = 4.0


@dataclass
class TrainedBPE:
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    alphabet: list[str]
    stats: dict = field(default_factory=dict)


def count_pretokens(texts_by_bucket: dict[str, list[str]]) -> dict[str, Counter]:
    counts = {}
    for bucket, texts in texts_by_bucket.items():
        counter = Counter()
        for text in texts:
            counter.update(pretokenize(to_model_text(text)))
        counts[bucket] = counter
    return counts


def _pool_small_buckets(counts: dict[str, Counter], min_share: float) -> dict[str, Counter]:
    mass = {b: sum(len(w) * c for w, c in ctr.items()) for b, ctr in counts.items()}
    total = sum(mass.values()) or 1
    pooled: dict[str, Counter] = {}
    for b, ctr in counts.items():
        key = b if mass[b] / total >= min_share else "other"
        pooled.setdefault(key, Counter()).update(ctr)
    return pooled


def _bucket_weights(counts: dict[str, Counter], temperature: float, max_weight: float = 4.0) -> dict[str, float]:
    mass = {b: sum(len(w) * c for w, c in ctr.items()) for b, ctr in counts.items()}
    total = sum(mass.values()) or 1
    shares = {b: m / total for b, m in mass.items() if m}
    tempered = {b: s ** temperature for b, s in shares.items()}
    norm = sum(tempered.values()) or 1
    return {b: min(max_weight, (tempered[b] / norm) / shares[b]) for b in shares}


def _alphabet(words: dict[str, float], coverage: float, limit: int | None = None) -> list[str]:
    chars = Counter()
    for word, freq in words.items():
        for ch in word:
            chars[ch] += freq
    total = sum(chars.values()) or 1
    kept, running = [], 0.0
    for ch, freq in chars.most_common():
        if running / total >= coverage or (limit is not None and len(kept) >= limit):
            break
        kept.append(ch)
        running += freq
    return sorted(kept)


def _initial_symbols(word: str, alphabet: set[str]) -> list[str]:
    symbols = []
    for ch in word:
        if ch in alphabet:
            symbols.append(ch)
        else:
            symbols.extend(BYTE_TOKENS[b] for b in ch.encode("utf-8"))
    return symbols


def _pairs(symbols: list[str]):
    return zip(symbols, symbols[1:])


def _merge_word(symbols: list[str], a: str, b: str, joined: str) -> list[str]:
    out, i, n = [], 0, len(symbols)
    while i < n:
        if i + 1 < n and symbols[i] == a and symbols[i + 1] == b:
            out.append(joined)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(texts_by_bucket: dict[str, list[str]], config: BPETrainerConfig,
              progress=None) -> TrainedBPE:
    counts = _pool_small_buckets(count_pretokens(texts_by_bucket), config.min_bucket_share)
    weights = _bucket_weights(counts, config.language_temperature, config.max_bucket_weight)
    word_freq: dict[str, float] = defaultdict(float)
    for bucket, counter in counts.items():
        w = weights.get(bucket, 0.0)
        for word, c in counter.items():
            word_freq[word] += c * w
    limit = max(0, int(config.vocab_size * config.max_alphabet_fraction))
    alphabet = _alphabet(word_freq, config.char_coverage, limit)
    alphabet_set = set(alphabet)

    vocab: dict[str, int] = {}
    for token in (*config.special_tokens, *BYTE_TOKENS, *alphabet):
        vocab.setdefault(token, len(vocab))
    if len(vocab) > config.vocab_size:
        raise ValueError(f"base vocabulary ({len(vocab)}) exceeds vocab_size {config.vocab_size}; "
                         "lower char_coverage")

    words = list(word_freq)
    freqs = [word_freq[w] for w in words]
    symbols = [_initial_symbols(w, alphabet_set) for w in words]
    pair_count: dict[tuple[str, str], float] = defaultdict(float)
    pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)
    for wi, syms in enumerate(symbols):
        for pair in _pairs(syms):
            pair_count[pair] += freqs[wi]
            pair_words[pair].add(wi)
    heap = [(-c, pair) for pair, c in pair_count.items()]
    heapq.heapify(heap)

    merges: list[tuple[str, str]] = []
    target = config.vocab_size
    while len(vocab) < target and heap:
        neg, pair = heapq.heappop(heap)
        current = pair_count.get(pair, 0.0)
        if current <= 0:
            continue
        if not math.isclose(-neg, current, rel_tol=1e-9, abs_tol=1e-9):
            heapq.heappush(heap, (-current, pair))  # stale entry: re-queue true count
            continue
        if current < config.min_pair_frequency * min(weights.values() or [1.0]):
            break
        a, b = pair
        joined = a + b
        if len(joined) > config.max_token_chars or joined in vocab:
            pair_count.pop(pair, None)
            continue
        merges.append(pair)
        vocab[joined] = len(vocab)
        touched: dict[tuple[str, str], float] = defaultdict(float)
        for wi in list(pair_words.pop(pair, ())):
            old = symbols[wi]
            new = _merge_word(old, a, b, joined)
            if len(new) == len(old):
                continue
            f = freqs[wi]
            for p in _pairs(old):
                pair_count[p] -= f
                touched[p] -= f
            for p in _pairs(new):
                pair_count[p] += f
                touched[p] += f
                pair_words[p].add(wi)
            symbols[wi] = new
        pair_count.pop(pair, None)
        for p, delta in touched.items():
            if delta > 0 and pair_count.get(p, 0.0) > 0:
                heapq.heappush(heap, (-pair_count[p], p))
        if progress and len(merges) % 500 == 0:
            progress(len(vocab), target)

    stats = {
        "bucket_weights": weights,
        "unique_pretokens": len(words),
        "alphabet_size": len(alphabet),
        "merges": len(merges),
        "final_vocab": len(vocab),
    }
    return TrainedBPE(vocab=vocab, merges=merges, alphabet=alphabet, stats=stats)
