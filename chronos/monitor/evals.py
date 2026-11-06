"""Task-agnostic evaluation view over validation/final events.

Trainers record whatever they measure as numeric keys on ``validation`` and
``final`` events: ``valid_ce`` for every run, plus task metrics such as
``probe_en_zh_bleu``, ``accuracy``, ``pass_at_1`` or ``reward_mean``.  This
module discovers them; the pages never hard-code a task or a language pair.

A key ``<subject>_<metric>`` whose last token is a known metric name is grouped
by that metric, so ``probe_en_zh_bleu`` and ``probe_zh_en_bleu`` share one
chart with series ``probe en_zh`` / ``probe zh_en``.  Everything else gets a
chart of its own.  Bookkeeping keys (sample counts, durations, token counts)
are listed as context, never charted.
"""
from __future__ import annotations

import math

# Keys that are bookkeeping, not quality metrics.
RESERVED = frozenset({"time", "event", "phase", "step", "total", "tokens_seen", "elapsed_s", "samples"})
CONTEXT_SUFFIXES = ("_n", "_s", "_h", "_tokens", "_count", "_samples", "_seconds", "_hours")
CONTEXT_KEYS = frozenset({"n", "hours", "valid_tokens", "wall_s", "wall_h"})
# Last-token metric names that group across subjects (case-insensitive).
KNOWN_METRICS = frozenset({
    "bleu", "chrf", "ter", "rouge1", "rouge2", "rougel", "meteor", "comet", "bertscore",
    "acc", "accuracy", "f1", "precision", "recall", "em", "exact", "auc",
    "ce", "loss", "ppl", "perplexity", "nll", "bpb", "bpc",
    "wer", "cer", "reward", "score", "pass1", "win", "kl",
})
PRIMARY = ("valid_ce", "valid_ppl")
# Direction of improvement, for chart captions; unknown families get none.
LOWER_IS_BETTER = frozenset({"ce", "loss", "ppl", "perplexity", "nll", "bpb", "bpc", "wer", "cer", "ter", "kl"})
HIGHER_IS_BETTER = frozenset({"bleu", "chrf", "rouge1", "rouge2", "rougel", "meteor", "comet", "bertscore", "acc",
                              "accuracy", "f1", "precision", "recall", "em", "exact", "auc", "reward", "pass1", "win"})


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_context_key(key: str) -> bool:
    return key in CONTEXT_KEYS or key.endswith(CONTEXT_SUFFIXES)


def split_key(key: str) -> tuple[str, str]:
    """Map a key to (metric family, series label).

    ``probe_en_zh_bleu`` → (``bleu``, ``probe en_zh``); ``bleu_zh`` → (``bleu``,
    ``bleu_zh``); ``accuracy`` → (``accuracy``, ``accuracy``); unknown keys
    form a family of their own.
    """
    head, sep, last = key.rpartition("_")
    if sep and head and last.lower() in KNOWN_METRICS:
        prefix, _, rest = head.partition("_")
        return last.lower(), (f"{prefix} {rest}" if rest else head)
    first, sep, _rest = key.partition("_")
    if sep and first.lower() in KNOWN_METRICS:
        return first.lower(), key
    return key, key


def eval_metrics(events: list[dict]) -> dict:
    """Discover metric keys on validation/final events.

    Returns ``{"families": [{"metric", "series": [{"key", "label"}], "direction"}],
    "keys": [...], "context": [...]}``; ``valid_ce``/``valid_ppl`` stay in
    ``keys`` but get no family of their own (the loss charts already show them).
    """
    keys: dict[str, None] = {}
    context: dict[str, None] = {}
    for e in events or []:
        if e.get("event") not in ("validation", "final"):
            continue
        for k, v in e.items():
            if k in RESERVED or not _num(v):
                continue
            (context if is_context_key(k) else keys)[k] = None
    families: dict[str, list[dict]] = {}
    for k in keys:
        if k in PRIMARY:
            continue
        metric, label = split_key(k)
        families.setdefault(metric, []).append({"key": k, "label": label})
    def direction(metric: str) -> str | None:
        return "lower" if metric in LOWER_IS_BETTER else "higher" if metric in HIGHER_IS_BETTER else None
    return {"families": [{"metric": m, "series": s, "direction": direction(m)} for m, s in families.items()],
            "keys": list(keys), "context": list(context)}


def sample_view(sample: dict) -> dict:
    """Normalise one ``samples.jsonl`` row for display, whatever the task.

    Recognised roles: input (``src``/``prompt``/``input``/``question``),
    reference (``ref``/``reference``/``target``/``answer``) and output
    (``hyp``/``output``/``prediction``/``response``/``completion``).  Every
    other scalar field becomes a tag (``direction``, ``set``, ``task``, ``score`` …).
    """
    roles = {"input": ("src", "prompt", "input", "question", "source"),
             "reference": ("ref", "reference", "target", "answer", "expected"),
             "output": ("hyp", "output", "prediction", "response", "completion", "generated")}
    used = {"time", "phase", "step"}
    out = {"time": sample.get("time"), "phase": sample.get("phase"), "step": sample.get("step")}
    for role, names in roles.items():
        for name in names:
            if isinstance(sample.get(name), str):
                out[role] = sample[name]
                used.add(name)
                break
    out["tags"] = {k: v for k, v in sample.items()
                   if k not in used and isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    return out
