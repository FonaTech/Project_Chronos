"""Chunked cross-entropy for MLX: the LM-head loss without ``[B, S, V]`` logits.

For long contexts the FP32 logits dominate peak memory (S=65536, V=16384:
4 GiB per row).  ``chunked_lm_loss`` walks the flattened hidden states in
``chunk`` rows.  Each chunk computes ``hidden @ W.T`` in FP32, its
logsumexp and the label logit, inside ``mx.checkpoint``, so the backward
pass recomputes that chunk's logits instead of keeping them.  Peak logit
memory is one chunk (``chunk * V * 4`` bytes).

The value equals the mean cross-entropy over labels that are not
``ignore_index``, in FP32, as the full path with ``logits_dtype=mx.float32``.
"""
import mlx.core as mx

DEFAULT_CHUNK = 4096


def _chunk_sum(h, w, lab, keep):
    """Summed CE of one chunk: rows ``h [n, H]``, head ``w [V, H]``, FP32."""
    logits = h.astype(mx.float32) @ w.astype(mx.float32).T
    lse = mx.logsumexp(logits, axis=-1)
    picked = mx.take_along_axis(logits, mx.stop_gradient(lab)[:, None], axis=-1)[:, 0]
    return ((lse - picked) * mx.stop_gradient(keep)).sum()


_chunk_sum_ckpt = mx.checkpoint(_chunk_sum)


def chunked_ce_sum(hidden, weight, labels, ignore_index=-100, chunk=DEFAULT_CHUNK):
    """``(sum of CE, number of supervised labels)`` over ``hidden [..., H]`` / ``labels [...]``."""
    H = hidden.shape[-1]
    h = hidden.reshape(-1, H)
    lb = labels.reshape(-1)
    keep = (lb != ignore_index).astype(mx.float32)
    safe = mx.where(lb != ignore_index, lb, 0).astype(mx.int32)
    n = h.shape[0]
    step = max(1, int(chunk))
    total = mx.zeros((), dtype=mx.float32)
    for start in range(0, n, step):
        end = min(n, start + step)
        rows = h[start:end]
        if start:
            # Serialize the chunks: chunk k+1 starts after chunk k's sum exists,
            # so the scheduler never holds several chunks' logits at once.
            rows = mx.depends(rows, total)
        total = total + _chunk_sum_ckpt(rows, weight, safe[start:end], keep[start:end])
    return total, keep.sum()


def chunked_lm_loss(hidden, weight, labels, ignore_index=-100, chunk=DEFAULT_CHUNK):
    """Mean CE of labels aligned with ``hidden`` (no shift), chunked."""
    total, count = chunked_ce_sum(hidden, weight, labels, ignore_index, chunk)
    return total / mx.maximum(count, 1.0)


def chunked_shifted_lm_loss(hidden, weight, labels, ignore_index=-100, chunk=DEFAULT_CHUNK):
    """Mean CE where position t predicts ``labels[:, t + 1]`` (the causal shift)."""
    return chunked_lm_loss(hidden[:, :-1, :], weight, labels[:, 1:], ignore_index, chunk)


def ce_chunk_for(config, seq_rows: int) -> int:
    """Rows per CE chunk from ``config.ce_chunk_size``; 0 = no chunking (full logits)."""
    size = int(getattr(config, "ce_chunk_size", 0) or 0)
    return 0 if size <= 0 or size >= int(seq_rows) else size
