"""Blockwise causal attention for long sequences (MLX): no ``[S, T]`` mask.

The fused SDPA kernel with an explicit boolean mask needs the ``[S, T]``
mask and, in its VJP, ``[nH, S, T]`` scores, so memory grows as S**2
(measured: one SWA layer at S=32768 peaked at 40 GiB extra).

``blockwise_attention`` splits the queries into blocks of ``block`` rows.
Block b only reads the keys it can see: ``[b*block - window + 1, (b+1)*block)``
for a sliding window, ``[0, (b+1)*block)`` for full causal attention.  Each
block runs the fused kernel with a small ``[block, span]`` mask, under
``mx.checkpoint`` so its scores are recomputed in the backward pass.  A
sliding-window layer then costs O(S * (window + block)), and the full
causal layer stores O(block * S) at a time instead of O(S**2).

``segments`` ``[B, S]`` (document ids, non-decreasing along a row) adds
document masking: a query attends only to keys of its own document.

Exactness: every query sees exactly the keys of the reference mask, and
softmax runs over that set inside the fused kernel, so the output equals
the full-mask path up to floating-point summation order.
"""
import mlx.core as mx


def _block_mask(q0, q1, k0, k1, window, seg_q, seg_k):
    """Boolean ``[(B,) 1, q1-q0, k1-k0]`` mask for absolute query rows q0..q1-1."""
    q_pos = mx.arange(q0, q1)[:, None]
    k_pos = mx.arange(k0, k1)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed = allowed & (k_pos > q_pos - int(window))
    if seg_q is None:
        return allowed
    same = seg_q[:, None, :, None] == seg_k[:, None, None, :]
    return allowed[None, None] & same


def _mask_for(q0, q1, k0, k1, window, seg_q, seg_k):
    """Array mask for one block, or ``"causal"`` for a square diagonal span
    where the plain causal rule applies (no mask array in the kernel)."""
    # MLX aligns "causal" to the bottom-right, so a block whose last query is
    # the last visible key needs no mask array whatever its key span.
    if seg_q is None and k1 == q1 and (window is None or window >= q1 - k0):
        return "causal"
    return mx.stop_gradient(_block_mask(q0, q1, k0, k1, window, seg_q, seg_k))


def _attend_block(q, k, v, mask, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


_attend_block_ckpt = mx.checkpoint(_attend_block)


def blockwise_attention(q, k, v, *, scale, window=None, block=1024, segments=None, past=0, checkpoint=True):
    """Causal (optionally windowed / document-masked) attention in query blocks.

    ``q [B, nH, S, d]``, ``k``/``v`` ``[B, nKV, T, d]`` with ``T = past + S``
    (``past`` cached keys precede the queries).  Returns ``[B, nH, S, dv]``.
    """
    S, T = q.shape[2], k.shape[2]
    block = max(1, int(block))
    run = _attend_block_ckpt if checkpoint else _attend_block
    seg = None if segments is None else mx.stop_gradient(segments)
    outs = []
    prev = None
    for start in range(0, S, block):
        end = min(S, start + block)
        qa0, qa1 = past + start, past + end
        k0 = 0 if window is None else max(0, qa0 - int(window) + 1)
        k1 = min(T, qa1)
        seg_q = None if seg is None else seg[:, qa0:qa1]
        seg_k = None if seg is None else seg[:, k0:k1]
        mask = _mask_for(qa0, qa1, k0, k1, window, seg_q, seg_k)
        qb = q[:, :, start:end]
        if prev is not None:
            qb = mx.depends(qb, prev)   # one block's scores live at a time
        prev = run(qb, k[:, :, k0:k1], v[:, :, k0:k1], mask, scale)
        outs.append(prev)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def _mla_block(qb, cb, rb, w_k, w_v, mask, scale):
    """One query block of MLA attention, K and V rebuilt from the latent slice.

    ``cb [B, t, L]`` latent, ``rb [B, t, nKV, r]`` rotated rope keys,
    ``w_k [nKV*nope, L]`` and ``w_v [nKV*d, L]`` the up-projections."""
    B, t, _ = cb.shape
    n_kv, r = rb.shape[2], rb.shape[3]
    k_nope = (cb @ w_k.T).reshape(B, t, n_kv, -1)
    v = (cb @ w_v.T).reshape(B, t, n_kv, -1)
    k = mx.concatenate([k_nope, rb], axis=-1).transpose(0, 2, 1, 3)
    return mx.fast.scaled_dot_product_attention(qb, k, v.transpose(0, 2, 1, 3), scale=scale, mask=mask)


_mla_block_ckpt = mx.checkpoint(_mla_block)


def blockwise_mla_attention(q, c_kv, k_rope, w_k, w_v, *, scale, block=1024, segments=None, past=0):
    """Full-causal MLA attention that never keeps the expanded K and V.

    ``q [B, nH, S, d]``; ``c_kv [B, T, L]``; ``k_rope [B, T, nKV, r]`` (already
    rotated); ``w_k``/``w_v`` the ``k_nope_proj`` / ``v_proj`` weights.  Every
    block runs under ``mx.checkpoint`` with the latent slice as its input, so
    the forward pass keeps ``c_kv`` (L wide) instead of K and V (nKV*d wide
    each), and the backward pass rebuilds them one block at a time.  The same
    keys and softmax set as :func:`blockwise_attention`, so the output is
    equal up to summation order.  Returns ``[B, nH, S, d]``."""
    S, T = q.shape[2], c_kv.shape[1]
    block = max(1, int(block))
    seg = None if segments is None else mx.stop_gradient(segments)
    outs, prev = [], None
    for start in range(0, S, block):
        end = min(S, start + block)
        qa0, qa1 = past + start, past + end
        k1 = min(T, qa1)
        seg_q = None if seg is None else seg[:, qa0:qa1]
        seg_k = None if seg is None else seg[:, :k1]
        mask = _mask_for(qa0, qa1, 0, k1, None, seg_q, seg_k)
        qb = q[:, :, start:end]
        if prev is not None:
            qb = mx.depends(qb, prev)
        prev = _mla_block_ckpt(qb, c_kv[:, :k1], k_rope[:, :k1], w_k, w_v, mask, scale)
        outs.append(prev)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def use_blockwise(config, S, T, training):
    """Rows per attention block from ``config.attention_block_size`` (0 = off).

    Only for multi-row queries without attention dropout, and only when the
    sequence is longer than one block (short sequences keep the exact
    single-kernel path)."""
    size = int(getattr(config, "attention_block_size", 0) or 0)
    if size <= 0 or S <= size:
        return 0
    if training and float(getattr(config, "dropout", 0.0) or 0.0) > 0.0:
        return 0
    return size
