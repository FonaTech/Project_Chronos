"""Torch kernels for the single-compute-dtype policy (CPU / CUDA / MPS).

Mirrors ``chronos/mlx/attention.py``: weights, activations and KV cache stay
in one compute dtype; FP32 is obtained inside fused kernels or at a few
small, deliberate boundaries (see ``chronos/precision.py``).

* ``RMSNorm``: FP32 statistics, one rounding.  CPU/CUDA use the fused
  ``F.rms_norm``; on MPS the fused low-precision kernel accumulates in the
  input dtype (measured bf16 rel. error 2.34e-3 vs 1.66e-3), so MPS keeps the
  explicit FP32-statistics form.
* RoPE: FP32 tables that ``model.to(dtype)`` never converts; the rotation is
  computed in FP32 and rounded once (MLX measured 1.96e-3 vs 2.48e-3 with
  BF16 tables).
* Attention: ``F.scaled_dot_product_attention`` with native GQA (no repeated
  K/V), a boolean mask and in-kernel dropout; FP32 softmax inside the kernel
  on every backend.  MPS bf16 measured rel. error 1.50e-3 vs 2.87e-3 for the
  explicit softmax path, and faster forward (prefill 5.6 vs 6.8 ms, decode
  0.21 vs 0.40 ms, B16 S512 GQA 6/2 d64).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_TORCH_VERSION = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
SDPA_NATIVE_GQA = _TORCH_VERSION >= (2, 5)


class RMSNorm(nn.Module):
    """RMSNorm with FP32 statistics and a single rounding to the input dtype.

    Parameter name (``weight``) and the ``RMSNorm`` class-name suffix are the
    checkpoint and no-decay optimizer-grouping contract.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "mps" or x.dtype == torch.float32:
            w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
            return F.rms_norm(x, (x.shape[-1],), w, self.eps)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


def rope_tables(dim: int, length: int, base: float, scaling, device) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 duplicated cos/sin tables [>=length, dim], shared by every layer."""
    from chronos.model.kernel import rope_tables as _shared
    return _shared(dim, length, base, scaling, device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, S, nH, d] (compute dtype), cos/sin [S, d] FP32 -> x.dtype, rounded once."""
    c, s = cos[:, None, :], sin[:, None, :]
    return (x * c + _rotate_half(x) * s).to(x.dtype)


def causal_allowed(S: int, T: int, device, *, past: int | None = None,
                   window: int | None = None) -> torch.Tensor:
    """Boolean [S, T] mask for queries at absolute key rows ``past..past+S-1``."""
    past = T - S if past is None else int(past)
    q_pos = torch.arange(S, device=device)[:, None] + past
    k_pos = torch.arange(T, device=device)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed = allowed & (k_pos > q_pos - int(window))
    return allowed


def key_keep_mask(attention_mask: torch.Tensor, S: int, T: int) -> torch.Tensor:
    """Caller padding mask -> boolean keep mask broadcastable to [B, 1, S, T].

    Accepts the tokenizer-style 2-D mask [B, total_len], a 3-D [B, S, T] mask
    or a 4-D mask that is either 0/1 or additive (0 / large negative).  The
    key axis is right-aligned (cached decoding, sliding-window eviction).
    """
    m = attention_mask
    if m.dim() == 4 and m.dtype != torch.bool and bool((m < 0).any()):
        m = m > -1e4
    elif m.dtype != torch.bool:
        m = m > 0.5
    if m.shape[-1] < T:
        pad = m.new_ones(*m.shape[:-1], T - m.shape[-1])
        m = torch.cat([pad, m], dim=-1)
    m = m[..., -T:]
    if m.dim() == 2:
        return m[:, None, None, :]
    if m.dim() == 3:
        return m[:, None, -S:, :]
    return m[..., -S:, :]


def build_attention_mask(S: int, T: int, device, *, past: int, window: int | None,
                         attention_mask: torch.Tensor | None):
    """Return ``(mask, is_causal)`` for :func:`attention`.

    The diagonal (each query's own key) is always kept, so left-padded query
    rows never have an empty softmax row (no NaN in any dtype).
    """
    if attention_mask is None:
        if window is None and S == T:
            return None, S > 1          # fused causal kernel (flash on CUDA)
        if window is None and S == 1:
            return None, False          # single-token decode sees every cached key
        return causal_allowed(S, T, device, past=past, window=window), False
    allowed = causal_allowed(S, T, device, past=past, window=window)
    diag = (torch.arange(T, device=device)[None, :] == torch.arange(S, device=device)[:, None] + past)
    return (allowed & key_keep_mask(attention_mask, S, T)) | diag, False


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, mask, is_causal: bool,
              dropout_p: float, scale: float | None = None) -> torch.Tensor:
    """Fused SDPA on q [B, nH, S, d] and k/v [B, nKV, T, d] in the compute dtype."""
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1 and not SDPA_NATIVE_GQA:
        k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
    kwargs = {"enable_gqa": True} if (n_rep > 1 and SDPA_NATIVE_GQA) else {}
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                          is_causal=is_causal, scale=scale, **kwargs)


def lm_head_logits(hidden: torch.Tensor, weight: torch.Tensor, logits_dtype: torch.dtype | None = None):
    """LM head whose logits are born in ``logits_dtype``.

    With ``logits_dtype=float32`` on a low-precision model, the hidden state
    [N, H] and head weight [V, H] are upcast (H/V of the logits' size) so the
    matmul accumulates and stores FP32 directly: no low-precision rounding of
    the logits and no extra full-size cast before cross-entropy (measured
    bf16-logit rel. error 1.4e-3 vs 1.1e-7).
    """
    if logits_dtype is None or logits_dtype == weight.dtype:
        return F.linear(hidden, weight)
    return F.linear(hidden.to(logits_dtype), weight.to(logits_dtype))
