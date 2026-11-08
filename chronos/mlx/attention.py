"""
chronos/mlx/attention.py — MLX-native MLA + Sliding Window Attention.

Design notes vs. the PyTorch version:
- MLA carries only the latent cache. Sliding layers carry K, V and an
  absolute position counter that survives cache eviction.
- RoPE: FP32 tables applied by one fused kernel (single rounding).
- Attention: mx.fast.scaled_dot_product_attention with native GQA and a
  boolean mask built here; softmax is FP32 inside the kernel, so the
  low-precision chain has no score/probability casts.
- SlidingWindow: slice oldest tokens with array indexing, not assignment.
- RoPE rotation is one shapeless-compiled kernel; the attention path itself
  is compiled as part of the training step (chronos/mlx/training/compiled.py).
"""
import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from chronos.mlx.checkpointing import Dropout as TapeDropout   # replayed under recomputation

from chronos.mlx.kv_quant import kv_read, kv_store


def _rope_freqs(
    dim: int,
    max_len: int = 4096,
    base: float = 1_000_000.0,
    rope_scaling: dict | None = None,
) -> mx.array:
    """Precompute RoPE tables [max_len, dim] matching ``chronos.model.kernel.precompute_freqs_cis``.

    The PyTorch path uses duplicated cos/sin columns and rotate-half over the
    whole last dimension. MLX must mirror that exactly for checkpoint parity.
    """
    half = max(dim // 2, 1)
    theta = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32)[:half] / dim))
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max = float(rope_scaling.get("original_max_position_embeddings", 2048))
        factor = float(rope_scaling.get("factor", 16))
        beta_fast = float(rope_scaling.get("beta_fast", 32.0))
        beta_slow = float(rope_scaling.get("beta_slow", 1.0))
        attn_factor = float(rope_scaling.get("attention_factor", 1.0))
        if max_len / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), half - 1)
            ramp = mx.clip((mx.arange(half, dtype=mx.float32) - low) / max(high - low, 0.001), 0, 1)
            theta = theta * (1 - ramp + ramp / factor)
    pos = mx.arange(max_len, dtype=mx.float32)
    freqs = mx.outer(pos, theta)               # [T, half]
    cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1) * attn_factor
    sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1) * attn_factor
    return cos, sin


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@partial(mx.compile, shapeless=True)   # decode grows T every step; no retrace
def _rope_fma(x: mx.array, rotated: mx.array, c: mx.array, s: mx.array) -> mx.array:
    # One fused kernel: FP32 tables, FP32 multiply-add in registers, a single
    # rounding to x.dtype.  No FP32 activation tensor is materialized.
    return (x * c + rotated * s).astype(x.dtype)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE to x [B, T, nH, d] with tables cos/sin [T, >=d] (FP32).

    The tables stay FP32 and are sliced to x.shape[-1]; ``_rope_fma`` fuses
    the rotation so low-precision runs round once instead of three times
    (measured BF16 relative error 1.96e-3 vs 2.48e-3 with BF16 tables).
    """
    _B, T, _nH, d = x.shape
    c = cos[:T, :d][:, None, :]
    s = sin[:T, :d][:, None, :]
    return _rope_fma(x, _rotate_half(x), c, s)


def _sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float, allowed: mx.array | None) -> mx.array:
    """Fused attention on [B, nH, S, d] queries and [B, nKV, T, d] keys/values.

    The kernel keeps scores and softmax in FP32 internally, handles GQA
    natively (no repeated K/V) and returns the compute dtype.  ``allowed`` is
    a boolean [S, T] mask; an array mask (not the "causal" string) is used
    because the vector kernels cannot produce the logsumexp the VJP needs.
    """
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=allowed)


def _causal_allowed(S: int, T: int, window: int | None = None, past: int | None = None) -> mx.array | None:
    """Boolean [S, T] mask for queries at absolute rows ``past..past+S-1``.

    Returns None for single-token decode without a window (all keys visible).
    """
    past = T - S if past is None else past
    q_pos = mx.arange(S)[:, None] + past
    k_pos = mx.arange(T)[None, :]
    if window is None:
        if S == 1:
            return None
        return k_pos <= q_pos
    return (k_pos <= q_pos) & (k_pos > q_pos - window)


def _merge_mask(allowed: mx.array | None, mask: mx.array | None, S: int, T: int) -> mx.array | None:
    """Combine the layer's own boolean mask with an optional caller mask.

    Callers may pass a boolean mask or a legacy additive one (0 / -inf);
    both are aligned to the right edge of the key axis.
    """
    if mask is None:
        return allowed
    m = mask if mask.dtype == mx.bool_ else (mask > -1e4)
    if m.shape[-1] < T:
        m = mx.pad(m, [(0, 0)] * (m.ndim - 1) + [(T - m.shape[-1], 0)], constant_values=True)
    m = m[..., -S:, -T:] if m.ndim >= 2 else m
    return m if allowed is None else (allowed & m)


class _RopeCacheMixin:
    """FP32 RoPE tables, built once and never converted per call.

    ``rope_layout="legacy_sliced"`` reproduces the v1-v3 checkpoints: MLA
    slices the first ``rope_dim`` columns of the head_dim table.  That is not
    a norm-preserving rotation when rope_dim < head_dim, so new configs use
    ``"per_head"`` (tables sized to the rotated dimension).  SWA rotates the
    full head_dim either way, so both layouts are identical there.
    """

    def _init_rope_cache(self, config, dim: int, rotated_dim: int | None = None) -> None:
        rope_len = int(getattr(config, "max_position_embeddings", 4096))
        layout = str(getattr(config, "rope_layout", "legacy_sliced") or "legacy_sliced")
        if layout == "per_head" and rotated_dim and int(rotated_dim) % 2:
            raise ValueError(f"rope_layout='per_head' needs an even rotated dimension, got {rotated_dim}")
        table_dim = int(rotated_dim) if (layout == "per_head" and rotated_dim) else int(dim)
        object.__setattr__(self, "_rope_dim", table_dim)
        object.__setattr__(self, "_rope_base", float(getattr(config, "rope_theta", 1_000_000.0)))
        object.__setattr__(self, "_rope_scaling", getattr(config, "rope_scaling", None))
        cos, sin = _rope_freqs(
            table_dim,
            max_len=rope_len,
            base=self._rope_base,
            rope_scaling=self._rope_scaling,
        )
        object.__setattr__(self, "_cos", cos)
        object.__setattr__(self, "_sin", sin)

    def _attend(self, q, k, v, scale: float, allowed):
        if self.training and getattr(self, "dropout_p", 0.0) > 0.0:
            # Attention-probability dropout needs the explicit probabilities;
            # the softmax stays FP32 and is rounded once for the PV product.
            if self.n_heads != self.n_kv_heads:
                rep = self.n_heads // self.n_kv_heads
                k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
            scores = (q @ k.swapaxes(-1, -2)).astype(mx.float32) * scale
            if allowed is not None:
                scores = mx.where(allowed, scores, -float("inf"))
            return self.attn_dropout(mx.softmax(scores, axis=-1)).astype(q.dtype) @ v
        return _sdpa(q, k, v, scale, allowed)

    def _ensure_rope_cache(self, needed_len: int) -> None:
        current_len = int(self._cos.shape[0])
        if int(needed_len) <= current_len:
            return
        # Grow geometrically to avoid reallocating every decode step past
        # max_position_embeddings.
        new_len = max(int(needed_len), current_len * 2)
        cos, sin = _rope_freqs(
            int(self._rope_dim),
            max_len=new_len,
            base=float(self._rope_base),
            rope_scaling=self._rope_scaling,
        )
        object.__setattr__(self, "_cos", cos)
        object.__setattr__(self, "_sin", sin)


class MLAAttentionMLX(_RopeCacheMixin, nn.Module):
    """
    Multi-head Latent Attention for MLX.

    KV cache stores only (c_kv [B, T, kv_latent_dim],), matching PyTorch.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.n_heads
        self.head_dim = config.head_dim
        self.kv_latent_dim = getattr(config, 'kv_latent_dim', 64)
        # rope_dim is clamped to ≤ head_dim−1 to leave room for nope. The
        # MLX _apply_rope below handles odd rope_dim safely (trailing element
        # passes through unchanged), so there's no evenness requirement.
        self.rope_dim = min(
            max(1, getattr(config, 'rope_dim', min(64, self.head_dim))),
            self.head_dim - 1,
        )
        self.nope_dim = self.head_dim - self.rope_dim
        H = config.hidden_size

        self.q_nope_proj = nn.Linear(H, self.n_heads * self.nope_dim, bias=False)
        self.q_rope_proj = nn.Linear(H, self.n_heads * self.rope_dim, bias=False)
        self.q_nope_norm = nn.RMSNorm(self.nope_dim, eps=config.rms_norm_eps)
        self.q_rope_norm = nn.RMSNorm(self.rope_dim, eps=config.rms_norm_eps)
        self.kv_down_proj = nn.Linear(H, self.kv_latent_dim, bias=False)
        self.kv_down_norm = nn.RMSNorm(self.kv_latent_dim, eps=config.rms_norm_eps)
        self.k_nope_proj  = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.nope_dim, bias=False)
        self.k_rope_proj  = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.rope_dim, bias=False)
        self.v_proj       = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj       = nn.Linear(self.n_heads * self.head_dim, H, bias=False)
        self.attn_dropout = TapeDropout(config.dropout)
        self.resid_dropout = TapeDropout(config.dropout)
        self.dropout_p = float(config.dropout)

        self._init_rope_cache(config, self.head_dim, rotated_dim=self.rope_dim)
        # Inference cache/decode policy (no parameters; see ChronosConfig).
        object.__setattr__(self, "mla_decode", str(getattr(config, "mla_decode", "expanded") or "expanded").lower())
        object.__setattr__(self, "cache_rope", bool(getattr(config, "mla_cache_rope", False)))
        object.__setattr__(self, "kv_int8", str(getattr(config, "kv_cache_dtype", "auto") or "auto").lower() == "int8")
        object.__setattr__(self, "kv_group", int(getattr(config, "kv_quant_group_size", 32) or 32))

    def _use_absorbed(self, S: int, T: int) -> bool:
        if self.mla_decode == "expanded" or (self.training and self.dropout_p > 0.0):
            return False
        if self.mla_decode == "absorbed":
            return True
        return (not self.training) and S <= 8 and T > S

    def _rotated_k_rope(self, c_kv: mx.array, start: int) -> mx.array:
        B, T, _ = c_kv.shape
        raw = self.k_rope_proj(c_kv).reshape(B, T, self.n_kv_heads, self.rope_dim)
        return _apply_rope(raw, self._cos[start:start + T], self._sin[start:start + T])

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,       # (c_kv,) or (c_kv, k_rope); arrays or QuantizedKVMLX
        kv_override=None,
        return_shared_kv: bool = False,
    ):
        """``kv_override``/``return_shared_kv``: v3 shared_first contract
        (see the PyTorch MLAAttention docstring).  The shared view is always
        ``(c_kv_full, k_rope_full)`` in the compute dtype."""
        B, S, H = x.shape
        dtype = x.dtype

        q_nope = self.q_nope_norm(self.q_nope_proj(x).reshape(B, S, self.n_heads, self.nope_dim))
        q_rope = self.q_rope_norm(self.q_rope_proj(x).reshape(B, S, self.n_heads, self.rope_dim))

        if kv_override is not None:
            c_kv_full = kv_override[0]
            k_rope = kv_override[1] if len(kv_override) > 1 else None
            T_full = c_kv_full.shape[1]
            self._ensure_rope_cache(T_full)
            new_cache = None
        else:
            c_kv = self.kv_down_norm(self.kv_down_proj(x))          # [B, S, kv_latent_dim]
            past_c = cache[0] if cache is not None else None
            c_kv_full = c_kv if past_c is None else mx.concatenate([kv_read(past_c, dtype), c_kv], axis=1)
            T_full = c_kv_full.shape[1]
            self._ensure_rope_cache(T_full)
            # MLX caches are lazy: unused (training) cache entries are never
            # evaluated, so building them costs nothing there.
            stored_c = kv_store(c_kv, past_c, self.kv_group) if self.kv_int8 else c_kv_full
            k_rope = None
            new_cache = (stored_c,)
            if self.cache_rope:
                past_len = T_full - S
                k_rope_new = self._rotated_k_rope(c_kv, past_len)
                past_r = cache[1] if cache is not None and len(cache) > 1 else None
                if past_c is not None and past_r is None:
                    past_r = self._rotated_k_rope(kv_read(past_c, dtype), 0)
                k_rope = k_rope_new if past_r is None else mx.concatenate([kv_read(past_r, dtype), k_rope_new], axis=1)
                stored_r = kv_store(k_rope_new, past_r, self.kv_group) if self.kv_int8 else k_rope
                new_cache = (stored_c, stored_r)
        if k_rope is None:
            k_rope = self._rotated_k_rope(c_kv_full, 0)               # [B, T, nKV, rope_dim]

        past_len = T_full - S
        q_rope = _apply_rope(q_rope, self._cos[past_len:past_len + S], self._sin[past_len:past_len + S])
        scale = 1.0 / math.sqrt(self.head_dim)
        allowed = _merge_mask(_causal_allowed(S, T_full), mask, S, T_full)
        if self._use_absorbed(S, T_full):
            out = self._absorbed(q_nope, q_rope, c_kv_full, k_rope, scale, allowed)
        else:
            k_nope = self.k_nope_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.nope_dim)
            v = self.v_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.head_dim)
            q = mx.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)   # [B, nH, S, d]
            k = mx.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)   # [B, nKV, T, d]
            out = self._attend(q, k, v.transpose(0, 2, 1, 3), scale, allowed)
            out = out.transpose(0, 2, 1, 3)
        out = self.resid_dropout(self.o_proj(out.reshape(B, S, -1)))
        if return_shared_kv:
            return out, new_cache, (c_kv_full, k_rope)
        return out, new_cache

    def _absorbed(self, q_nope, q_rope, c_kv, k_rope, scale, allowed):
        """Scores against the latent cache; W_uk folded into q, W_uv into the output.

        Fused SDPA with key width latent+rope and value width latent (MQA per
        KV group, FP32 softmax inside the kernel).  Returns [B, S, nH, d].
        """
        B, S, nH, _ = q_nope.shape
        T, L, G, R = c_kv.shape[1], self.kv_latent_dim, self.n_kv_heads, self.n_heads // self.n_kv_heads
        w_k = self.k_nope_proj.weight.reshape(G, self.nope_dim, L)
        w_v = self.v_proj.weight.reshape(G, self.head_dim, L)
        qn = q_nope.reshape(B, S, G, R, self.nope_dim).transpose(0, 2, 3, 1, 4)    # [B, G, R, S, n]
        q_lat = (qn @ w_k[None, :, None]).reshape(B, nH, S, L)                      # [B, nH, S, L]
        q = mx.concatenate([q_lat, q_rope.transpose(0, 2, 1, 3)], axis=-1)          # [B, nH, S, L+r]
        c = mx.broadcast_to(c_kv[:, None], (B, G, T, L))
        k = mx.concatenate([c, k_rope.transpose(0, 2, 1, 3)], axis=-1)              # [B, G, T, L+r]
        lat = _sdpa(q, k, c, scale, allowed)                                         # [B, nH, S, L]
        lat = lat.reshape(B, G, R, S, L)
        out = lat @ w_v.transpose(0, 2, 1)[None, :, None]                            # [B, G, R, S, d]
        return out.transpose(0, 3, 1, 2, 4).reshape(B, S, nH, self.head_dim)


class SlidingWindowAttentionMLX(_RopeCacheMixin, nn.Module):
    """
    Standard multi-head attention with a sliding KV window for MLX.
    KV cache is capped at window_size tokens — O(1) memory per step.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.n_heads
        self.head_dim = config.head_dim
        self.window_size = getattr(config, 'sliding_window_size', 2048)
        H = config.hidden_size

        self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = TapeDropout(config.dropout)
        self.resid_dropout = TapeDropout(config.dropout)
        self.dropout_p = float(config.dropout)

        self._init_rope_cache(config, self.head_dim)
        object.__setattr__(self, "kv_int8", str(getattr(config, "kv_cache_dtype", "auto") or "auto").lower() == "int8")
        object.__setattr__(self, "kv_group", int(getattr(config, "kv_quant_group_size", 32) or 32))
        # Extra retained entries so self-speculative decoding can roll back up
        # to ``cache_slack`` rejected tokens without losing in-window keys.
        object.__setattr__(self, "cache_slack", 0)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,  # (k [B, T_past, nKV, d], v [B, T_past, nKV, d], absolute_position)
        kv_override=None,
        return_shared_kv: bool = False,
    ):
        B, S, H = x.shape

        q = self.q_norm(self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim))
        if kv_override is not None:
            k, v, end = kv_override
            past_len = int(end) - S
            cached_len = k.shape[1] - S
            self._ensure_rope_cache(past_len + S)
            q = _apply_rope(q, self._cos[past_len:past_len + S], self._sin[past_len:past_len + S])
        else:
            k_new = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim))
            v_new = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim)
            cached_len = cache[0].shape[1] if cache is not None else 0
            past_len = (cache[2] if len(cache) > 2 else cached_len) if cache is not None else 0
            self._ensure_rope_cache(past_len + S)
            q = _apply_rope(q, self._cos[past_len:past_len + S], self._sin[past_len:past_len + S])
            k_new = _apply_rope(k_new, self._cos[past_len:past_len + S], self._sin[past_len:past_len + S])
            if cache is not None:
                k_past, v_past = cache[:2]
                k = mx.concatenate([kv_read(k_past, k_new.dtype), k_new], axis=1)
                v = mx.concatenate([kv_read(v_past, v_new.dtype), v_new], axis=1)
            else:
                k = k_new
                v = v_new

        T_full = k.shape[1]

        # All queries need their OWN causal window. Truncating keys before
        # attention leaks future tokens into early queries during long prefill.
        q_positions = mx.arange(S)[:, None] + cached_len
        k_positions = mx.arange(T_full)[None, :]
        allowed = k_positions <= q_positions
        if self.window_size is not None:              # None: dense (use_hybrid_attention=False)
            allowed = allowed & (k_positions > q_positions - self.window_size)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = self._attend(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
                           scale, _merge_mask(allowed, mask, S, T_full))
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)

        # Cache stores [B, T, nKV, d] (pre-repeat, pre-transpose)
        keep = None if self.window_size is None else self.window_size + int(self.cache_slack)
        if self.kv_int8 and kv_override is None:
            # Quantize only the new tokens (each token once), then evict.
            qk = kv_store(k_new, cache[0] if cache is not None else None, self.kv_group)
            qv = kv_store(v_new, cache[1] if cache is not None else None, self.kv_group)
            start = 0 if keep is None else max(0, qk.shape[1] - keep)
            new_cache = (qk[:, start:], qv[:, start:], past_len + S)
        else:
            new_cache = (k, v, past_len + S) if keep is None else (k[:, -keep:], v[:, -keep:], past_len + S)
        if return_shared_kv:
            return self.resid_dropout(self.o_proj(out)), new_cache, (k, v, past_len + S)
        return self.resid_dropout(self.o_proj(out)), new_cache
