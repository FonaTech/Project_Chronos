"""
chronos/model/hybrid_attention.py

Hybrid attention for Project Chronos:
- MLAAttention: Multi-head Latent Attention (DeepSeek-style)
  Compresses KV cache via low-rank projection: hidden_dim → kv_latent_dim
  Cache stores latent vectors instead of full KV → ~8-16x smaller KV cache

- SlidingWindowAttention: local attention with fixed window
  KV cache capped at window_size tokens → O(1) memory per step

- HybridAttentionBlock: alternates MLA (even layers) / SlidingWindow (odd layers)
  Even layers: MLA for global context with compressed cache
  Odd layers: SlidingWindow for local patterns with bounded cache
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ChronosConfig
from .kv_quant import kv_read, kv_store, resolve_kv_cache_dtype
from .precision_ops import RMSNorm, apply_rope, attention, build_attention_mask, causal_allowed, rope_tables


class _RopeTables:
    """FP32 RoPE tables owned by one attention layer.

    Stored as plain attributes, not buffers, so ``model.to(bf16)`` never
    rounds them; they follow the activations' device on first use and grow
    geometrically past ``max_position_embeddings``.  ``rope_layout`` picks
    the table width (see ``ChronosConfig.rope_layout``).
    """

    def _init_rope(self, config, full_dim: int, rotated_dim: int) -> None:
        layout = str(getattr(config, "rope_layout", "legacy_sliced") or "legacy_sliced")
        if layout == "per_head" and int(rotated_dim) % 2:
            raise ValueError(f"rope_layout='per_head' needs an even rotated dimension, got {rotated_dim}")
        self._rope_table_dim = int(rotated_dim) if layout == "per_head" else int(full_dim)
        self._rope_base = float(getattr(config, "rope_theta", 1_000_000.0))
        self._rope_scaling = getattr(config, "rope_scaling", None)
        self._rope_len = int(getattr(config, "max_position_embeddings", 4096))
        self._rope_cos = self._rope_sin = None

    def _rope(self, start: int, length: int, width: int, device):
        """FP32 cos/sin rows ``start..start+length-1`` cut to ``width`` columns."""
        need = int(start) + int(length)
        if self._rope_cos is None or self._rope_cos.device != device or self._rope_cos.shape[0] < need:
            size = max(need, self._rope_len, 2 * (self._rope_cos.shape[0] if self._rope_cos is not None else 0))
            self._rope_cos, self._rope_sin = rope_tables(self._rope_table_dim, size, self._rope_base,
                                                         self._rope_scaling, device)
        return self._rope_cos[start:need, :width], self._rope_sin[start:need, :width]


def _key_padding_bias(attention_mask: torch.Tensor, key_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Return an additive key-padding bias broadcastable to [B, H, S, T].

    Chronos callers pass the tokenizer-style 2-D mask [B, total_seq_len].
    Cached decoding and sliding-window layers only attend to the final
    ``key_len`` positions, so slice from the right before broadcasting.
    """
    if attention_mask.dim() == 2:
        if attention_mask.shape[-1] < key_len:
            pad = attention_mask.new_ones(attention_mask.shape[0], key_len - attention_mask.shape[-1])
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
        keep = attention_mask[:, -key_len:].to(dtype=dtype)
        return (1.0 - keep)[:, None, None, :] * -1e9

    if attention_mask.dim() == 3:
        keep = attention_mask[:, :, -key_len:].to(dtype=dtype)
        return (1.0 - keep)[:, None, :, :] * -1e9

    if attention_mask.dim() == 4:
        if attention_mask.shape[-1] < key_len:
            pad_shape = list(attention_mask.shape)
            pad_shape[-1] = key_len - attention_mask.shape[-1]
            pad = attention_mask.new_ones(pad_shape)
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
        mask = attention_mask[..., -key_len:].to(dtype=dtype)
        if mask.min() < 0:
            return mask
        return (1.0 - mask) * -1e9

    raise ValueError(f"attention_mask must be 2-D, 3-D, or 4-D, got shape {tuple(attention_mask.shape)}")


def _sdpa_mixed_value_width(device) -> bool:
    """True when fused SDPA accepts a value width different from Q/K.

    CPU and CUDA kernels do; MPS raises a shape error (checked on torch 2.10),
    so the absorbed MLA path uses explicit matmuls there.
    """
    return device.type != "mps"


# ── MLA: Multi-head Latent Attention ─────────────────────────────

class MLAAttention(_RopeTables, nn.Module):
    """
    Multi-head Latent Attention (DeepSeek MLA style).

    KV compression:
        c_kv = W_kv_down(x)          [B, S, kv_latent_dim]   ← cached
        k, v  = W_kv_up(c_kv)        [B, S, nH, dH]          ← recomputed from cache

    Query decoupled RoPE:
        q_nope = W_q_nope(x)         [B, S, nH, dH - rope_dim]
        q_rope = W_q_rope(x)         [B, S, nH, rope_dim]     ← RoPE applied here
        k_rope = W_k_rope(c_kv)      [B, S, nH, rope_dim]     ← RoPE applied here

    Cache stores only c_kv (latent) + k_rope → much smaller than full KV.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        # KV latent dimension: much smaller than n_kv_heads * head_dim
        self.kv_latent_dim = getattr(config, 'kv_latent_dim',
                                     max(64, self.n_kv_heads * self.head_dim // 8))
        # RoPE dimension per head (subset of head_dim); must leave room for nope
        self.rope_dim = min(getattr(config, 'rope_dim', min(64, self.head_dim)), self.head_dim - 1)
        self.rope_dim = max(1, self.rope_dim)  # at least 1
        self.nope_dim = self.head_dim - self.rope_dim

        H = config.hidden_size

        # Query projections
        self.q_nope_proj = nn.Linear(H, self.n_heads * self.nope_dim, bias=False)
        self.q_rope_proj = nn.Linear(H, self.n_heads * self.rope_dim, bias=False)
        self.q_nope_norm = RMSNorm(self.nope_dim, eps=config.rms_norm_eps)
        self.q_rope_norm = RMSNorm(self.rope_dim, eps=config.rms_norm_eps)

        # KV down-projection (what gets cached)
        self.kv_down_proj = nn.Linear(H, self.kv_latent_dim, bias=False)
        self.kv_down_norm = RMSNorm(self.kv_latent_dim, eps=config.rms_norm_eps)

        # KV up-projections (recomputed from cache at inference)
        self.k_nope_proj = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.nope_dim, bias=False)
        self.k_rope_proj = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.rope_dim, bias=False)
        self.v_proj = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self._init_rope(config, self.head_dim, self.rope_dim)
        # Inference-time cache/decode policy (no parameters; see ChronosConfig).
        self.mla_decode = str(getattr(config, "mla_decode", "expanded") or "expanded").lower()
        self.cache_rope = bool(getattr(config, "mla_cache_rope", False))
        self.kv_int8 = resolve_kv_cache_dtype(config) == "int8"
        self.kv_group = int(getattr(config, "kv_quant_group_size", 32) or 32)

    def _use_absorbed(self, S: int, T: int) -> bool:
        if self.mla_decode == "expanded":
            return False
        if self.mla_decode == "absorbed":
            return True
        # auto: cached short-chunk decode only (prefill is cheaper expanded).
        return (not self.training) and S <= 8 and T > S

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None,
                kv_override=None, return_shared_kv=False):
        """MLA attention.

        Cache entries are ``(c_kv,)`` or, with ``mla_cache_rope``,
        ``(c_kv, k_rope)`` (post-RoPE rope keys ``[B, T, nKV, rope_dim]``);
        either tensor may be a :class:`QuantizedKV` when ``kv_cache_dtype``
        is ``int8``.  ``kv_override`` is the shared-KV recurrent contract:
        the same tuple (always in the compute dtype) returned by iteration
        zero with ``return_shared_kv=True``; it already contains this chunk,
        so queries come from ``x`` while keys/values are re-read.
        """
        B, S, _ = x.shape
        del position_embeddings          # layer-owned FP32 tables (rope_layout aware)
        dtype = x.dtype

        q_nope = self.q_nope_norm(self.q_nope_proj(x).view(B, S, self.n_heads, self.nope_dim))
        q_rope = self.q_rope_norm(self.q_rope_proj(x).view(B, S, self.n_heads, self.rope_dim))
        if kv_override is not None:
            c_kv_full = kv_override[0]
            k_rope = kv_override[1] if len(kv_override) > 1 else None
            past_len = c_kv_full.shape[1] - S
        else:
            past_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        cos_r, sin_r = self._rope(past_len, S, self.rope_dim, x.device)     # FP32 [S, rope_dim]
        q_rope = apply_rope(q_rope, cos_r, sin_r)                           # [B, S, nH, rope_dim]

        if kv_override is None:
            # KV latent (this is what we cache).  Each token is quantized
            # once when appended; the prefix is read back in the compute
            # dtype while this chunk stays exact.
            c_kv = self.kv_down_norm(self.kv_down_proj(x))                  # [B, S, latent]
            quant = self.kv_int8 and use_cache
            past_c = past_key_value[0] if past_key_value is not None else None
            c_kv_full = c_kv if past_c is None else torch.cat([kv_read(past_c, dtype), c_kv], dim=1)
            stored_c = (kv_store(c_kv, past_c, True, self.kv_group) if quant else c_kv_full) if use_cache else None
            k_rope = stored_r = None
            if self.cache_rope:
                k_rope_new = apply_rope(
                    self.k_rope_proj(c_kv).view(B, S, self.n_kv_heads, self.rope_dim), cos_r, sin_r)
                past_r = past_key_value[1] if past_key_value is not None and len(past_key_value) > 1 else None
                if past_c is not None and past_r is None:     # cache built without rope keys
                    past_r = self._rotated_k_rope(kv_read(past_c, dtype), 0)
                k_rope = k_rope_new if past_r is None else torch.cat([kv_read(past_r, dtype), k_rope_new], 1)
                stored_r = (kv_store(k_rope_new, past_r, True, self.kv_group) if quant else k_rope) if use_cache else None
            if k_rope is None:                                              # [B, T, nKV, rope_dim]
                k_rope = self._rotated_k_rope(c_kv_full, 0)
            past_kv = ((stored_c,) if stored_r is None else (stored_c, stored_r)) if use_cache else None
            # Transient view for recurrent reader iterations (never persisted):
            # carries the rotated rope keys so readers skip re-projecting them.
            shared_kv = (c_kv_full, k_rope)
        else:
            past_kv = None
            shared_kv = kv_override
            if k_rope is None:
                k_rope = self._rotated_k_rope(c_kv_full, 0)

        T = c_kv_full.shape[1]
        mask, is_causal = build_attention_mask(S, T, x.device, past=T - S, window=None,
                                               attention_mask=attention_mask)
        dropout_p = self.dropout if self.training else 0.0
        if self._use_absorbed(S, T):
            out = self._absorbed_attention(q_nope, q_rope, c_kv_full, k_rope, mask, is_causal, dropout_p)
        else:
            k_nope = self.k_nope_proj(c_kv_full).view(B, T, self.n_kv_heads, self.nope_dim)
            v = self.v_proj(c_kv_full).view(B, T, self.n_kv_heads, self.head_dim)
            q = torch.cat([q_nope, q_rope], dim=-1)                         # [B, S, nH, head_dim]
            k = torch.cat([k_nope, k_rope], dim=-1)                         # [B, T, nKV, head_dim]
            # Fused attention: native GQA, FP32 softmax inside the kernel.
            out = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask=mask,
                            is_causal=is_causal, dropout_p=dropout_p)
            out = out.transpose(1, 2)

        out = out.reshape(B, S, -1)
        if return_shared_kv:
            return self.resid_dropout(self.o_proj(out)), past_kv, shared_kv
        return self.resid_dropout(self.o_proj(out)), past_kv

    def _rotated_k_rope(self, c_kv: torch.Tensor, start: int) -> torch.Tensor:
        B, T, _ = c_kv.shape
        raw = self.k_rope_proj(c_kv).view(B, T, self.n_kv_heads, self.rope_dim)
        cos, sin = self._rope(start, T, self.rope_dim, c_kv.device)
        return apply_rope(raw, cos, sin)

    def _absorbed_attention(self, q_nope, q_rope, c_kv, k_rope, mask, is_causal, dropout_p):
        """Attention against the latent cache without expanding K/V.

        Per KV group ``g`` (heads ``g*n_rep .. g*n_rep+n_rep-1``):
            score = (q_nope · W_k_nope[g]) · c_kv + q_rope · k_rope[g]
            out   = W_v[g] · Σ softmax(score) · c_kv
        which equals the expanded path in exact arithmetic.  Returns
        ``[B, S, nH, head_dim]`` in the compute dtype.
        """
        B, S, nH, _ = q_nope.shape
        T, L, G, R = c_kv.shape[1], self.kv_latent_dim, self.n_kv_heads, self.n_rep
        scale = 1.0 / math.sqrt(self.head_dim)
        w_k = self.k_nope_proj.weight.view(G, self.nope_dim, L)
        w_v = self.v_proj.weight.view(G, self.head_dim, L)
        q_lat = torch.einsum("bsgrn,gnl->bgrsl", q_nope.view(B, S, G, R, self.nope_dim), w_k)
        q_lat = q_lat.reshape(B, nH, S, L)                                    # head h = g*R + r
        q_r = q_rope.transpose(1, 2)                                          # [B, nH, S, rope]
        k_r = k_rope.transpose(1, 2)                                          # [B, G, T, rope]
        if _sdpa_mixed_value_width(c_kv.device):
            # SDPA with a latent value width (supported on CPU/CUDA): one
            # fused kernel, FP32 softmax inside, native GQA over the G groups.
            k = torch.cat([c_kv[:, None].expand(B, G, T, L), k_r], dim=-1)    # [B, G, T, L+rope]
            v = c_kv[:, None].expand(B, G, T, L)
            lat = attention(torch.cat([q_lat, q_r], dim=-1), k, v, mask=mask, is_causal=is_causal,
                            dropout_p=dropout_p, scale=scale)                 # [B, nH, S, L]
        else:
            # MPS SDPA rejects V width != Q/K width: explicit matmuls in the
            # compute dtype with an FP32 softmax.
            scores = (q_lat * scale) @ c_kv[:, None].transpose(-1, -2)        # [B, nH, S, T]
            scores = scores + ((q_r * scale).view(B, G, R, S, -1)
                               @ k_r[:, :, None].transpose(-1, -2)).view(B, nH, S, T)
            if mask is None and is_causal:
                mask = causal_allowed(S, T, c_kv.device, past=T - S)
            scores = scores.float()
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            if dropout_p:
                probs = F.dropout(probs, p=dropout_p, training=True)
            lat = probs.to(c_kv.dtype) @ c_kv[:, None]                        # [B, nH, S, L]
        out = torch.einsum("bgrsl,gdl->bsgrd", lat.view(B, G, R, S, L), w_v)  # [B, S, G, R, d]
        return out.reshape(B, S, nH, self.head_dim)


# ── Sliding Window Attention ──────────────────────────────────────

_CONFIG_WINDOW = object()    # sentinel: take ``sliding_window_size`` from the config


class SlidingWindowAttention(_RopeTables, nn.Module):
    """
    Local sliding window attention. KV cache is capped at window_size tokens,
    preventing unbounded growth in long-context generation.

    During prefill: full causal attention within each window.
    During decode:  KV cache evicts oldest tokens beyond window_size.
    """

    def __init__(self, config: ChronosConfig, window_size=_CONFIG_WINDOW):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout
        # ``None`` = no window (GlobalAttention): full causal GQA, no eviction.
        self.window_size = getattr(config, 'sliding_window_size', 2048) if window_size is _CONFIG_WINDOW else window_size
        # Extra retained cache entries (outside the attention window, masked
        # out) so speculative decoding can roll back rejected draft tokens.
        self.cache_slack = 0

        H = config.hidden_size
        self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self._init_rope(config, self.head_dim, self.head_dim)
        self.kv_int8 = resolve_kv_cache_dtype(config) == "int8"
        self.kv_group = int(getattr(config, "kv_quant_group_size", 32) or 32)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None,
                kv_override=None, return_shared_kv=False):
        """Sliding-window attention.

        ``kv_override`` is ``(k_full, v_full, absolute_position)`` returned by
        iteration zero with ``return_shared_kv=True``: post-RoPE keys and
        values for the retained prefix plus this chunk, before eviction.
        """
        B, S, _ = x.shape
        del position_embeddings          # layer-owned FP32 tables
        if kv_override is not None:
            xk, xv, past_len = kv_override[0], kv_override[1], int(kv_override[2]) - S
            cached_len = xk.shape[1] - S
        else:
            # cos/sin are full-length buffers; slice to current positions
            cached_len = past_key_value[0].shape[1] if past_key_value is not None else 0
            past_len = (past_key_value[2] if len(past_key_value) > 2 else cached_len) if past_key_value is not None else 0
        cos_s, sin_s = self._rope(past_len, S, self.head_dim, x.device)

        xq = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim))
        if kv_override is None:
            xk = self.k_norm(self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim))
            xv = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
            xq, xk = apply_rope(xq, cos_s, sin_s), apply_rope(xk, cos_s, sin_s)  # [B, S, nH/nKV, d]
            new_k, new_v = xk, xv
            if past_key_value is not None:
                xk = torch.cat([kv_read(past_key_value[0], xk.dtype), xk], dim=1)
                xv = torch.cat([kv_read(past_key_value[1], xv.dtype), xv], dim=1)
        else:
            xq = apply_rope(xq, cos_s, sin_s)

        # Keep full keys for this chunk's queries, then bound the persistent
        # cache. Track absolute positions after eviction for correct RoPE.
        keep = None if self.window_size is None else self.window_size + int(self.cache_slack)
        past_kv = None
        if use_cache and self.kv_int8 and kv_override is None:
            # Quantize only the new tokens (each token once); evict afterwards.
            prev = past_key_value if past_key_value is not None else (None, None)
            qk = kv_store(new_k, prev[0], True, self.kv_group)
            qv = kv_store(new_v, prev[1], True, self.kv_group)
            if keep is not None:
                qk, qv = qk[:, max(0, qk.shape[1] - keep):], qv[:, max(0, qv.shape[1] - keep):]
            past_kv = (qk, qv, past_len + S)
        elif use_cache:
            past_kv = (xk, xv, past_len + S) if keep is None else (xk[:, -keep:], xv[:, -keep:], past_len + S)
        shared_kv = (xk, xv, past_len + S)

        T = xk.shape[1]
        mask, is_causal = build_attention_mask(S, T, x.device, past=cached_len, window=self.window_size,
                                               attention_mask=attention_mask)
        out = attention(xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), mask=mask,
                        is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0)

        out = out.transpose(1, 2).reshape(B, S, -1)
        if return_shared_kv:
            return self.resid_dropout(self.o_proj(out)), past_kv, shared_kv
        return self.resid_dropout(self.o_proj(out)), past_kv


class GlobalAttention(SlidingWindowAttention):
    """Full causal GQA, used when ``use_hybrid_attention=False``.

    It is the sliding-window kernel without a window. Parameter names
    (``q/k/v/o_proj``, ``q_norm``, ``k_norm``) and the ``(k, v, abs_pos)``
    cache are the same, so every recurrent mode works on it, as do int8 KV,
    padding masks and speculative rollback.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__(config, window_size=None)


def make_attention(layer_id: int, config: ChronosConfig) -> nn.Module:
    """
    Layer assignment (``use_hybrid_attention``, default True):
      even layers → MLAAttention  (global context, compressed KV cache)
      odd  layers → SlidingWindowAttention (local patterns, bounded KV cache)
    With hybrid attention off, every layer is GlobalAttention (full GQA).
    """
    if not getattr(config, "use_hybrid_attention", True):
        return GlobalAttention(config)
    if layer_id % 2 == 0:
        return MLAAttention(config)
    return SlidingWindowAttention(config)
