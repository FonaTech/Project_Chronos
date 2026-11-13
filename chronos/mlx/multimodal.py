"""MLX twin of ``chronos.model.multimodal``: image/audio encoders and the gated
cross-attention that fuses modality tokens into the text states.

Same parameters and numerics as the Torch module (parity-tested), so a
checkpoint trains on either backend.  Layout differences are handled at
conversion time (``torch_fusion_state`` / ``load_torch_fusion_state``): MLX
convolutions are channels-last (Conv2d weight ``[O,kh,kw,I]``, Conv1d
``[O,k,I]``) and the Torch ``MultiheadAttention`` packs q/k/v into one
``in_proj``.  The fused output only feeds the token embeddings, so routing,
recurrence, KV caches and the cluster objectives are unchanged.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

MODALITY_KEYS = ("image", "audio", "image_embeds", "audio_embeds",
                 "image_mask", "audio_mask", "image_embeds_mask", "audio_embeds_mask")


def _mx(value, dtype=None):
    """torch/numpy/list → mx.array (no copy for mx input)."""
    if value is None or isinstance(value, mx.array):
        return value if value is None or dtype is None else value.astype(dtype)
    try:
        import torch
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu()
            if t.dtype == torch.bfloat16:
                t = t.float()
            value = t.numpy()
    except ModuleNotFoundError:
        pass
    arr = mx.array(value)
    return arr if dtype is None else arr.astype(dtype)


class PatchImageEncoderMLX(nn.Module):
    """``[B,3,H,W]`` (or ``[B,N,3,H,W]``) images → ``[B,T,H]`` patch tokens."""

    def __init__(self, hidden_size: int, patch_size: int = 16, max_tokens: int = 196):
        super().__init__()
        self.patch = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size, bias=False)
        self.max_tokens = int(max_tokens)
        self.norm = nn.LayerNorm(hidden_size)

    def __call__(self, images: mx.array) -> mx.array:
        b = images.shape[0]
        if images.ndim == 5:
            n, c, h, w = images.shape[1:]
            images = images.reshape(b * n, c, h, w)
        x = self.patch(images.transpose(0, 2, 3, 1).astype(self.patch.weight.dtype))  # NHWC
        x = x.reshape(x.shape[0], -1, x.shape[-1])          # row-major patches (Torch flatten(2))
        x = x.reshape(b, -1, x.shape[-1])
        return self.norm(x[:, : self.max_tokens])


class ConvAudioEncoderMLX(nn.Module):
    """``[B,T]`` (or ``[B,1,T]``) waveform → ``[B,T',H]`` frame tokens."""

    def __init__(self, hidden_size: int, max_tokens: int = 128):
        super().__init__()
        mid = max(32, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, 400, stride=160, padding=120), nn.SiLU(),
            nn.Conv1d(mid, hidden_size, 5, stride=2, padding=2), nn.SiLU(),
        )
        self.max_tokens = int(max_tokens)
        self.norm = nn.LayerNorm(hidden_size)

    def __call__(self, audio: mx.array) -> mx.array:
        if audio.ndim == 3:
            audio = audio[:, 0]
        x = self.net(audio[:, :, None].astype(self.net.layers[0].weight.dtype))   # NLC
        return self.norm(x[:, : self.max_tokens])


def _validity(present, b: int, n: int, active_col: mx.array, name: str) -> mx.array:
    """``[B, n]`` bool: keys a row may attend (Torch ``~key_padding_mask``)."""
    if present is None:
        return mx.broadcast_to(active_col[:, None], (b, n))
    present = _mx(present).astype(mx.bool_)
    if present.ndim == 2:
        if present.shape[0] != b:
            raise ValueError(f"{name} modality mask batch dimension mismatch")
        if present.shape[1] >= n:
            return present[:, :n]
        return mx.concatenate([present, mx.zeros((b, n - present.shape[1]), dtype=mx.bool_)], axis=1)
    present = present.reshape(-1)
    if present.size == 1:
        present = mx.broadcast_to(present, (b,))
    if present.size != b:
        raise ValueError(f"{name} modality mask batch dimension mismatch")
    return mx.broadcast_to(mx.logical_and(active_col, present)[:, None], (b, n))


class MultimodalFusionMLX(nn.Module):
    """Gated cross-attention of text states to image/audio tokens."""

    def __init__(self, config):
        super().__init__()
        h = int(config.hidden_size)
        heads = max(1, min(int(config.modality_fusion_heads), int(config.num_attention_heads)))
        while h % heads:
            heads -= 1
        enabled = set(getattr(config, "modalities", []) or [])
        self.image_encoder = (PatchImageEncoderMLX(h, config.image_patch_size, config.max_image_tokens)
                              if "image" in enabled else None)
        self.audio_encoder = (ConvAudioEncoderMLX(h, config.max_audio_tokens)
                              if "audio" in enabled else None)
        self.hidden_size, self.num_heads = h, heads
        self.image_embedding = mx.zeros((1, 1, h))
        self.audio_embedding = mx.zeros((1, 1, h))
        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, h)
        self.v_proj = nn.Linear(h, h)
        self.out_proj = nn.Linear(h, h)
        self.gate = mx.zeros(())
        self.attn_dropout = nn.Dropout(float(config.dropout))
        self.dropout = nn.Dropout(float(config.modality_dropout))
        self.freeze_encoders = bool(config.freeze_modality_encoders)
        if self.freeze_encoders:
            for encoder in (self.image_encoder, self.audio_encoder):
                if encoder is not None:
                    encoder.freeze()

    def _fit_width(self, x: mx.array) -> mx.array:
        width = x.shape[-1]
        if width == self.hidden_size:
            return x
        if width > self.hidden_size:
            return x[..., : self.hidden_size]
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.hidden_size - width)])

    def _embeds(self, value, text_dtype, offset):
        x = _mx(value, text_dtype)
        x = x[:, None] if x.ndim == 2 else x
        return self._fit_width(x) + offset.astype(text_dtype)

    def __call__(self, text_states: mx.array, modality_inputs=None, modality_mask=None):
        if not modality_inputs:
            return text_states, {}
        b, dtype = text_states.shape[0], text_states.dtype
        parts, states = [], {}            # parts: (modality, tokens, validity input)
        for name, encoder, offset in (("image", self.image_encoder, self.image_embedding),
                                      ("audio", self.audio_encoder, self.audio_embedding)):
            embeds = modality_inputs.get(f"{name}_embeds")
            if embeds is not None:
                tokens = self._embeds(embeds, dtype, offset)
                parts.append((name, tokens, modality_inputs.get(f"{name}_embeds_mask")))
                states[name] = tokens
            raw = modality_inputs.get(name)
            if raw is not None and embeds is None:
                if encoder is None:
                    raise ValueError(f"{name} input provided but {name} modality is disabled")
                tokens = encoder(_mx(raw, mx.float32))
                if self.freeze_encoders:
                    tokens = mx.stop_gradient(tokens)
                tokens = (tokens + offset).astype(dtype)
                parts.append((name, tokens, modality_inputs.get(f"{name}_mask")))
                states[name] = mx.concatenate([states[name], tokens], axis=1) if name in states else tokens
        if not parts:
            return text_states, {}
        present = {m for m, _, _ in parts}
        if modality_mask is None:
            active = mx.broadcast_to(mx.array([["image" in present, "audio" in present]]), (b, 2))
        else:
            raw_mask = _mx(modality_mask, mx.float32)
            if raw_mask.ndim == 1:
                col = raw_mask.reshape(-1, 1) if raw_mask.size != 1 else mx.broadcast_to(raw_mask.reshape(1, 1), (b, 1))
                active = mx.broadcast_to(col > 0, (b, 2))
            elif raw_mask.shape[-1] == 2:
                active = raw_mask > 0
            else:
                raise ValueError("modality_mask must be [B] or [B,2] (image,audio)")
        memory = self.dropout(mx.concatenate([t for _, t, _ in parts], axis=1))
        valid = mx.concatenate([_validity(p, b, t.shape[1], active[:, 0 if m == "image" else 1], m)
                                for m, t, p in parts], axis=1)
        cols = sorted({0 if m == "image" else 1 for m in present})
        no_media = mx.logical_not(mx.any(active[:, cols], axis=1))
        # Rows without media keep one key (no all-masked softmax) and get a zero update.
        first = mx.arange(valid.shape[1])[None, :] == 0
        valid = mx.logical_or(valid, mx.logical_and(no_media[:, None], first))
        fused = self._attend(text_states, memory, valid)
        fused = mx.where(no_media[:, None, None], mx.zeros_like(fused), fused)
        return text_states + mx.sigmoid(self.gate).astype(dtype) * fused, states

    def _attend(self, x: mx.array, memory: mx.array, valid: mx.array) -> mx.array:
        b, s, h = x.shape
        t, nh = memory.shape[1], self.num_heads
        hd = h // nh
        q = self.q_proj(x).reshape(b, s, nh, hd).transpose(0, 2, 1, 3)
        k = self.k_proj(memory).reshape(b, t, nh, hd).transpose(0, 2, 1, 3)
        v = self.v_proj(memory).reshape(b, t, nh, hd).transpose(0, 2, 1, 3)
        scores = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * (hd ** -0.5)
        scores = mx.where(valid[:, None, None, :], scores, mx.array(-mx.inf, dtype=mx.float32))
        probs = self.attn_dropout(mx.softmax(scores, axis=-1))
        out = (probs @ v.astype(mx.float32)).astype(x.dtype)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, s, h))


# ---- Torch ↔ MLX state (prefix ``model.multimodal_fusion.``) ---------------
PREFIX = "model.multimodal_fusion."


def _pairs(fusion: MultimodalFusionMLX):
    """``(torch key, getter, setter, layout)``; layout: None | 'conv2d' | 'conv1d'."""
    out = [("gate", fusion, "gate", None),
           ("modality_embeddings.image", fusion, "image_embedding", None),
           ("modality_embeddings.audio", fusion, "audio_embedding", None),
           ("attn.out_proj.weight", fusion.out_proj, "weight", None),
           ("attn.out_proj.bias", fusion.out_proj, "bias", None)]
    if fusion.image_encoder is not None:
        enc = fusion.image_encoder
        out += [("image_encoder.patch.weight", enc.patch, "weight", "conv2d"),
                ("image_encoder.norm.weight", enc.norm, "weight", None),
                ("image_encoder.norm.bias", enc.norm, "bias", None)]
    if fusion.audio_encoder is not None:
        enc = fusion.audio_encoder
        for i in (0, 2):
            conv = enc.net.layers[i]
            out += [(f"audio_encoder.net.{i}.weight", conv, "weight", "conv1d"),
                    (f"audio_encoder.net.{i}.bias", conv, "bias", None)]
        out += [("audio_encoder.norm.weight", enc.norm, "weight", None),
                ("audio_encoder.norm.bias", enc.norm, "bias", None)]
    return out


def _to_mlx_layout(arr: mx.array, layout):
    if layout == "conv2d":
        return arr.transpose(0, 2, 3, 1)       # [O,I,kh,kw] → [O,kh,kw,I]
    if layout == "conv1d":
        return arr.transpose(0, 2, 1)          # [O,I,k] → [O,k,I]
    return arr


def _to_torch_layout(arr: mx.array, layout):
    if layout == "conv2d":
        return arr.transpose(0, 3, 1, 2)
    if layout == "conv1d":
        return arr.transpose(0, 2, 1)
    return arr


def load_torch_fusion_state(fusion: MultimodalFusionMLX, sd, convert, missing: list) -> None:
    """Copy ``model.multimodal_fusion.*`` tensors from a Torch state dict.
    ``convert(tensor) -> mx.array`` applies the target dtype."""
    for key, obj, attr, layout in _pairs(fusion):
        full = PREFIX + key
        if full not in sd:
            missing.append(full)
            continue
        value = _to_mlx_layout(convert(sd[full]), layout)
        if tuple(value.shape) != tuple(getattr(obj, attr).shape):
            raise ValueError(f"MLX checkpoint shape mismatch for {full}: {tuple(value.shape)} != {tuple(getattr(obj, attr).shape)}")
        setattr(obj, attr, value)
    w, b = PREFIX + "attn.in_proj_weight", PREFIX + "attn.in_proj_bias"
    if w not in sd or b not in sd:
        missing.extend(k for k in (w, b) if k not in sd)
        return
    weight, bias, h = convert(sd[w]), convert(sd[b]), fusion.hidden_size
    for i, proj in enumerate((fusion.q_proj, fusion.k_proj, fusion.v_proj)):
        proj.weight = weight[i * h:(i + 1) * h]
        proj.bias = bias[i * h:(i + 1) * h]


def torch_fusion_state(fusion: MultimodalFusionMLX, to_torch) -> dict:
    """``model.multimodal_fusion.*`` Torch tensors (``to_torch(mx.array)``)."""
    sd = {PREFIX + key: to_torch(mx.contiguous(_to_torch_layout(getattr(obj, attr), layout)))
          for key, obj, attr, layout in _pairs(fusion)}
    sd[PREFIX + "attn.in_proj_weight"] = to_torch(mx.concatenate(
        [fusion.q_proj.weight, fusion.k_proj.weight, fusion.v_proj.weight], axis=0))
    sd[PREFIX + "attn.in_proj_bias"] = to_torch(mx.concatenate(
        [fusion.q_proj.bias, fusion.k_proj.bias, fusion.v_proj.bias], axis=0))
    return sd


def batch_modalities(batch):
    """``(ids, labels, modality_inputs | None, modality_mask | None)`` from a
    ``(ids, labels)`` pair or a ``multimodal_collate`` dict."""
    if isinstance(batch, dict):
        media = batch.get("modality_inputs") or None
        return batch["input_ids"], batch["labels"], media, batch.get("modality_mask")
    return batch[0], batch[1], None, None
