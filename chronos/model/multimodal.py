"""Dependency-light image/audio encoders and text fusion for Chronos v2."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class PatchImageEncoder(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int = 16, max_tokens: int = 196):
        super().__init__()
        self.patch = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size, bias=False)
        self.max_tokens = int(max_tokens)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:  # [B,N,C,H,W] -> first image axis flattened as tokens
            b, n, c, h, w = images.shape
            images = images.reshape(b * n, c, h, w)
            x = self.patch(images).flatten(2).transpose(1, 2)
            x = x.reshape(b, n * x.shape[1], x.shape[2])
        else:
            x = self.patch(images).flatten(2).transpose(1, 2)
        return self.norm(x[:, : self.max_tokens])


class ConvAudioEncoder(nn.Module):
    def __init__(self, hidden_size: int, max_tokens: int = 128):
        super().__init__()
        mid = max(32, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, 400, stride=160, padding=120), nn.SiLU(),
            nn.Conv1d(mid, hidden_size, 5, stride=2, padding=2), nn.SiLU(),
        )
        self.max_tokens = int(max_tokens)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        x = self.net(audio.float()).transpose(1, 2)
        return self.norm(x[:, : self.max_tokens])


class MultimodalFusion(nn.Module):
    """Cross-attend text states to lazily present modality tokens."""
    def __init__(self, config):
        super().__init__()
        h = int(config.hidden_size)
        heads = max(1, min(int(config.modality_fusion_heads), int(config.num_attention_heads)))
        while h % heads:
            heads -= 1
        enabled = set(getattr(config, "modalities", []) or [])
        self.image_encoder = (
            PatchImageEncoder(h, config.image_patch_size, config.max_image_tokens)
            if "image" in enabled else None
        )
        self.audio_encoder = (
            ConvAudioEncoder(h, config.max_audio_tokens)
            if "audio" in enabled else None
        )
        self.hidden_size = h
        self.modality_embeddings = nn.ParameterDict({
            "image": nn.Parameter(torch.zeros(1, 1, h)),
            "audio": nn.Parameter(torch.zeros(1, 1, h)),
        })
        self.attn = nn.MultiheadAttention(h, heads, dropout=config.dropout, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(float(config.modality_dropout))
        if bool(config.freeze_modality_encoders):
            for encoder in (self.image_encoder, self.audio_encoder):
                if encoder is None:
                    continue
                for p in encoder.parameters():
                    p.requires_grad_(False)

    def forward(
        self,
        text_states: torch.Tensor,
        modality_inputs: Optional[Dict[str, torch.Tensor]],
        modality_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        if not modality_inputs:
            return text_states, {}
        tokens = []
        token_modalities = []
        token_validity = []
        states = {}
        images = modality_inputs.get("image")
        def _fit_width(features: torch.Tensor) -> torch.Tensor:
            """Project arbitrary cached feature widths without new modules.

            Exported CLIP/Whisper features often differ from Chronos hidden
            size.  Deterministic pad/truncate keeps serving lazy and avoids
            creating uninitialized LazyLinear parameters when only one
            modality is present in a checkpoint.
            """
            width = features.shape[-1]
            if width == self.hidden_size:
                return features
            if width > self.hidden_size:
                return features[..., :self.hidden_size]
            return torch.nn.functional.pad(features, (0, self.hidden_size - width))

        image_embeds = modality_inputs.get("image_embeds")
        if image_embeds is not None:
            image_states = image_embeds.to(text_states.device, text_states.dtype)
            if image_states.dim() == 2:
                image_states = image_states.unsqueeze(1)
            image_states = _fit_width(image_states)
            image_states = image_states + self.modality_embeddings["image"]
            tokens.append(image_states)
            token_modalities.append("image")
            present = modality_inputs.get("image_embeds_mask")
            token_validity.append(present)
            states["image"] = image_states
        if images is not None and image_embeds is None:
            if self.image_encoder is None:
                raise ValueError("image input provided but image modality is disabled")
            image_states = self.image_encoder(images.to(text_states.device))
            image_states = image_states + self.modality_embeddings["image"]
            tokens.append(image_states)
            token_modalities.append("image")
            token_validity.append(modality_inputs.get("image_mask"))
            states["image"] = (
                torch.cat([states["image"], image_states], dim=1)
                if "image" in states else image_states
            )
        audio = modality_inputs.get("audio")
        audio_embeds = modality_inputs.get("audio_embeds")
        if audio_embeds is not None:
            audio_states = audio_embeds.to(text_states.device, text_states.dtype)
            if audio_states.dim() == 2:
                audio_states = audio_states.unsqueeze(1)
            audio_states = _fit_width(audio_states)
            audio_states = audio_states + self.modality_embeddings["audio"]
            tokens.append(audio_states)
            token_modalities.append("audio")
            token_validity.append(modality_inputs.get("audio_embeds_mask"))
            states["audio"] = audio_states
        if audio is not None and audio_embeds is None:
            if self.audio_encoder is None:
                raise ValueError("audio input provided but audio modality is disabled")
            audio_states = self.audio_encoder(audio.to(text_states.device))
            audio_states = audio_states + self.modality_embeddings["audio"]
            tokens.append(audio_states)
            token_modalities.append("audio")
            token_validity.append(modality_inputs.get("audio_mask"))
            states["audio"] = (
                torch.cat([states["audio"], audio_states], dim=1)
                if "audio" in states else audio_states
            )
        if not tokens:
            return text_states, {}
        memory = self.dropout(torch.cat(tokens, dim=1))
        # A mixed batch contains zero-padded media for samples that do not
        # carry that modality.  Use a per-modality key padding mask so those
        # rows cannot attend to another sample's placeholder representation.
        # Scalar masks from v1 remain valid and mean "any modality present".
        if modality_mask is None:
            active = torch.ones(
                (text_states.shape[0], 2), device=text_states.device, dtype=torch.bool
            )
            active[:, 0] = "image" in token_modalities
            active[:, 1] = "audio" in token_modalities
        else:
            raw_mask = modality_mask.to(text_states.device)
            if raw_mask.dim() == 1:
                active = raw_mask.view(-1, 1).expand(-1, 2) > 0
                # Preserve the exact v1 behavior for a scalar batch mask.
                if raw_mask.numel() == 1 and text_states.shape[0] > 1:
                    active = raw_mask.reshape(1, 1).expand(text_states.shape[0], 2) > 0
            elif raw_mask.shape[-1] == 2:
                active = raw_mask > 0
            else:
                raise ValueError("modality_mask must be [B] or [B,2] (image,audio)")
        padding_parts = []
        for modality, modality_tokens, present in zip(token_modalities, tokens, token_validity):
            col = 0 if modality == "image" else 1
            valid = active[:, col]
            if present is not None:
                present = torch.as_tensor(present, device=text_states.device, dtype=torch.bool)
                if present.dim() == 2:
                    if present.shape[0] != text_states.shape[0]:
                        raise ValueError(f"{modality} modality mask batch dimension mismatch")
                    # Padding mask is token-granular for pre-encoded features;
                    # preserve it instead of reducing all rows to one flag.
                    if present.shape[1] != modality_tokens.shape[1]:
                        present = present[:, :modality_tokens.shape[1]]
                        if present.shape[1] < modality_tokens.shape[1]:
                            present = torch.nn.functional.pad(
                                present, (0, modality_tokens.shape[1] - present.shape[1])
                            )
                    padding_parts.append(~present)
                    continue
                present = present.flatten()
                if present.numel() == 1:
                    present = present.expand(text_states.shape[0])
                if present.numel() != text_states.shape[0]:
                    raise ValueError(f"{modality} modality mask batch dimension mismatch")
                valid = valid & present
            padding_parts.append(~valid.unsqueeze(1).expand(-1, modality_tokens.shape[1]))
        key_padding_mask = torch.cat(padding_parts, dim=1) if padding_parts else None
        # MultiheadAttention does not define all-masked rows (softmax NaN).
        # Keep one harmless key for those rows and zero their fused output.
        represented_cols = sorted({0 if m == "image" else 1 for m in token_modalities})
        no_media = ~active[:, represented_cols].any(dim=1)
        if key_padding_mask is not None and no_media.any():
            key_padding_mask[no_media, 0] = False
        fused, _ = self.attn(
            text_states, memory, memory, key_padding_mask=key_padding_mask, need_weights=False
        )
        if no_media.any():
            fused = fused.masked_fill(no_media.view(-1, 1, 1), 0.0)
        return text_states + torch.sigmoid(self.gate) * fused, states
