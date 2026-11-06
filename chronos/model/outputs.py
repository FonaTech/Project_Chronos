"""Stable public outputs for Chronos v2."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Optional

import torch
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.utils import ModelOutput


@dataclass
class ChronosCausalLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None
    aux_loss: Optional[torch.Tensor] = None
    lookahead_probs: Optional[torch.Tensor] = None
    halting_probs: Optional[torch.Tensor] = None
    ponder_loss: Optional[torch.Tensor] = None
    recurrent_stability_loss: Optional[torch.Tensor] = None
    recurrent_distill_loss: Optional[torch.Tensor] = None
    recurrent_contraction: Optional[torch.Tensor] = None
    recurrent_depth_used: Optional[torch.Tensor] = None
    cache_hints: Optional[dict] = None
    modality_states: Optional[dict] = None

    @property
    def recurrent_state(self) -> dict:
        """Return recurrent diagnostics in a stable, serialization-friendly
        mapping for trainers and serving adapters."""
        return {
            "halting_probs": self.halting_probs,
            "ponder_loss": self.ponder_loss,
            "stability_loss": self.recurrent_stability_loss,
            "distill_loss": self.recurrent_distill_loss,
            "contraction": self.recurrent_contraction,
            "depth_used": self.recurrent_depth_used,
            "cache_hints": self.cache_hints,
        }

    def legacy_tuple(self):
        primary = MoeCausalLMOutputWithPast(
            loss=self.loss,
            aux_loss=self.aux_loss,
            logits=self.logits,
            past_key_values=self.past_key_values,
            hidden_states=self.hidden_states,
            attentions=self.attentions,
        )
        return primary, self.lookahead_probs

    def __iter__(self):
        # The canonical contract follows HuggingFace ModelOutput iteration.
        # A temporary opt-in bridge exists for external v1 scripts; in-repo
        # callers use fields or ``legacy_output=True``/``legacy_tuple()``.
        if os.environ.get("CHRONOS_LEGACY_OUTPUT_ITER", "1") == "1":
            return iter(self.legacy_tuple())
        return super().__iter__()
