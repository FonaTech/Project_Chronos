"""Framework-neutral OpenAI/Ollama serving adapter.

The adapter intentionally contains no FastAPI dependency.  It accepts the
standard request dictionaries used by either HTTP framework and delegates
execution to :class:`ChronosInferenceEngine`; applications can mount the two
methods in FastAPI, Flask, Gradio, or a worker queue without changing model
semantics.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from .api_compat import (
    message_text,
    openai_chat_response,
    ollama_chat_response,
    ollama_generate_response,
    resolve_modality_inputs,
)


class ChronosServingAdapter:
    """Expose a loaded Chronos engine through standard chat/generate schemas."""

    def __init__(self, engine, tokenizer, *, model_name: str = "chronos"):
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name

    def _encode(self, messages, prompt: Optional[str] = None):
        text = prompt if prompt is not None else message_text(messages)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        ids = encoded.input_ids
        mask = encoded.get("attention_mask") if hasattr(encoded, "get") else getattr(encoded, "attention_mask", None)
        modalities = resolve_modality_inputs(
            messages,
            image_size=getattr(self.engine.config, "image_size", 224),
            audio_sample_rate=getattr(self.engine.config, "audio_sample_rate", 16000),
            max_audio_seconds=getattr(self.engine.config, "max_audio_seconds", 30.0),
        )
        return ids, mask, modalities

    def generate(self, *, messages=None, prompt=None, max_tokens: int = 128,
                 temperature: float = 0.85, top_p: float = 0.95,
                 top_k: int = 50, do_sample: bool = True,
                 miss_policy: str = "on_demand") -> tuple[str, dict]:
        ids, mask, modalities = self._encode(messages or [], prompt=prompt)
        output = self.engine.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            do_sample=bool(do_sample),
            miss_policy=miss_policy,
            modality_inputs=modalities or None,
            modality_mask=(torch.ones(ids.shape[0]) if modalities else None),
        )
        generated = output[:, ids.shape[1]:]
        text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        stats = dict(getattr(self.engine, "last_stats", {}) or {})
        stats.setdefault("prompt_tokens", int(ids.shape[1]))
        stats.setdefault("completion_tokens", int(generated.shape[1]))
        return text, stats

    def openai_chat_completion(self, request: Mapping[str, Any]) -> dict:
        messages = request.get("messages") or []
        text, stats = self.generate(
            messages=messages,
            max_tokens=request.get("max_tokens", 128),
            temperature=request.get("temperature", 0.85),
            top_p=request.get("top_p", 0.95),
            top_k=request.get("top_k", 50),
            do_sample=request.get("do_sample", True),
            miss_policy=request.get("chronos_miss_policy", "on_demand"),
        )
        return openai_chat_response(
            str(request.get("model", self.model_name)), text,
            prompt_tokens=stats.get("prompt_tokens", 0),
            completion_tokens=stats.get("completion_tokens", 0),
            chronos_metrics=stats if request.get("include_chronos_metrics") else None,
        )

    def ollama_chat(self, request: Mapping[str, Any]) -> dict:
        messages = request.get("messages") or []
        options = request.get("options") or {}
        text, stats = self.generate(
            messages=messages,
            max_tokens=request.get("num_predict", options.get("num_predict", 128)),
            temperature=request.get("temperature", options.get("temperature", 0.85)),
            top_p=request.get("top_p", options.get("top_p", 0.95)),
            top_k=request.get("top_k", options.get("top_k", 50)),
            do_sample=bool(request.get("do_sample", True)),
            miss_policy=request.get("chronos_miss_policy", "on_demand"),
        )
        return ollama_chat_response(
            str(request.get("model", self.model_name)), text,
            prompt_tokens=stats.get("prompt_tokens", 0),
            completion_tokens=stats.get("completion_tokens", 0),
            chronos_metrics=stats if request.get("include_chronos_metrics") else None,
        )

    def ollama_generate(self, request: Mapping[str, Any]) -> dict:
        text, stats = self.generate(
            prompt=request.get("prompt", ""),
            max_tokens=request.get("options", {}).get("num_predict", 128),
            temperature=request.get("options", {}).get("temperature", 0.85),
            top_p=request.get("options", {}).get("top_p", 0.95),
            top_k=request.get("options", {}).get("top_k", 50),
            do_sample=bool(request.get("do_sample", True)),
            miss_policy=request.get("chronos_miss_policy", "on_demand"),
        )
        return ollama_generate_response(
            str(request.get("model", self.model_name)), text,
            prompt_tokens=stats.get("prompt_tokens", 0),
            completion_tokens=stats.get("completion_tokens", 0),
            chronos_metrics=stats if request.get("include_chronos_metrics") else None,
        )
