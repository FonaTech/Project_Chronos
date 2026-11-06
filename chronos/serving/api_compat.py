"""OpenAI/Ollama request and response compatibility helpers."""
from __future__ import annotations

import time
import uuid
import base64
import io
import torch
from typing import Iterable, Optional


def message_text(messages: Iterable[dict]) -> str:
    parts = []
    for message in messages or []:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(item.get("text", "")) for item in content
                                  if isinstance(item, dict) and item.get("type") in {"text", "input_text"})
        parts.append(f"{message.get('role', 'user')}: {content}")
    return "\n".join(parts)


def modality_inputs_from_messages(messages: Iterable[dict]) -> dict:
    modalities = {"image": [], "audio": []}
    for message in messages or []:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in {"image", "image_url", "input_image"}:
                value = item.get("image") or item.get("image_url")
                if isinstance(value, dict):
                    value = value.get("url")
                if value:
                    modalities["image"].append(value)
            elif kind in {"audio", "input_audio"}:
                value = item.get("audio") or item.get("input_audio")
                if isinstance(value, dict):
                    # OpenAI input_audio uses {data, format}; retain the
                    # format only as metadata and decode the base64 payload
                    # through the same WAV/PCM resolver as JSONL training.
                    value = value.get("data") or value.get("url")
                    if value and not str(value).startswith("data:"):
                        value = f"data:audio/wav;base64,{value}"
                if value:
                    modalities["audio"].append(value)
    return {key: value for key, value in modalities.items() if value}


def resolve_modality_inputs(messages: Iterable[dict], *, image_size: int = 224,
                            audio_sample_rate: int = 16000,
                            max_audio_seconds: float = 30.0) -> dict:
    """Decode API URL/data-URI media into tensors accepted by ChronosFusion.

    Local paths and HTTP URLs are delegated to the dataset loaders so serving
    and training share identical normalization.  Invalid optional media is
    omitted, allowing a text fallback rather than crashing a standard client.
    """
    from chronos.data.flexible_dataset import _load_audio, _load_image
    out = {}
    raw = modality_inputs_from_messages(messages)
    for kind, values in raw.items():
        decoded = []
        for value in values:
            try:
                if isinstance(value, str) and value.startswith("data:") and "," in value:
                    head, payload = value.split(",", 1)
                    blob = base64.b64decode(payload)
                    if kind == "image":
                        from PIL import Image
                        decoded.append(_load_image(Image.open(io.BytesIO(blob)), image_size))
                    else:
                        # WAV is the portable baseline supported by the
                        # dependency-light training loader.  Unsupported
                        # codecs fail closed and become a text fallback.
                        decoded.append(_load_audio(blob, audio_sample_rate, max_audio_seconds))
                elif kind == "image":
                    decoded.append(_load_image(value, image_size))
                else:
                    decoded.append(_load_audio(value, audio_sample_rate, max_audio_seconds))
            except Exception:
                continue
        if decoded:
            if kind == "image":
                # [B,N,C,H,W] lets PatchImageEncoder treat multiple images as
                # one request; a single image remains semantically identical.
                out[kind] = torch.stack(decoded).unsqueeze(0)
            else:
                # The baseline ConvAudioEncoder consumes one waveform per
                # request.  Preserve deterministic first-item semantics (the
                # dataset path follows the same policy) instead of treating
                # the number of clips as convolution channels.
                out[kind] = decoded[0].flatten().unsqueeze(0)
    return out


def openai_chat_response(model: str, text: str, *, prompt_tokens: int = 0,
                         completion_tokens: int = 0, chronos_metrics: Optional[dict] = None) -> dict:
    result = {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens), "total_tokens": int(prompt_tokens + completion_tokens)},
    }
    if chronos_metrics is not None:
        result["chronos_metrics"] = chronos_metrics
    return result


def ollama_chat_response(model: str, text: str, *, done: bool = True,
                         prompt_tokens: int = 0, completion_tokens: int = 0,
                         chronos_metrics: Optional[dict] = None) -> dict:
    result = {
        "model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": text}, "done": bool(done),
        "done_reason": "stop" if done else None, "prompt_eval_count": int(prompt_tokens),
        "eval_count": int(completion_tokens),
    }
    if chronos_metrics is not None:
        result["chronos_metrics"] = chronos_metrics
    return result


def ollama_generate_response(model: str, text: str, **kwargs) -> dict:
    result = ollama_chat_response(model, text, **kwargs)
    result["response"] = text
    result.pop("message", None)
    return result
