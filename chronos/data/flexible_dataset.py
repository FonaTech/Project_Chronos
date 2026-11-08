"""
chronos/data/flexible_dataset.py

Streaming JSONL loaders with byte-offset indexing. Three variants cover
the six Chronos training stages without ever loading the file into RAM:

  FlexibleDataset       — pretrain / language modeling. Auto-detects
                           common text-field schemas (text, content,
                           instruction+output, conversations, messages).
  StreamingSFTDataset   — supervised fine-tuning. Uses the tokenizer's
                           chat template; only assistant tokens contribute
                           to the loss (label = -100 elsewhere). Reads the
                           common `conversations` jsonl layout.
  StreamingDPODataset   — preference pairs. Returns the dict shape DPO /
                           ORPO trainers expect.

Each dataset stores only `offsets: list[int]` (8 bytes/sample) plus the
file path. `__getitem__` does seek+readline+json.loads on demand, so a
multi-GB corpus uses < 1 GB of resident memory.

The file handle is opened lazily per process (re-opens after fork so
DataLoader num_workers > 0 still works) and protected by a thread lock
so num_workers = 0 paths with concurrent access are safe.
"""
from __future__ import annotations

import json
import os
import threading
import warnings
import wave
import urllib.request

import torch
from torch.utils.data import Dataset


# ── Field-name auto-detection (FlexibleDataset only) ──────────────────

_TEXT_KEYS = ("text", "content", "story", "passage", "document", "article")
_PAIR_KEYS = (
    ("instruction", "output"),
    ("instruction", "response"),
    ("prompt", "response"),
    ("prompt", "completion"),
    ("input", "output"),
    ("question", "answer"),
    ("query", "answer"),
)

_IMAGE_KEYS = ("image", "images", "image_path", "image_paths", "image_url", "image_urls")
_AUDIO_KEYS = ("audio", "audios", "audio_path", "audio_paths", "audio_url", "audio_urls", "wav")


class ModalityRegistry:
    """Single registry used by datasets, trainers and serving adapters."""

    fields = {"image": _IMAGE_KEYS, "audio": _AUDIO_KEYS}

    def detect(self, record: dict) -> set[str]:
        return detect_modalities(record)

    def summary(self, data_path: str, max_records: int = 10000) -> dict:
        return dataset_modality_summary(data_path, max_records=max_records)


class RecordDetector:
    """Compatibility façade for callers that need schema-only detection."""

    def __call__(self, record: dict) -> set[str]:
        return detect_modalities(record)

    detect = __call__


def detect_modalities(record: dict) -> set[str]:
    """Detect media modalities without opening files or allocating tensors."""
    found = set()
    # Accept the canonical v2 normalized record as input as well as the raw
    # heterogeneous JSONL forms.  This makes dataset summaries, collators and
    # serving adapters true round-trip consumers of the same schema.
    nested = record.get("modalities")
    if isinstance(nested, dict):
        if nested.get("image") not in (None, "", [], {}):
            found.add("image")
        if nested.get("audio") not in (None, "", [], {}):
            found.add("audio")
        if nested.get("image_embeds") is not None:
            found.add("image")
        if nested.get("audio_embeds") is not None:
            found.add("audio")
    if any(record.get(key) not in (None, "", [], {}) for key in _IMAGE_KEYS):
        found.add("image")
    if any(record.get(key) not in (None, "", [], {}) for key in _AUDIO_KEYS):
        found.add("audio")
    if any(key in record for key in ("image_embeds", "image_features")):
        found.add("image")
    if any(key in record for key in ("audio_embeds", "audio_features")):
        found.add("audio")
    return found


def normalize_multimodal_record(record: dict) -> dict:
    """Normalize heterogeneous media fields to the canonical v2 schema."""
    out = {
        "messages": record.get("messages") or record.get("conversations") or [],
        "modalities": {"image": [], "audio": []},
        "labels": record.get("labels", {}),
        "metadata": {k: v for k, v in record.items() if k not in _IMAGE_KEYS + _AUDIO_KEYS},
    }
    nested = record.get("modalities") if isinstance(record.get("modalities"), dict) else {}
    for key in _IMAGE_KEYS:
        value = record.get(key)
        if value not in (None, "", [], {}):
            out["modalities"]["image"] = list(value) if isinstance(value, (list, tuple)) else [value]
            break
    if not out["modalities"]["image"] and nested.get("image") not in (None, "", [], {}):
        value = nested["image"]
        out["modalities"]["image"] = list(value) if isinstance(value, (list, tuple)) else [value]
    for key in _AUDIO_KEYS:
        value = record.get(key)
        if value not in (None, "", [], {}):
            out["modalities"]["audio"] = list(value) if isinstance(value, (list, tuple)) else [value]
            break
    if not out["modalities"]["audio"] and nested.get("audio") not in (None, "", [], {}):
        value = nested["audio"]
        out["modalities"]["audio"] = list(value) if isinstance(value, (list, tuple)) else [value]
    # Preserve already-materialized features for embedding-only datasets.
    for name, aliases in (("image_embeds", ("image_embeds", "image_features")),
                          ("audio_embeds", ("audio_embeds", "audio_features"))):
        for key in aliases:
            if key in record:
                out["modalities"][name] = record[key]
                break
        if name not in out["modalities"] and nested.get(name) is not None:
            out["modalities"][name] = nested[name]
    return out


def dataset_modality_summary(data_path: str, max_records: int = 10000) -> dict:
    """Stream a JSONL prefix and return deterministic modality statistics."""
    counts = {"text": 0, "image": 0, "audio": 0, "multimodal": 0, "invalid": 0}
    scanned = 0
    with open(data_path, "rb") as handle:
        for raw in handle:
            if scanned >= max_records:
                break
            if not raw.strip():
                continue
            scanned += 1
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["invalid"] += 1
                continue
            mods = detect_modalities(record)
            if not mods:
                counts["text"] += 1
            else:
                for mod in mods:
                    counts[mod] += 1
                if len(mods) > 1:
                    counts["multimodal"] += 1
    counts["scanned"] = scanned
    counts["detected_modalities"] = [m for m in ("image", "audio") if counts[m] > 0]
    return counts


def _extract_text(record: dict) -> str:
    for k in _TEXT_KEYS:
        if k in record and record[k]:
            return str(record[k])
    for k1, k2 in _PAIR_KEYS:
        if k1 in record and k2 in record:
            parts = [str(record[k1]).strip(), str(record[k2]).strip()]
            return "\n".join(p for p in parts if p)
    if "conversations" in record and isinstance(record["conversations"], list):
        return " ".join(
            str(c.get("value", c.get("content", "")))
            for c in record["conversations"]
        )
    if "messages" in record and isinstance(record["messages"], list):
        return " ".join(str(m.get("content", "")) for m in record["messages"])
    parts = [str(v) for v in record.values()
             if isinstance(v, (str, int, float)) and str(v).strip()]
    return " ".join(parts)


# ── Shared streaming infrastructure ───────────────────────────────────

class _StreamingJSONLBase(Dataset):
    """Base class providing byte-offset index + lazy per-process file handle."""

    def __init__(self, data_path: str):
        self.data_path = os.path.abspath(data_path)
        self.offsets: list[int] = []
        with open(self.data_path, "rb") as f:
            pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(pos)
                pos = f.tell()
        if not self.offsets:
            raise ValueError(f"No JSONL records in {data_path}")
        self._fh = None
        self._fh_lock = threading.Lock()

    def _get_fh(self):
        pid = os.getpid()
        if self._fh is None or getattr(self._fh, "_chronos_pid", None) != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.data_path, "rb")
            self._fh._chronos_pid = pid
        return self._fh

    def _read_record(self, index: int) -> dict:
        offset = self.offsets[index]
        with self._fh_lock:
            fh = self._get_fh()
            fh.seek(offset)
            raw = fh.readline()
        return json.loads(raw.decode("utf-8"))

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        """Make streaming datasets safe for macOS/Windows spawn workers.

        DataLoader with num_workers > 0 pickles the dataset. File handles and
        thread locks cannot be pickled, and each worker must open its own file
        descriptor anyway so seeks are process-local.
        """
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fh = None
        self._fh_lock = threading.Lock()

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


# ── Pretrain / generic LM ────────────────────────────────────────────

class FlexibleDataset(_StreamingJSONLBase):
    """Pretrain-style streaming dataset; auto-detects the text field."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        first = self._read_record(0)
        detected = _extract_text(first)
        field_hint = (
            next((k for k in _TEXT_KEYS if k in first), None)
            or next((f"{k1}+{k2}" for k1, k2 in _PAIR_KEYS
                     if k1 in first and k2 in first), None)
            or ("conversations" if "conversations" in first else "auto")
        )
        print(f"[FlexibleDataset] {len(self.offsets)} records (streaming), "
              f"detected format: '{field_hint}', "
              f"sample preview: {detected[:80]!r}")

    def __getitem__(self, index):
        record = self._read_record(index)
        text = _extract_text(record)
        tok = self.tokenizer(
            text, add_special_tokens=False,
            max_length=self.max_length - 2, truncation=True,
        )
        tokens = (
            [self.tokenizer.bos_token_id]
            + tok.input_ids
            + [self.tokenizer.eos_token_id]
        )
        pad_id = self.tokenizer.pad_token_id or 0
        tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        return input_ids, labels


def _media_value(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def record_media(record: dict, *, image_size: int = 224, audio_sample_rate: int = 16000,
                 max_audio_seconds: float = 30.0, media_error_policy: str = "warn"):
    """``(modality_inputs, modality_mask[2])`` for one record, or ``None``.

    Pre-encoded ``*_embeds`` take precedence over a path/value of the same
    modality.  Used by the preference (DPO/ORPO) and GRPO prompt loaders; the
    pair or the rollout group shares the record's media."""
    if not detect_modalities(record):
        return None
    normalized = normalize_multimodal_record(record)
    media = {}
    for name in ("image_embeds", "audio_embeds"):
        value = normalized["modalities"].get(name)
        if value is not None:
            try:
                media[name] = torch.as_tensor(value, dtype=torch.float32)
            except Exception as exc:
                if media_error_policy == "raise":
                    raise ValueError(f"invalid {name}: {exc}") from exc
                warnings.warn(f"invalid {name}: {exc}", RuntimeWarning, stacklevel=2)
    for modality, loader in (
        ("image", lambda v: _load_image(v, image_size)),
        ("audio", lambda v: _load_audio(v, audio_sample_rate, max_audio_seconds)),
    ):
        if f"{modality}_embeds" in media:
            continue
        value = _media_value(normalized["modalities"].get(modality))
        if value is None:
            continue
        try:
            media[modality] = loader(value)
        except Exception as exc:
            if media_error_policy == "raise":
                raise
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
    if not media:
        return None
    mask = torch.tensor([float("image" in media or "image_embeds" in media),
                         float("audio" in media or "audio_embeds" in media)])
    return media, mask


class MediaResolver:
    """Lazy media resolver shared by JSONL and API serving paths."""

    def __init__(self, *, image_size: int = 224, audio_sample_rate: int = 16000,
                 max_audio_seconds: float = 30.0):
        self.image_size = int(image_size)
        self.audio_sample_rate = int(audio_sample_rate)
        self.max_audio_seconds = float(max_audio_seconds)

    def image(self, value):
        return _load_image(value, self.image_size)

    def audio(self, value):
        return _load_audio(value, self.audio_sample_rate, self.max_audio_seconds)

    def resolve(self, modality: str, value):
        if modality == "image":
            return self.image(value)
        if modality == "audio":
            return self.audio(value)
        raise ValueError(f"unsupported modality: {modality}")


class MultimodalCollator:
    """Callable wrapper retaining the functional collator API."""

    def __call__(self, batch):
        return multimodal_collate(batch)


def _load_image(value, size: int) -> torch.Tensor:
    try:
        from PIL import Image
        if isinstance(value, torch.Tensor):
            image = value
            if image.dim() == 3 and image.shape[0] not in (1, 3):
                image = image.permute(2, 0, 1)
            return torch.nn.functional.interpolate(
                image.float().unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
            ).squeeze(0).div(255.0 if image.max() > 1.5 else 1.0)
        if isinstance(value, Image.Image):
            image = value
        elif str(value).startswith(("http://", "https://")):
            with urllib.request.urlopen(str(value), timeout=10) as response:
                import io
                image = Image.open(io.BytesIO(response.read()))
        else:
            image = Image.open(str(value))
        image = image.convert("RGB").resize((size, size))
        import numpy as np
        return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float().div(255.0)
    except Exception as exc:
        raise ValueError(f"unable to load image {value!r}: {exc}") from exc


def _load_audio(value, sample_rate: int, max_seconds: float) -> torch.Tensor:
    try:
        if isinstance(value, torch.Tensor):
            audio = value.float().flatten()
            return audio[: int(sample_rate * max_seconds)]
        # wave.open accepts paths and binary file-like objects.  Supporting
        # bytes here makes API data-URI audio use exactly the JSONL decoder.
        source = value
        if isinstance(value, (bytes, bytearray)):
            import io
            source = io.BytesIO(value)
        with wave.open(source, "rb") as handle:
            n = min(handle.getnframes(), int(sample_rate * max_seconds))
            raw = handle.readframes(n)
            import numpy as np
            audio = torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy()).float().div(32768.0)
            channels = int(handle.getnchannels())
            if channels > 1:
                audio = audio.view(-1, channels).mean(dim=1)
            source_rate = int(handle.getframerate())
            if source_rate != sample_rate and audio.numel() > 1:
                target = int(audio.numel() * sample_rate / source_rate)
                audio = torch.nn.functional.interpolate(
                    audio.view(1, 1, -1), size=target, mode="linear", align_corners=False
                ).view(-1)
            return audio[: int(sample_rate * max_seconds)]
    except Exception as exc:
        raise ValueError(f"unable to load audio {value!r}: {exc}") from exc


class MultimodalStreamingDataset(FlexibleDataset):
    """Streaming LM dataset with lazy image/audio materialization.

    Text-only records retain the regular tuple contract. Media records return
    a dict consumed by ``multimodal_collate`` and ChronosTrainer.
    """
    def __init__(self, data_path, tokenizer, max_length=512, *, image_size=224,
                 audio_sample_rate=16000, max_audio_seconds=30.0,
                 media_error_policy="warn"):
        super().__init__(data_path, tokenizer, max_length=max_length)
        self.image_size = int(image_size)
        self.audio_sample_rate = int(audio_sample_rate)
        self.max_audio_seconds = float(max_audio_seconds)
        self.media_error_policy = str(media_error_policy)
        self.modality_summary = dataset_modality_summary(data_path)

    def __getitem__(self, index):
        record = self._read_record(index)
        base_ids, base_labels = _tokenize_record(record, self.tokenizer, self.max_length)
        mods = detect_modalities(record)
        if not mods:
            return base_ids, base_labels
        normalized = normalize_multimodal_record(record)
        media = {}
        # Pre-encoded features are first-class inputs.  They are intentionally
        # kept on CPU and only batched here; the model/projector decides how to
        # move them to the execution device.
        for name in ("image_embeds", "audio_embeds"):
            value = normalized["modalities"].get(name)
            if value is not None:
                try:
                    media[name] = torch.as_tensor(value, dtype=torch.float32)
                except Exception as exc:
                    if self.media_error_policy == "raise":
                        raise ValueError(f"invalid {name}: {exc}") from exc
                    warnings.warn(f"invalid {name}: {exc}", RuntimeWarning, stacklevel=2)
        for modality, loader in (
            ("image", lambda v: _load_image(v, self.image_size)),
            ("audio", lambda v: _load_audio(v, self.audio_sample_rate, self.max_audio_seconds)),
        ):
            # A pre-encoded representation takes precedence over a path/value.
            if f"{modality}_embeds" in media:
                continue
            value = _media_value(normalized["modalities"].get(modality))
            if value is None:
                continue
            try:
                media[modality] = loader(value)
            except Exception as exc:
                if self.media_error_policy == "raise":
                    raise
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        if not media:
            return base_ids, base_labels
        return {
            "input_ids": base_ids,
            "labels": base_labels,
            "modality_inputs": media,
            # Per-modality presence mask.  The old scalar mask is accepted by
            # the model, but a two-column mask is required for mixed batches
            # so a text+image sample never accidentally attends to an audio
            # placeholder (and vice versa).
            "modality_mask": torch.tensor([
                float("image" in media or "image_embeds" in media),
                float("audio" in media or "audio_embeds" in media),
            ]),
            "metadata": normalized.get("metadata", {}),
        }


def _tokenize_record(record, tokenizer, max_length):
    text = _extract_text(record)
    tok = tokenizer(text, add_special_tokens=False, max_length=max_length - 2, truncation=True)
    tokens = [tokenizer.bos_token_id] + tok.input_ids + [tokenizer.eos_token_id]
    pad_id = tokenizer.pad_token_id or 0
    tokens = tokens + [pad_id] * (max_length - len(tokens))
    ids = torch.tensor(tokens, dtype=torch.long)
    labels = ids.clone()
    labels[ids == pad_id] = -100
    return ids, labels


def multimodal_collate(batch):
    """Collate mixed text/media records without invoking absent encoders."""
    ids, labels, masks = [], [], []
    images, audios, image_embeds, audio_embeds = [], [], [], []
    has_image = has_audio = has_image_embeds = has_audio_embeds = False
    for item in batch:
        if isinstance(item, dict):
            ids.append(item["input_ids"]); labels.append(item["labels"])
            media_item = item.get("modality_inputs", {})
            image = media_item.get("image")
            audio = media_item.get("audio")
            image_embed = media_item.get("image_embeds")
            audio_embed = media_item.get("audio_embeds")
            item_mask = item.get("modality_mask")
            if isinstance(item_mask, dict):
                mask = torch.tensor([
                    float(item_mask.get("image", image is not None or image_embed is not None)),
                    float(item_mask.get("audio", audio is not None or audio_embed is not None)),
                ])
            else:
                mask = torch.as_tensor(item_mask if item_mask is not None else 1.0).flatten()
                if mask.numel() == 1:
                    # Legacy scalar means "this record has media".  Infer
                    # the exact columns from materialized inputs.
                    active = float(mask.item())
                    mask = torch.tensor([
                        active * float(image is not None or image_embed is not None),
                        active * float(audio is not None or audio_embed is not None),
                    ])
                elif mask.numel() != 2:
                    raise ValueError("modality_mask must have one scalar or two [image,audio] values")
            masks.append(mask.float())
            has_image = has_image or image is not None
            has_audio = has_audio or audio is not None
            has_image_embeds = has_image_embeds or image_embed is not None
            has_audio_embeds = has_audio_embeds or audio_embed is not None
            images.append(image); audios.append(audio)
            image_embeds.append(image_embed); audio_embeds.append(audio_embed)
        else:
            ids.append(item[0]); labels.append(item[1]); masks.append(torch.zeros(2))
            images.append(None); audios.append(None)
            image_embeds.append(None); audio_embeds.append(None)
    result = {"input_ids": torch.stack(ids), "labels": torch.stack(labels),
            "modality_mask": torch.stack(masks)}
    # A pre-encoded representation takes precedence for every sample.  Do not
    # build both branches in one batch; doing so wastes encoder compute and
    # makes duplicate modality tokens dependent on batch composition.
    if has_image and not has_image_embeds:
        shape = next(x.shape for x in images if x is not None)
        result["modality_inputs"] = {"image": torch.stack([
            x if x is not None else torch.zeros(shape) for x in images
        ]), "image_mask": torch.tensor([x is not None for x in images])}
    if has_audio and not has_audio_embeds:
        width = max(x.numel() for x in audios if x is not None)
        result.setdefault("modality_inputs", {})["audio"] = torch.stack([
            (torch.nn.functional.pad(x.flatten(), (0, width - x.numel()))
             if x is not None else torch.zeros(width)) for x in audios
        ])
        result["modality_inputs"]["audio_mask"] = torch.tensor([
            x is not None for x in audios
        ])
    def _pad_embeddings(values):
        present = [v.float() if v is not None else None for v in values]
        exemplar = next(v for v in present if v is not None)
        if exemplar.dim() == 1:
            exemplar = exemplar.unsqueeze(0)
        width = exemplar.shape[-1]
        lengths = [v.shape[-2] if v is not None and v.dim() > 1 else 1 for v in present]
        out = exemplar.new_zeros((len(values), max(lengths), width))
        valid = torch.zeros((len(values), max(lengths)), dtype=torch.bool)
        for i, value in enumerate(present):
            if value is None:
                continue
            value = value.unsqueeze(0) if value.dim() == 1 else value
            if value.shape[-1] != width:
                raise ValueError("mixed pre-encoded modality embedding widths in one batch")
            out[i, :value.shape[-2]] = value
            valid[i, :value.shape[-2]] = True
        return out, valid
    if has_image_embeds:
        image_batch, image_valid = _pad_embeddings(image_embeds)
        result.setdefault("modality_inputs", {})["image_embeds"] = image_batch
        result["modality_inputs"]["image_embeds_mask"] = image_valid
    if has_audio_embeds:
        audio_batch, audio_valid = _pad_embeddings(audio_embeds)
        result.setdefault("modality_inputs", {})["audio_embeds"] = audio_batch
        result["modality_inputs"]["audio_embeds_mask"] = audio_valid
    return result


# ── SFT (conversations → assistant-token loss mask) ─────────────────

class StreamingSFTDataset(_StreamingJSONLBase):
    """Streaming chat-SFT dataset (``conversations`` jsonl records).

    Reads `conversations: [{role, content, ...}, ...]` records, applies
    the tokenizer's chat template, and masks every position outside an
    assistant turn with -100 so the loss only fires on assistant tokens.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Sentinels marking the assistant turn boundary in encoded form.
        # ChatML-style turn markers (`bos`+'assistant\n' / `eos`+'\n').
        self.bos_id = tokenizer(
            f'{tokenizer.bos_token}assistant\n', add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f'{tokenizer.eos_token}\n', add_special_tokens=False
        ).input_ids
        print(f"[StreamingSFTDataset] {len(self.offsets)} records (streaming)")

    def _build_prompt(self, conversations) -> str:
        messages = []
        tools = None
        for m in conversations:
            m = dict(m)
            if m.get("role") == "system" and m.get("tools"):
                tools = json.loads(m["tools"]) if isinstance(m["tools"], str) else m["tools"]
            if m.get("tool_calls") and isinstance(m["tool_calls"], str):
                m["tool_calls"] = json.loads(m["tool_calls"])
            messages.append(m)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools,
        )

    def _label_mask(self, input_ids: list[int]) -> list[int]:
        """-100 outside every <bos>assistant\\n ... <eos>\\n span."""
        labels = [-100] * len(input_ids)
        i = 0
        nb = len(self.bos_id)
        ne = len(self.eos_id)
        while i < len(input_ids):
            if input_ids[i:i + nb] == self.bos_id:
                start = i + nb
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + ne] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + ne, self.max_length)):
                    labels[j] = input_ids[j]
                i = end + ne if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        record = self._read_record(index)
        convs = record.get("conversations")
        if not convs:
            # Graceful fall-through for mixed corpora — treat as pretrain.
            text = _extract_text(record)
            ids = self.tokenizer(text, truncation=True,
                                 max_length=self.max_length).input_ids
        else:
            prompt = self._build_prompt(convs)
            ids = self.tokenizer(prompt).input_ids[:self.max_length]
        pad_id = self.tokenizer.pad_token_id or 0
        ids = ids + [pad_id] * (self.max_length - len(ids))
        labels = self._label_mask(ids)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class MultimodalStreamingSFTDataset(StreamingSFTDataset):
    """SFT records with lazy media and optional pre-encoded features."""

    def __init__(self, data_path, tokenizer, max_length=1024, *, image_size=224,
                 audio_sample_rate=16000, max_audio_seconds=30.0,
                 media_error_policy="warn"):
        super().__init__(data_path, tokenizer, max_length=max_length)
        self.image_size = int(image_size)
        self.audio_sample_rate = int(audio_sample_rate)
        self.max_audio_seconds = float(max_audio_seconds)
        self.media_error_policy = str(media_error_policy)
        self.modality_summary = dataset_modality_summary(data_path)

    def __getitem__(self, index):
        input_ids, labels = super().__getitem__(index)
        record = self._read_record(index)
        if not detect_modalities(record):
            return input_ids, labels
        normalized = normalize_multimodal_record(record)
        media = {}
        for name in ("image_embeds", "audio_embeds"):
            value = normalized["modalities"].get(name)
            if value is not None:
                try:
                    media[name] = torch.as_tensor(value, dtype=torch.float32)
                except Exception as exc:
                    if self.media_error_policy == "raise":
                        raise ValueError(f"invalid {name}: {exc}") from exc
                    warnings.warn(f"invalid {name}: {exc}", RuntimeWarning, stacklevel=2)
        for modality, loader in (
            ("image", lambda v: _load_image(v, self.image_size)),
            ("audio", lambda v: _load_audio(v, self.audio_sample_rate, self.max_audio_seconds)),
        ):
            if f"{modality}_embeds" in media:
                continue
            value = _media_value(normalized["modalities"].get(modality))
            if value is None:
                continue
            try:
                media[modality] = loader(value)
            except Exception as exc:
                if self.media_error_policy == "raise":
                    raise
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        if not media:
            return input_ids, labels
        return {
            "input_ids": input_ids, "labels": labels,
            "modality_inputs": media,
            "modality_mask": torch.tensor([
                float("image" in media or "image_embeds" in media),
                float("audio" in media or "audio_embeds" in media),
            ]),
            "metadata": normalized.get("metadata", {}),
        }


# ── DPO / ORPO (chosen + rejected pair) ─────────────────────────────

class StreamingDPODataset(_StreamingJSONLBase):
    """Streaming preference-pair dataset (``chosen`` / ``rejected`` jsonl records).

    Returns the dict shape the DPO/ORPO trainers expect:
      {x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected}
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(
            f'{tokenizer.bos_token}assistant\n', add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f'{tokenizer.eos_token}\n', add_special_tokens=False
        ).input_ids
        print(f"[StreamingDPODataset] {len(self.offsets)} pairs (streaming)")

    def _loss_mask(self, ids: list[int]) -> list[int]:
        mask = [0] * len(ids)
        i = 0
        nb = len(self.bos_id); ne = len(self.eos_id)
        while i < len(ids):
            if ids[i:i + nb] == self.bos_id:
                start = i + nb
                end = start
                while end < len(ids):
                    if ids[end:end + ne] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + ne, self.max_length)):
                    mask[j] = 1
                i = end + ne if end < len(ids) else len(ids)
            else:
                i += 1
        return mask

    def __getitem__(self, index):
        rec = self._read_record(index)
        chosen = rec["chosen"]; rejected = rec["rejected"]

        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False,
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False,
        )

        c_enc = self.tokenizer(chosen_prompt, truncation=True,
                               max_length=self.max_length, padding="max_length")
        r_enc = self.tokenizer(rejected_prompt, truncation=True,
                               max_length=self.max_length, padding="max_length")
        c_ids = c_enc["input_ids"]
        r_ids = r_enc["input_ids"]
        c_mask = self._loss_mask(c_ids)
        r_mask = self._loss_mask(r_ids)
        return {
            "x_chosen":     torch.tensor(c_ids[:-1], dtype=torch.long),
            "y_chosen":     torch.tensor(c_ids[1:],  dtype=torch.long),
            "mask_chosen":  torch.tensor(c_mask[1:], dtype=torch.long),
            "x_rejected":   torch.tensor(r_ids[:-1], dtype=torch.long),
            "y_rejected":   torch.tensor(r_ids[1:],  dtype=torch.long),
            "mask_rejected":torch.tensor(r_mask[1:], dtype=torch.long),
        }


class MultimodalStreamingDPODataset(StreamingDPODataset):
    """Preference pairs with the prompt's image/audio.

    Media fields sit on the record (``image``/``audio``/``*_embeds`` or a
    ``modalities`` dict, as for SFT); chosen and rejected answer the same
    multimodal prompt, so both sides of the pair condition on the same media.
    Text-only records keep the ``StreamingDPODataset`` dict."""

    def __init__(self, data_path, tokenizer, max_length=4096, *, image_size=224,
                 audio_sample_rate=16000, max_audio_seconds=30.0, media_error_policy="warn"):
        super().__init__(data_path, tokenizer, max_length=max_length)
        self.media_kwargs = dict(image_size=int(image_size), audio_sample_rate=int(audio_sample_rate),
                                 max_audio_seconds=float(max_audio_seconds),
                                 media_error_policy=str(media_error_policy))
        self.modality_summary = dataset_modality_summary(data_path)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        found = record_media(self._read_record(index), **self.media_kwargs)
        if found is not None:
            item["modality_inputs"], item["modality_mask"] = found
        return item


PREFERENCE_KEYS = ("x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected")


def preference_collate(batch):
    """Stack DPO/ORPO pairs; media (one set per pair) is batched with the
    ``multimodal_collate`` rules, so mixed text/media pairs keep exact masks."""
    out = {key: torch.stack([item[key] for item in batch]) for key in PREFERENCE_KEYS}
    if any(item.get("modality_inputs") for item in batch):
        stub = torch.zeros(1, dtype=torch.long)
        media = multimodal_collate([
            {"input_ids": stub, "labels": stub, "modality_inputs": item["modality_inputs"],
             "modality_mask": item.get("modality_mask")} if item.get("modality_inputs") else (stub, stub)
            for item in batch])
        out["modality_inputs"], out["modality_mask"] = media.get("modality_inputs", {}), media["modality_mask"]
    return out
