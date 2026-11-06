"""Inference-only int8 KV-cache storage (Torch).

``QuantizedKV`` stores a ``[B, T, *, D]`` cache tensor as symmetric int8
with one FP32 scale per token and ``group_size`` channels along the last
axis.  Each token is quantized exactly once, when it is appended; the cache
is dequantized into the compute dtype when attention reads it (one rounding
per read, no FP32 activation copy).  FP32 scales keep tiny latent norms from
underflowing and cost 4/group_size bytes per channel (12.5 % at 32).

The object mimics the parts of the tensor interface the cache plumbing uses
(``shape``, ``[:, :n]`` slicing, ``numel``/``element_size`` byte counting),
so ``truncate_cache`` rollback and sequence-length bookkeeping stay generic.
"""
from __future__ import annotations

import torch

KV_CACHE_DTYPES = ("auto", "int8")


def resolve_kv_cache_dtype(config) -> str:
    value = str(getattr(config, "kv_cache_dtype", "auto") or "auto").lower()
    if value in ("fp16", "bf16", "fp32", "compute", "none"):
        value = "auto"
    if value not in KV_CACHE_DTYPES:
        raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}, got {value!r}")
    return value


class QuantizedKV:
    __slots__ = ("data", "scale", "group_size", "dtype")

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, group_size: int, dtype: torch.dtype):
        self.data, self.scale, self.group_size, self.dtype = data, scale, int(group_size), dtype

    # ---- construction -----------------------------------------------------
    @classmethod
    def quantize(cls, x: torch.Tensor, group_size: int = 32) -> "QuantizedKV":
        d = int(x.shape[-1])
        g = int(group_size) if group_size and d % int(group_size) == 0 else d
        groups = x.float().reshape(*x.shape[:-1], d // g, g)
        scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
        data = torch.round(groups / scale).clamp_(-127, 127).to(torch.int8).reshape(x.shape)
        return cls(data, scale, g, x.dtype)

    def append(self, x: torch.Tensor) -> "QuantizedKV":
        new = QuantizedKV.quantize(x, self.group_size)
        return QuantizedKV(torch.cat([self.data, new.data], dim=1), torch.cat([self.scale, new.scale], dim=1),
                           self.group_size, self.dtype)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        d = int(self.data.shape[-1])
        groups = self.data.reshape(*self.data.shape[:-1], d // self.group_size, self.group_size)
        return (groups.float() * self.scale).reshape(self.data.shape).to(dtype or self.dtype)

    # ---- tensor-like surface used by cache plumbing -----------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    def __getitem__(self, index):
        if not (isinstance(index, tuple) and len(index) == 2 and index[0] == slice(None)
                and isinstance(index[1], slice) and index[1].step in (None, 1)):
            raise TypeError("QuantizedKV supports only [:, start:stop] token slicing")
        return QuantizedKV(self.data[index], self.scale[index], self.group_size, self.dtype)

    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size() + self.scale.numel() * self.scale.element_size()

    def numel(self) -> int:          # byte accounting helpers treat numel*element_size as bytes
        return self.nbytes()

    @staticmethod
    def element_size() -> int:
        return 1


def kv_store(x: torch.Tensor, past, quantize: bool, group_size: int):
    """Append ``x`` to ``past`` (tensor, QuantizedKV or None) in the cache format."""
    if quantize:
        if isinstance(past, QuantizedKV):
            return past.append(x)
        q = QuantizedKV.quantize(x, group_size)
        if past is not None:     # a float prefix (e.g. a cache built before int8 was enabled)
            q = QuantizedKV.quantize(torch.cat([past.to(x.dtype), x], dim=1), group_size)
        return q
    if isinstance(past, QuantizedKV):
        past = past.dequantize(x.dtype)
    return x if past is None else torch.cat([past, x], dim=1)


def kv_read(entry, dtype: torch.dtype) -> torch.Tensor:
    return entry.dequantize(dtype) if isinstance(entry, QuantizedKV) else entry


def entry_nbytes(entry) -> int:
    """Bytes held by one cache entry (tensors, QuantizedKV, ints)."""
    total = 0
    for item in entry or ():
        if isinstance(item, QuantizedKV):
            total += item.nbytes()
        elif isinstance(item, torch.Tensor):
            total += item.numel() * item.element_size()
    return total
