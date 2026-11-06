"""Inference-only int8 KV-cache storage (MLX), mirroring chronos.model.kv_quant.

Symmetric int8 with one FP32 scale per token per ``group_size`` channels of
the last axis.  Each token is quantized once when appended and dequantized
into the compute dtype when attention reads the cache.
"""
from __future__ import annotations

import mlx.core as mx


class QuantizedKVMLX:
    __slots__ = ("data", "scale", "group_size", "dtype")

    def __init__(self, data: mx.array, scale: mx.array, group_size: int, dtype):
        self.data, self.scale, self.group_size, self.dtype = data, scale, int(group_size), dtype

    @classmethod
    def quantize(cls, x: mx.array, group_size: int = 32) -> "QuantizedKVMLX":
        d = int(x.shape[-1])
        g = int(group_size) if group_size and d % int(group_size) == 0 else d
        groups = x.astype(mx.float32).reshape(*x.shape[:-1], d // g, g)
        scale = mx.maximum(mx.max(mx.abs(groups), axis=-1, keepdims=True), 1e-12) / 127.0
        data = mx.clip(mx.round(groups / scale), -127, 127).astype(mx.int8).reshape(x.shape)
        return cls(data, scale, g, x.dtype)

    def append(self, x: mx.array) -> "QuantizedKVMLX":
        new = QuantizedKVMLX.quantize(x, self.group_size)
        return QuantizedKVMLX(mx.concatenate([self.data, new.data], axis=1),
                              mx.concatenate([self.scale, new.scale], axis=1), self.group_size, self.dtype)

    def dequantize(self, dtype=None) -> mx.array:
        d = int(self.data.shape[-1])
        groups = self.data.reshape(*self.data.shape[:-1], d // self.group_size, self.group_size)
        return (groups.astype(mx.float32) * self.scale).reshape(self.data.shape).astype(dtype or self.dtype)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        if not (isinstance(index, tuple) and len(index) == 2 and index[0] == slice(None)
                and isinstance(index[1], slice) and index[1].step in (None, 1)):
            raise TypeError("QuantizedKVMLX supports only [:, start:stop] token slicing")
        return QuantizedKVMLX(self.data[index], self.scale[index], self.group_size, self.dtype)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes) + int(self.scale.nbytes)

    def arrays(self):
        return (self.data, self.scale)


def kv_store(x: mx.array, past, group_size: int) -> QuantizedKVMLX:
    """Append ``x`` to an int8 cache (``past`` may be None, int8 or float)."""
    if isinstance(past, QuantizedKVMLX):
        return past.append(x)
    if past is not None:
        x = mx.concatenate([past.astype(x.dtype), x], axis=1)
    return QuantizedKVMLX.quantize(x, group_size)


def kv_read(entry, dtype) -> mx.array:
    return entry.dequantize(dtype) if isinstance(entry, QuantizedKVMLX) else entry


def entry_nbytes(entry) -> int:
    total = 0
    for item in entry or ():
        if isinstance(item, QuantizedKVMLX):
            total += item.nbytes
        elif isinstance(item, mx.array):
            total += int(item.nbytes)
    return total


def cache_arrays(cache):
    """Flatten a cache list into arrays (for mx.eval / async_eval)."""
    out = []
    for entry in cache or ():
        for item in entry or ():
            if isinstance(item, QuantizedKVMLX):
                out.extend(item.arrays())
            elif isinstance(item, mx.array):
                out.append(item)
    return out
