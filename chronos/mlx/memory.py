"""MLX buffer-cache limit, enforced by the kernel for every MLX model.

MLX keeps freed Metal buffers in a cache for reuse.  Its default cache
limit equals the memory limit, so on a unified-memory Mac the cache can grow
until macOS pages and training slows down (measured on a 64 GB machine:
footprint 55.8 → 13.6 GB and ~2.4k → ~5.2k tok/s after an 8 GiB limit).
The limit bounds only reclaimable cache; weights, activations and gradients
are active memory and are not affected.

Default: ``min(physical memory / 8, 16 GiB)``.  A caller value (recipe
``extra.mlx_cache_limit_gb``, ``CHRONOS_MLX_CACHE_LIMIT_GB``) may lower it,
never raise it above that ceiling.  Values are GiB (×1024³); metrics report
decimal GB (8 GiB shows as 8.59 GB).
"""
from __future__ import annotations

import os

GIB = 1024 ** 3
CEILING_GIB = 16.0
FRACTION = 1.0 / 8.0
ENV = "CHRONOS_MLX_CACHE_LIMIT_GB"

_applied: dict | None = None


def physical_memory_bytes() -> int:
    """Total physical memory (MLX device info, then sysctl/sysconf)."""
    try:
        import mlx.core as mx
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        size = int(info.get("memory_size") or 0)
        if size > 0:
            return size
    except Exception:  # noqa: BLE001 - fall back to the OS
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def ceiling_bytes(total: int | None = None) -> int:
    """``min(total / 8, 16 GiB)``; 16 GiB when the total is unknown."""
    total = physical_memory_bytes() if total is None else int(total)
    cap = int(CEILING_GIB * GIB)
    return cap if total <= 0 else min(int(total * FRACTION), cap)


def resolve_cache_limit(requested_gb=None, total: int | None = None) -> tuple[int, str]:
    """``(bytes, source)``: the limit to apply.  ``requested_gb`` (GiB, > 0) or
    the environment variable may lower the ceiling; larger values are clamped."""
    ceiling = ceiling_bytes(total)
    raw, source = requested_gb, "recipe"
    if raw in (None, ""):
        raw, source = os.environ.get(ENV), "env"
    if raw in (None, ""):
        return ceiling, "default min(RAM/8, 16 GiB)"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"mlx_cache_limit_gb must be a positive number of GiB, got {raw!r}") from None
    if not value > 0:
        raise ValueError(f"mlx_cache_limit_gb must be > 0 GiB, got {raw!r}")
    want = int(value * GIB)
    if want > ceiling:
        return ceiling, f"{source} {value:g} GiB clamped to min(RAM/8, 16 GiB)"
    return want, source


def apply_cache_limit(requested_gb=None, *, log=None, force: bool = False) -> dict | None:
    """Set the MLX cache limit once per process (again with ``force`` or a new
    request).  Returns ``{"limit_gb", "previous_gb", "ceiling_gb", "source"}``
    in GiB, or ``None`` without MLX."""
    global _applied
    try:
        import mlx.core as mx
    except ModuleNotFoundError:
        return None
    if _applied is not None and not force and requested_gb in (None, "", _applied.get("requested")):
        return _applied
    limit, source = resolve_cache_limit(requested_gb)
    previous = int(mx.set_cache_limit(limit))
    _applied = {"limit_gb": round(limit / GIB, 3), "previous_gb": round(previous / GIB, 3),
                "ceiling_gb": round(ceiling_bytes() / GIB, 3), "source": source,
                "requested": requested_gb}
    if log is not None:
        log(f"[run] MLX cache limit: {limit / GIB:g} GiB (previous {previous / GIB:.1f} GiB; {source})")
    return _applied


def applied() -> dict | None:
    """The last applied limit (``None`` before the first MLX model)."""
    return None if _applied is None else {k: v for k, v in _applied.items() if k != "requested"}
