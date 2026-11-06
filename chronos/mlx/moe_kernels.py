"""Sparse expert execution for MLX: sorted ``gather_mm`` grouped SwiGLU.

Only the top-k selected experts run for each token (no dense E-way
compute).  Known backend defects and how they are handled:

* mlx 0.31.x on Apple M5 GPUs lacks fp32 GatherMM Metal kernels and returns
  wrong bf16 results (measured 2026-11-05, relative gradient error ~1.0).
* mlx 0.31.1 on M1 Max: a ``[:, None, :]`` row view feeding a sorted
  gather and a ``squeeze`` before the inverse permutation gave wrong forward
  values / down-projection gradients (performance branch, 2026-11-06).
  ``grouped_swiglu`` therefore uses explicit ``reshape(-1, 1, H)`` /
  ``reshape(-1, H)`` layouts (measured identical to the dense oracle on mlx
  0.32.4 / M5 at N=8192, E=36, H=384, I=160).
* CPU GatherMM is float32-only.

Instead of trusting a version number, :func:`sparse_dispatch_ok` runs a
one-time self-check per (device, dtype) against the dense oracle (value and
all gradients, > 64 sorted assignments) and falls back to the masked-dense
path if it fails.  ``dense_reference_swiglu`` is that oracle.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

MIN_MLX_GATHER_MM = (0, 32, 4)     # first version validated on M5 (fp32 + bf16 Metal kernels)
_SELF_CHECK: dict[tuple[str, str], bool] = {}


def mlx_version_tuple() -> tuple[int, int, int]:
    parts = (mx.__version__.split("+")[0].split(".") + ["0", "0"])[:3]
    return tuple(int("".join(ch for ch in p if ch.isdigit()) or 0) for p in parts)


def gather_mm_supported() -> bool:
    """``gather_mm`` exists and passes the self-check on the default device (fp32)."""
    return sparse_dispatch_ok(mx.float32)


def sparse_dispatch_ok(dtype) -> bool:
    """Whether sorted ``gather_mm`` dispatch is correct for ``dtype`` on the
    current default device (memoised self-check; ~10 ms once per process)."""
    if not hasattr(mx, "gather_mm"):
        return False
    device = mx.default_device()
    if device.type == mx.DeviceType.cpu and not cpu_gather_mm_dtype_ok(dtype):
        return False
    key = (str(device), str(dtype))
    if key not in _SELF_CHECK:
        _SELF_CHECK[key] = _self_check(dtype)
    return _SELF_CHECK[key]


def _self_check(dtype) -> bool:
    """Sorted grouped SwiGLU vs the masked-dense oracle on this device/dtype.

    192 assignments (above any small-batch shortcut), all four weight/input
    gradients plus the routing-weight gradient.  Reported defects are O(0.1-1)
    relative errors; TF32-class / bf16 rounding is O(1e-3-1e-2).
    """
    try:
        key = mx.random.key(20261106)
        k = mx.random.split(key, 6)
        N, K, E, H, I = 96, 2, 6, 32, 24
        x = mx.random.normal((N, H), key=k[0]).astype(dtype)
        wg = (mx.random.normal((E, I, H), key=k[1]) * 0.2).astype(dtype)
        wu = (mx.random.normal((E, I, H), key=k[2]) * 0.2).astype(dtype)
        wd = (mx.random.normal((E, H, I), key=k[3]) * 0.2).astype(dtype)
        scores = mx.softmax(mx.random.normal((N, E), key=k[4]), axis=-1)
        idx = mx.argpartition(-scores, kth=K - 1, axis=-1)[:, :K]
        w = mx.take_along_axis(scores, idx, axis=-1).astype(dtype)
        cot = mx.random.normal((N, H), key=k[5])

        def run(fn):
            def loss(x, wg, wu, wd, w):
                return (fn(x, idx, w, wg, wu, wd).astype(mx.float32) * cot).sum()
            v, g = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(x, wg, wu, wd, w)
            mx.eval(v, g)
            return [v] + list(g)

        tol = 2e-2 if dtype == mx.float32 else 8e-2
        for a, b in zip(run(grouped_swiglu), run(dense_reference_swiglu)):
            a, b = a.astype(mx.float32).reshape(-1), b.astype(mx.float32).reshape(-1)
            err = mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-12)
            if not bool(mx.isfinite(err).item()) or float(err.item()) > tol:
                return False
        return True
    except Exception:
        return False


def stack_expert_weights(experts) -> tuple[mx.array, mx.array, mx.array]:
    """Stack per-expert SwiGLU weights into [E, out, in] arrays (differentiable)."""
    return (
        mx.stack([e.gate_proj.weight for e in experts]),
        mx.stack([e.up_proj.weight for e in experts]),
        mx.stack([e.down_proj.weight for e in experts]),
    )


def cpu_gather_mm_dtype_ok(dtype) -> bool:
    """MLX CPU GatherMM is float32-only (mlx 0.32.4)."""
    return dtype == mx.float32


def _gather_mm(a: mx.array, b: mx.array, idx: mx.array) -> mx.array:
    return mx.gather_mm(a, b, rhs_indices=idx, sorted_indices=True)


def grouped_swiglu(x: mx.array, topk_idx: mx.array, topk_weight: mx.array,
                   w_gate: mx.array, w_up: mx.array, w_down: mx.array,
                   fallback: tuple[mx.array, mx.array] | None = None) -> mx.array:
    """Σ_k w[n,k]·SwiGLU_{idx[n,k]}(x[n]) with one sorted gather_mm per projection.

    x: [N, H]; topk_idx/topk_weight: [N, K]; w_gate/w_up: [E, I, H]; w_down: [E, H, I].
    ``fallback=(available[E] bool, shared_out[N, H])`` replaces the output of
    every unavailable expert with the shared-expert output (cache-miss
    simulation, identical to the PyTorch masked path).
    """
    N, K = topk_idx.shape
    H = x.shape[-1]
    flat = mx.stop_gradient(topk_idx).reshape(-1)
    order = mx.stop_gradient(mx.argsort(flat))
    sorted_experts = flat[order]
    # Explicit reshapes (not ``[:, None, :]`` / ``squeeze``): see module doc.
    rows = x[order // K].reshape(-1, 1, H)                 # [N·K, 1, H]
    gate = _gather_mm(rows, w_gate.swapaxes(-1, -2), sorted_experts)
    up = _gather_mm(rows, w_up.swapaxes(-1, -2), sorted_experts)
    out = _gather_mm(nn.silu(gate) * up, w_down.swapaxes(-1, -2), sorted_experts).reshape(-1, H)
    out = out[mx.stop_gradient(mx.argsort(order))].reshape(N, K, H)
    if fallback is not None:
        available, shared = fallback
        live = mx.stop_gradient(available[topk_idx])[..., None]          # [N, K, 1]
        out = mx.where(live, out, shared[:, None, :].astype(out.dtype))
    return (out * topk_weight[..., None].astype(out.dtype)).sum(axis=1)


def dense_reference_swiglu(x: mx.array, topk_idx: mx.array, topk_weight: mx.array,
                           w_gate: mx.array, w_up: mx.array, w_down: mx.array,
                           fallback=None) -> mx.array:
    """Masked-dense oracle: runs every expert on every token (tests only)."""
    out = mx.zeros(x.shape, dtype=x.dtype)
    for e in range(w_gate.shape[0]):
        y = (nn.silu(x @ w_gate[e].T) * (x @ w_up[e].T)) @ w_down[e].T
        if fallback is not None and not bool(fallback[0][e].item()):
            y = fallback[1].astype(y.dtype)
        weight = mx.where(topk_idx == e, topk_weight, 0).sum(axis=-1, keepdims=True)
        out = out + y * weight.astype(y.dtype)
    return out


def training_availability(num_experts: int, miss_prob: float) -> mx.array:
    """Random per-call expert availability; at least one expert stays live."""
    keep = mx.random.uniform(shape=(num_experts,)) >= miss_prob
    forced = mx.arange(num_experts) == mx.random.randint(0, num_experts, ())
    return mx.where(mx.any(keep), keep, forced)
