"""Training telemetry for MLX runs: heartbeat, per-tensor statistics, routing.

Design rules (the run is the product, telemetry must not cost it):
* nothing here changes training numerics — it only reads parameters,
  accumulated gradients and router probabilities that already exist;
* every statistic for one telemetry step is reduced on device and brought to
  the host with **one** ``mx.eval`` + ``tolist`` (no per-tensor ``.item()``);
* FP32 casts appear only inside these diagnostic reductions (sums of squares
  of BF16 values must not accumulate in BF16); they run every
  ``tensor_stats_every_s`` seconds, never in the training step itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten

# Name fragment -> component group (first match wins).
GROUP_RULES = (
    (".mlp.shared_experts.", "shared_expert"),
    (".mlp.experts.", "experts"),
    (".mlp.gate.", "router"),
    (".self_attn.", "attention"),
    ("lookahead_router", "lookahead"),
    ("recurrent_controller", "recurrent"),
    ("embed_tokens", "embedding"),
    ("lm_head", "head"),
    ("norm", "norm"),
)


def component(name: str) -> str:
    for frag, group in GROUP_RULES:
        if frag in name:
            return group
    return "other"


def _sig(v: float, digits: int = 4) -> float:
    return float(f"{v:.{digits}g}")


def _atomic_json(path: str, payload: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def tensor_stats(params: dict, grads: dict | None = None, before: dict | None = None,
                 grad_scale: float = 1.0) -> dict:
    """Per-tensor RMS / mean / |max|, gradient RMS and relative update size.

    ``params``/``grads``/``before`` are flat ``{name: array}`` dicts (``grads``
    is the FP32 accumulation *before* clipping; ``grad_scale`` turns the sum
    over micro-batches into the mean).  ``before`` holds the parameter arrays
    from before the optimizer update (MLX arrays are immutable, so keeping the
    references costs no copy).
    """
    names = list(params)
    zero = mx.array(0.0, dtype=mx.float32)
    rows = []
    for name in names:
        p = params[name].astype(mx.float32)
        cells = [p.sum(), (p * p).sum(), mx.abs(p).max()]
        g = grads.get(name) if grads else None
        cells.append(mx.square(g.astype(mx.float32)).sum() if g is not None else zero)
        b = before.get(name) if before else None
        cells.append(mx.square(p - b.astype(mx.float32)).sum() if b is not None else zero)
        rows.append(mx.stack(cells))
    table = mx.stack(rows) if rows else mx.zeros((0, 5))
    mx.eval(table)
    values = table.tolist()

    tensors, groups = [], {}
    tot = {"numel": 0, "sumsq": 0.0, "grad_sumsq": 0.0, "delta_sumsq": 0.0}
    for name, (s, sq, amax, gsq, dsq) in zip(names, values):
        arr = params[name]
        n = max(1, int(arr.size))
        mean = s / n
        var = max(0.0, sq / n - mean * mean)
        gsq *= grad_scale * grad_scale
        entry = {"name": name, "group": component(name), "shape": list(arr.shape), "dtype": str(arr.dtype).split(".")[-1],
                 "rms": _sig((sq / n) ** 0.5), "mean": _sig(mean), "std": _sig(var ** 0.5), "absmax": _sig(amax)}
        if grads:
            entry["grad_rms"] = _sig((gsq / n) ** 0.5)
        if before:
            entry["update_ratio"] = _sig((dsq ** 0.5) / max(sq ** 0.5, 1e-30))
        tensors.append(entry)
        acc = groups.setdefault(entry["group"], {"tensors": 0, "numel": 0, "sumsq": 0.0, "grad_sumsq": 0.0, "delta_sumsq": 0.0})
        for bucket in (acc, tot):
            bucket["numel"] += n
            bucket["sumsq"] += sq
            bucket["grad_sumsq"] += gsq
            bucket["delta_sumsq"] += dsq
        acc["tensors"] += 1

    def summarize(b):
        out = {"numel": b["numel"], "rms": _sig((b["sumsq"] / max(1, b["numel"])) ** 0.5)}
        if grads:
            out["grad_norm"] = _sig(b["grad_sumsq"] ** 0.5)
        if before:
            out["update_ratio"] = _sig(b["delta_sumsq"] ** 0.5 / max(b["sumsq"] ** 0.5, 1e-30))
        if "tensors" in b:
            out["tensors"] = b["tensors"]
        return out

    return {"tensors": tensors, "groups": {k: summarize(v) for k, v in sorted(groups.items())},
            "stats": summarize(tot), "grad_note": "pre-clip, mean over micro-batches" if grads else None}


def expert_table(tensors: list[dict]) -> dict:
    """``[layer][expert]`` weight RMS and gradient RMS (gate/up/down pooled)."""
    cells: dict[int, dict[int, list]] = {}
    for t in tensors:
        parts = t["name"].split(".")
        if ".mlp.experts." not in t["name"]:
            continue
        try:
            layer = int(parts[parts.index("layers") + 1])
            expert = int(parts[parts.index("experts") + 1])
        except (ValueError, IndexError):
            continue
        n = 1
        for d in t["shape"]:
            n *= d
        slot = cells.setdefault(layer, {}).setdefault(expert, [0, 0.0, 0.0])
        slot[0] += n
        slot[1] += t["rms"] ** 2 * n
        slot[2] += t.get("grad_rms", 0.0) ** 2 * n
    layers = sorted(cells)
    width = max((max(v) + 1 for v in cells.values()), default=0)
    def grid(i):
        return [[_sig((cells[l].get(e, [1, 0.0, 0.0])[i] / max(1, cells[l].get(e, [1])[0])) ** 0.5) for e in range(width)]
                for l in layers]
    return {"layers": layers, "weight_rms": grid(1), "grad_rms": grid(2)} if layers else {}


def routing_snapshot(model) -> dict | None:
    """Per virtual layer: load share, mean router probability and entropy of
    the last forward's dispatches (``[V, E]``), with ``L<layer>·<iteration>``
    labels so recurrent iterations of one physical layer sit side by side."""
    from chronos.mlx.moe import ChronosMLXMOE
    probs, labels = [], []
    for li, layer in enumerate(getattr(model, "layers", [])):
        moe = getattr(layer, "mlp", None)
        if isinstance(moe, ChronosMLXMOE):
            for it, p in enumerate(moe.iteration_router_probs):
                probs.append(p)
                labels.append(f"L{li}·{it}")
    if not probs:
        return None
    top_k = int(model.config.num_experts_per_tok)
    router = mx.stack(probs, axis=2)                                      # [B, S, V, E]
    E, V = router.shape[-1], router.shape[2]
    flat = router.reshape(-1, V, E).astype(mx.float32)
    top = mx.argpartition(-flat, kth=top_k - 1, axis=-1)[..., :top_k]
    load = (top[..., None] == mx.arange(E)).astype(mx.float32).sum(-2).mean(0) / top_k
    importance = flat.mean(0)
    entropy = -(flat * mx.log(mx.maximum(flat, 1e-9))).sum(-1).mean(0)
    mx.eval(load, importance, entropy)
    r4 = lambda grid: [[round(v, 4) for v in row] for row in grid]
    return {"labels": labels, "experts": E, "top_k": top_k, "load": r4(load.tolist()),
            "importance": r4(importance.tolist()), "entropy": [round(v, 4) for v in entropy.tolist()],
            "entropy_max": round(math.log(E), 4), "source": "last micro-batch of the step"}


def code_fingerprint(root: str | None = None) -> dict:
    """SHA-256 over the kernel and trainer sources this process imported, so
    the monitor can tell which code revision a long run is executing."""
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    files = []
    for sub in ("chronos/mlx", "chronos/mlx/training", "chronos/model"):
        folder = os.path.join(root, sub)
        if os.path.isdir(folder):
            files += sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".py"))
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as fh:
            digest.update(os.path.relpath(path, root).encode() + b"\0" + fh.read())
    return {"sha256": digest.hexdigest()[:16], "files": len(files)}


def runtime_info(model, spec, precision) -> dict:
    import numpy as np
    params = dict(tree_flatten(model.parameters()))
    by_group: dict[str, int] = {}
    for name, arr in params.items():
        by_group[component(name)] = by_group.get(component(name), 0) + int(arr.size)
    try:
        device = mx.device_info()
    except Exception:  # noqa: BLE001 - device info is optional metadata
        device = {}
    return {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "host": socket.gethostname(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": {"mlx": getattr(mx, "__version__", "?"), "numpy": np.__version__},
        "device": {k: device.get(k) for k in ("device_name", "architecture", "memory_size",
                                              "max_recommended_working_set_size") if k in device},
        "default_device": str(mx.default_device()),
        "precision": precision.as_dict() if precision is not None else {"compute": spec.dtype},
        "parameters": {"total": sum(by_group.values()), "by_group": dict(sorted(by_group.items()))},
        "phases": [{"name": p.name, "accum": max(1, p.tokens_per_step // p.micro_tokens),
                    "micro_rows": max(1, p.micro_tokens // p.seq_len) if p.name == "pretrain" else None,
                    "micro_tokens": p.micro_tokens, "tokens_per_step": p.tokens_per_step} for p in spec.phases],
        "code": code_fingerprint(),
    }


class Heartbeat:
    """``status.json``: the run's own view of itself, rewritten atomically."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "status.json")
        self.fields: dict = {"pid": os.getpid(), "host": socket.gethostname(), "state": "starting",
                             "started": time.time(), "error": None}

    def beat(self, state: str | None = None, **fields) -> dict:
        if state is not None:
            self.fields["state"] = state
        self.fields.update(fields)
        self.fields["updated"] = time.time()
        self.fields["updated_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            _atomic_json(self.path, self.fields)
        except OSError:
            pass                    # telemetry must never stop training
        return self.fields
