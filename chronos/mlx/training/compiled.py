"""``mx.compile`` for one MLX training micro step (loss + gradients).

Measured on mlx 0.32.4 (see the 0.3.2a0 CHANGELOG): the compiled step gives
the eager loss and gradients up to FP32 summation order (<= 1.7e-5 relative
on CPU FP32), consumes the RNG stream identically, and runs 1.08-1.18x
faster on the 50M en->zh config because elementwise chains (RMSNorm,
SwiGLU, softmax/CE, router regularizers) fuse into fewer kernels.

What compiling needs here, and how it is handled:

* **Mutable state.**  Model parameters and ``mx.random.state`` are declared
  as compile inputs *and* outputs, so a step reads the live weights and the
  dropout / cache-miss draws advance the global RNG exactly as eagerly.
* **Python decisions are part of the graph.**  The executed recurrent depth,
  whether ACT mixing is active, the training flag, input shapes and dtypes
  all change the traced graph.  Each distinct signature gets its own
  compiled function; ``depth`` is sampled once per step outside the trace
  and pinned for the call (``depth_override``-free: see ``_pinned_depth``).
* **Side outputs.**  The loss reads, and telemetry logs, arrays the forward
  pass stores on modules (router probabilities, loss components, recurrent
  diagnostics).  Inside a trace those are placeholders, so they are returned
  as explicit outputs and written back after the call.
* **Fallback.**  Anything else (multimodal inputs, a lazy expert, an
  unexpected trace error) runs eagerly; ``mode="auto"`` logs why once.
"""
from __future__ import annotations

import contextlib
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

# ── side outputs the regularized loss and the telemetry read ──────────────

_MODEL_SIDE = ("_last_loss_components", "_last_stage_metrics", "last_recurrent_state", "_last_lookahead_probs")


def _moes(model):
    from chronos.mlx.moe import ChronosMLXMOE
    return [layer.mlp for layer in getattr(model, "layers", []) if isinstance(getattr(layer, "mlp", None), ChronosMLXMOE)]


def _split(obj, arrays: list):
    """Array leaves -> index placeholders; everything else kept (bools, ints)."""
    if isinstance(obj, mx.array):
        arrays.append(obj)
        return ("__array__", len(arrays) - 1)
    if isinstance(obj, dict):
        return {k: _split(v, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_split(v, arrays) for v in obj)
    return obj


def _join(spec, arrays):
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "__array__":
        return arrays[spec[1]]
    if isinstance(spec, dict):
        return {k: _join(v, arrays) for k, v in spec.items()}
    if isinstance(spec, (list, tuple)):
        return type(spec)(_join(v, arrays) for v in spec)
    return spec


def _side_get(model) -> dict:
    return {"model": {k: getattr(model, k, None) for k in _MODEL_SIDE},
            "moe": [[m._last_router_probs, list(m._iteration_router_probs), m._shared_topk_idx] for m in _moes(model)]}


def _side_set(model, side: dict) -> None:
    for k, v in side["model"].items():
        object.__setattr__(model, k, v)
    for m, (last, probs, topk) in zip(_moes(model), side["moe"]):
        object.__setattr__(m, "_last_router_probs", last)
        object.__setattr__(m, "_iteration_router_probs", list(probs))
        object.__setattr__(m, "_shared_topk_idx", topk)


def _state_slots(node, out: list) -> list:
    """``(container, key, array)`` for every array leaf of a module's state tree."""
    for key, value in list(node.items() if isinstance(node, dict) else enumerate(node)):
        if isinstance(value, mx.array):
            out.append((node, key, value))
        elif isinstance(value, (dict, list)):
            _state_slots(value, out)
    return out


def _capture(model) -> dict:
    """Weights, RNG key and side outputs as they are before a first trace."""
    return {"slots": _state_slots(model.state, []),
        "rng": [int(w) for w in mx.random.state[0].tolist()],
        "side": _side_get(model)}


def _restore(model, snap: dict) -> None:
    """Undo what a failed trace left behind: the saved arrays go back into the module tree, the
    RNG is re-seeded from the saved key words (the stream continues exactly), side outputs are reset."""
    for node, key, value in snap["slots"]:
        node[key] = value
    hi, lo = snap["rng"]
    mx.random.seed((hi << 32) | lo)
    _side_set(model, snap["side"])


def _static(value):
    """Hashable stand-in for a non-array argument, or ``None`` when unsupported."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("v", value)
    if isinstance(value, nn.Module):
        return ("m", id(value))
    return None


def env_mode(default: str = "auto") -> str:
    import os
    mode = (os.environ.get("CHRONOS_MLX_COMPILE", "") or default).strip().lower()
    return {"1": "on", "true": "on", "0": "off", "false": "off", "none": "off"}.get(mode, mode)


class CompiledStep:
    """``nn.value_and_grad(model, loss_fn)`` with a compiled graph per signature.

    Call it exactly like the eager ``grad_fn``: ``loss, grads = step(*args)``.
    ``args`` may mix arrays, ``None``/scalars and the model itself.  After the
    call the model's side outputs (router probabilities, loss components,
    recurrent diagnostics) hold real arrays, as after an eager call.

    ``mode``: ``"auto"`` compiles and falls back to eager (logging the
    reason once) when a signature cannot be compiled; ``"on"`` raises
    instead; ``"off"`` is the eager ``grad_fn``.
    """

    def __init__(self, model, loss_fn: Callable, *, mode: str = "auto", log=print, max_graphs: int = 128):
        from collections import OrderedDict
        mode = str(mode or "auto").lower()
        if mode not in {"auto", "on", "off"}:
            raise ValueError(f"mlx compile mode must be auto|on|off, got {mode!r}")
        self.model, self.mode, self.log, self.max_graphs = model, mode, log, int(max_graphs)
        self.eager = nn.value_and_grad(model, loss_fn)
        self._graphs: OrderedDict = OrderedDict()
        self._specs: dict = {}
        self._failed: set = set()
        self._logged: set = set()
        self.stats = {"compiled": 0, "eager": 0, "traces": 0}

    # -- eligibility ---------------------------------------------------------
    def _eager_reason(self, args) -> str | None:
        if self.mode == "off":
            return "disabled"
        from chronos.mlx.moe import LazyFeedForwardMLX
        model = self.model
        if getattr(model, "multimodal_fusion", None) is not None:
            return "multimodal model (encoders run eagerly)"
        if any(isinstance(e, LazyFeedForwardMLX) for m in _moes(model) for e in m.experts):
            return "lazy (paged) experts"
        if any(not isinstance(a, mx.array) and _static(a) is None for a in args):
            return "non-array batch argument"
        return None

    def _note(self, reason: str) -> None:
        if reason not in self._logged:
            self._logged.add(reason)
            self.log(f"[mlx-compile] eager step: {reason}")

    def _plan(self, args):
        """(graph key, depth to pin) for this call; Python decisions included."""
        model, cfg = self.model, self.model.config
        depth = adaptive = None
        if cfg.recurrent_enabled and cfg.recurrent_depth > 1:
            controller = model.recurrent_controller
            depth = int(controller.execution_depth(cfg, model.training))
            adaptive = bool(controller.adaptive_active(cfg, model.training))
        sig = tuple(("a", tuple(a.shape), str(a.dtype)) if isinstance(a, mx.array) else _static(a) for a in args)
        # Plain config values decide branches too (and callers may toggle them).
        conf = tuple(sorted((k, v) for k, v in vars(cfg).items()
                            if v is None or isinstance(v, (bool, int, float, str))))
        return (sig, bool(model.training), depth, adaptive, conf), depth

    def _prewarm(self, args) -> None:
        """Grow RoPE tables before tracing, so no module stores a traced table."""
        length = max([a.shape[1] for a in args if isinstance(a, mx.array) and a.ndim == 2] or [0])
        if length:
            for _, module in self.model.named_modules():
                grow = getattr(module, "_ensure_rope_cache", None)
                if grow is not None:
                    grow(length)

    # -- compiled graph ------------------------------------------------------
    def _build(self, key, args):
        model = self.model
        spec = [None if isinstance(a, mx.array) else a for a in args]
        slots = [i for i, a in enumerate(args) if isinstance(a, mx.array)]
        meta: dict = {}
        state = [model.state, mx.random.state]

        def step(*arrays):
            full = list(spec)
            for i, a in zip(slots, arrays):
                full[i] = a
            loss, grads = self.eager(*full)
            flat: list = []
            meta["side"] = _split(_side_get(model), flat)
            return loss, grads, flat

        fn = mx.compile(step, inputs=state, outputs=state)
        return fn, meta

    @contextlib.contextmanager
    def _pin(self, depth):
        controller = getattr(self.model, "recurrent_controller", None)
        if controller is None or depth is None:
            yield
            return
        object.__setattr__(controller, "_pinned_depth", depth)
        try:
            yield
        finally:
            object.__setattr__(controller, "_pinned_depth", None)

    def __call__(self, *args):
        reason = self._eager_reason(args)
        key = depth = None
        if reason is None:
            key, depth = self._plan(args)
            if key in self._failed:
                reason = "signature failed to compile earlier"
        if reason is not None:
            if reason != "disabled":
                self._note(reason)
            self.stats["eager"] += 1
            with self._pin(depth):
                return self.eager(*args)
        hit = self._graphs.get(key)
        fresh = hit is None
        if fresh:
            self._prewarm(args)
            hit = self._build(key, args)
            self._graphs[key] = hit
            self.stats["traces"] += 1
            while len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
        fn, meta = hit
        arrays = [a for a in args if isinstance(a, mx.array)]
        # A trace that fails half way leaves tracers in the weights, the RNG key and the side outputs
        # (mx.compile swaps them back only on success), so snapshot before tracing and restore on failure.
        snapshot = _capture(self.model) if fresh else None
        try:
            with self._pin(depth):
                loss, grads, flat = fn(*arrays)
        except Exception as exc:        # trace-time failure: never lose the step
            self._graphs.pop(key, None)
            if snapshot is not None:
                _restore(self.model, snapshot)
            if self.mode == "on":
                raise
            self._failed.add(key)
            self._note(f"compile failed ({type(exc).__name__}: {str(exc)[:160]})")
            self.stats["eager"] += 1
            with self._pin(depth):
                return self.eager(*args)
        _side_set(self.model, _join(meta["side"], flat))
        self.stats["compiled"] += 1
        return loss, grads


def make_grad_fn(model, loss_fn: Callable, *, mode: str | None = None, log=print):
    """Training ``grad_fn``: compiled (``CHRONOS_MLX_COMPILE``/``mode``, default auto) or eager."""
    mode = env_mode() if mode is None else str(mode).lower()
    if mode == "off":
        return nn.value_and_grad(model, loss_fn)
    return CompiledStep(model, loss_fn, mode=mode, log=log)
