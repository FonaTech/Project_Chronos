"""Activation recomputation (``config.activation_checkpointing``) for MLX training.

Each block call becomes an ``mx.checkpoint`` region: the backward pass keeps
only the block's input and output and recomputes everything in between.
Peak memory in Chronos training is dominated by per-virtual-layer
activations, and a recurrent group runs one physical block once per
iteration, so this is where the memory goes.

A plain ``mx.checkpoint`` would be silently wrong for these blocks, for two
reasons:

* **Random draws.**  Dropout and the MoE cache-miss simulation
  (``fallback_mask_prob``) draw fresh bits when the region is recomputed, so
  the backward pass would differentiate a different network than the one the
  forward pass ran (measured on mlx 0.32.4: same loss, different gradients).
  Every draw goes through :func:`draw`, which records it on the region's tape
  during the forward pass and replays the recorded array during the
  recomputation.  Draws are consumed in the same order and count as without
  checkpointing, so the global RNG stream (and resume) is unchanged.
* **Side effects.**  The MoE appends its router probabilities once per call,
  and the regularized loss reads them.  The recomputation must not append a
  second copy, and the loss must read the forward-pass arrays (the
  recomputed ones are internal to the backward pass).  :func:`recomputing`
  tells modules that the current call is a recomputation.

Only arrays cross a region boundary, packed into one output per dtype so
no two outputs can be the same graph node (see :func:`_pack`);
:func:`call_block` rebuilds the non-array parts of a block's return value
(e.g. the SWA absolute position) around them.
"""
from __future__ import annotations

import threading
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

_STATE = threading.local()


class _Tape:
    __slots__ = ("draws", "pos", "replay", "calls")

    def __init__(self):
        self.draws: list = []
        self.pos = 0
        self.replay = False
        self.calls = 0


def _tape() -> _Tape | None:
    return getattr(_STATE, "tape", None)


def recomputing() -> bool:
    """True while a checkpointed region is being recomputed for its gradient."""
    tape = _tape()
    return tape is not None and tape.replay


def recorded(fn: Callable[[], object]):
    """``fn()`` in the forward pass; the same value again in its recomputation.

    For values that carry no gradient but decide what a block computes:
    random draws, routing indices, and module state read inside a block
    (another forward may change that state before the backward pass runs).
    Outside a checkpointed region this is just ``fn()``.
    """
    tape = _tape()
    if tape is None:
        return fn()
    if tape.replay:
        if tape.pos >= len(tape.draws):
            raise RuntimeError("checkpoint recomputation recorded more values than its forward pass")
        value = tape.draws[tape.pos]
    else:
        value = fn()
        if isinstance(value, mx.array):
            value = mx.stop_gradient(value)
        tape.draws.append(value)
    tape.pos += 1
    return value


draw = recorded       # random arrays: the common case


def dropout(module, x: mx.array) -> mx.array:
    """``nn.Dropout`` semantics with a tape-recorded mask."""
    keep = float(getattr(module, "_p_1", 1.0))
    if keep == 1.0 or not module.training:
        return x
    mask = draw(lambda: mx.random.bernoulli(keep, x.shape))
    return (mask * x) * (1.0 / keep)


def checkpointed(module, fn: Callable[..., tuple]) -> Callable[..., tuple]:
    """``fn(*arrays)`` recomputed in the backward pass, with tape replay.

    ``module`` owns the trainable parameters ``fn`` uses (they are passed
    through the region so their gradients are recomputed too).  ``fn`` must
    take and return arrays (or tuples of arrays) only.
    """
    tape = _Tape()

    def region(params, *args):
        prev = _tape()
        tape.replay = tape.calls > 0        # 1st call: forward; later calls: recomputation
        tape.calls += 1
        tape.pos = 0
        # The forward call receives the module's own arrays; a recomputation
        # receives its own primals, which must be the ones the ops capture.
        # They are swapped back afterwards so the module never keeps them.
        saved = module.trainable_parameters() if tape.replay else None
        if saved is not None:
            module.update(params)
        _STATE.tape = tape
        try:
            return fn(*args)
        finally:
            _STATE.tape = prev
            if saved is not None:
                module.update(saved)

    run = mx.checkpoint(region)

    def call(*args):
        return run(module.trainable_parameters(), *args)

    return call


class Dropout(nn.Dropout):
    """``nn.Dropout`` whose mask is replayed inside a recomputation."""

    def __call__(self, x: mx.array) -> mx.array:
        return dropout(self, x)


def side_output_sink() -> list | None:
    """Inside a checkpointed block: list collecting ``(owner, kind, array)`` side outputs."""
    return getattr(_STATE, "sink", None)


class _Slot:
    pass


def _split(obj, arrays: list):
    """Replace every array leaf with ``_Slot`` and collect it (tuples/lists/dicts)."""
    if isinstance(obj, mx.array):
        arrays.append(obj)
        return _Slot
    if isinstance(obj, (tuple, list)):
        return type(obj)(_split(o, arrays) for o in obj)
    if isinstance(obj, dict):
        return {k: _split(v, arrays) for k, v in obj.items()}
    return obj                                      # None / int / float / bool / str


def _join(spec, it):
    if spec is _Slot:
        return next(it)
    if isinstance(spec, (tuple, list)):
        return type(spec)(_join(s, it) for s in spec)
    if isinstance(spec, dict):
        return {k: _join(v, it) for k, v in spec.items()}
    return spec


def _pack(arrays: list) -> tuple[tuple, list]:
    """Region outputs as one array per dtype (a flat concatenation).

    mlx 0.32.4 keeps the cotangent of only the first copy when a checkpointed
    function returns the same graph node more than once, silently dropping
    the others' gradients.  Aliases are not visible from Python: a block's
    new cache and its shared KV view may be one object, but a full-length
    slice (``k[:, -keep:]`` with ``keep >= T``) or a same-dtype ``astype`` is
    a *new* Python object over the *same* node.  Packing makes every output
    a distinct node by construction; the concatenation's backward splits
    the cotangent exactly, and an aliased array receives the sum of its
    copies' cotangents through ordinary autodiff inside the recomputation.
    """
    groups: dict = {}
    layout = []
    for a in arrays:
        members = groups.setdefault(a.dtype, [])
        layout.append((a.dtype, len(members), tuple(a.shape), a.size))
        members.append(a)
    order = list(groups)
    packed = tuple(groups[d][0] if len(groups[d]) == 1 else
                   mx.concatenate([a.reshape(-1) for a in groups[d]]) for d in order)
    starts: dict = {}
    for d in order:
        offset, starts[d] = 0, []
        for a in groups[d]:
            starts[d].append(offset)
            offset += a.size
    return packed, [(order.index(d), len(groups[d]) == 1, starts[d][j], shape, size)
                    for d, j, shape, size in layout]


def _unpack(packed: tuple, layout: list) -> list:
    out = []
    for g, single, start, shape, size in layout:
        out.append(packed[g] if single else packed[g][start:start + size].reshape(shape))
    return out


def call_block(block, hidden: mx.array, **kwargs):
    """``block(hidden, **kwargs)`` as one recomputed region.

    ``kwargs`` and the block's return value may mix arrays with plain Python
    values (``None``, ints such as the SWA absolute position); only the
    arrays cross the region boundary.  Side outputs a module hands to
    :func:`side_output_sink` (MoE router probabilities) are returned as
    region outputs and delivered to their owner via ``owner.accept_side_output``,
    so a loss term that reads them is differentiated through the
    recomputation instead of keeping the block's activations alive.
    """
    arrays = [hidden]
    in_spec = _split(kwargs, arrays)
    meta: dict = {}

    def fn(h, *rest):
        prev = side_output_sink()
        sink: list = []
        _STATE.sink = sink
        try:
            out = block(h, **_join(in_spec, iter(rest)))
        finally:
            _STATE.sink = prev
        flat: list = []
        meta["spec"] = _split(out, flat)
        meta["owners"] = [(owner, kind) for owner, kind, _ in sink]
        flat += [value for _, _, value in sink]
        packed, meta["layout"] = _pack(flat)
        return packed

    outputs = _unpack(checkpointed(block, fn)(*arrays), meta["layout"])
    owners = meta["owners"]
    n_main = len(outputs) - len(owners)
    for (owner, kind), value in zip(owners, outputs[n_main:]):
        owner.accept_side_output(kind, value)
    return _join(meta["spec"], iter(outputs[:n_main]))
