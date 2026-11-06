"""Torch training under the single-compute-dtype policy (CPU / CUDA / MPS).

The model's parameters, activations and gradients live in the compute dtype
(``chronos.precision.resolve_torch``), so forward/backward run without the
per-op weight casts that ``torch.autocast`` performs on every call.  The
optimizer owns FP32 masters, mirroring ``MixedPrecisionAdamW`` on MLX:

* each low-precision gradient is folded into an FP32 master gradient the
  moment autograd finishes it (``register_post_accumulate_grad_hook``); the
  model gradient is freed, so accumulation over micro-batches is exact FP32
  and no extra low-precision grad buffer stays alive;
* AdamW updates the FP32 masters (moments are FP32 as well);
* the masters are copied back to the compute dtype once per update.

FP16 needs loss scaling (its range is too small for gradients); the scaler
operates on the FP32 master gradients.  BF16/FP32 need none.  A model that
is already FP32 gets no masters at all (zero overhead).
"""
from __future__ import annotations

import torch

from chronos.precision import PrecisionPolicy, resolve_torch


class _MasterWeights:
    """FP32 master copies for low-precision parameters of one model."""

    def __init__(self, params):
        self.pairs: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        self.master_of: dict[int, torch.Tensor] = {}
        self._handles = []
        for p in params:
            if not p.requires_grad or p.dtype == torch.float32 or id(p) in self.master_of:
                continue
            master = p.detach().float().clone()
            master.requires_grad_(True)
            self.master_of[id(p)] = master
            self.pairs.append((p, master))
            self._handles.append(p.register_post_accumulate_grad_hook(self._fold))

    def _fold(self, p: torch.nn.Parameter) -> None:
        master = self.master_of[id(p)]
        if p.grad is None:
            return
        if master.shape != p.shape:          # paged-out expert (zero-sized data): no update
            p.grad = None
            return
        if master.grad is None:
            master.grad = p.grad.float()
        else:
            master.grad.add_(p.grad)         # FP32 += low precision (promoted in-kernel)
        p.grad = None

    @torch.no_grad()
    def sync_to_model(self) -> None:
        for p, master in self.pairs:
            if master.shape == p.shape:
                p.copy_(master)              # one rounding per update

    @torch.no_grad()
    def sync_from_model(self) -> None:
        """Refresh masters after the model weights were replaced (resume)."""
        for p, master in self.pairs:
            if master.shape == p.shape:
                master.copy_(p)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _swap_params_to_masters(optimizer: torch.optim.Optimizer, masters: _MasterWeights) -> None:
    """Point every optimizer group at the FP32 masters (groups/decay kept)."""
    if not masters.pairs:
        return
    for group in optimizer.param_groups:
        group["params"] = [masters.master_of.get(id(p), p) for p in group["params"]]
    # PagedAdamW resolves names -> params for its SSD moment pages.
    by_name = getattr(optimizer, "_param_by_name", None)
    if isinstance(by_name, dict):
        names = getattr(optimizer, "_names", {})
        new_names = {}
        for name, p in list(by_name.items()):
            m = masters.master_of.get(id(p), p)
            by_name[name] = m
            new_names[id(m)] = name
        if isinstance(names, dict):
            names.update(new_names)


def prepare_torch_training(model, optimizer_factory, device, requested="auto"):
    """Place ``model`` in the policy dtype and build an FP32-master optimizer.

    ``optimizer_factory(model)`` builds the usual optimizer (decay groups,
    paging) from the *model's* parameters; they are then replaced by FP32
    masters.  Returns ``(model, optimizer, policy)``; ``optimizer`` exposes
    ``chronos_masters`` for :func:`optimizer_step`.
    """
    policy: PrecisionPolicy = resolve_torch(requested, device)
    model = model.to(device=device, dtype=policy.torch_dtype())
    optimizer = optimizer_factory(model)
    masters = _MasterWeights(model.parameters())
    _swap_params_to_masters(optimizer, masters)
    optimizer.chronos_masters = masters
    model._chronos_precision = policy
    return model, optimizer, policy


def grad_scaler_for(policy: PrecisionPolicy, device):
    """Loss scaling only for FP16 compute (on any backend that runs FP16)."""
    dev = str(device).split(":")[0]
    enabled = policy.compute == "float16" and dev in ("cuda", "mps", "cpu", "xpu")
    return torch.amp.GradScaler(dev, enabled=enabled)


def optimizer_step(scaler, optimizer, model, grad_clip: float) -> float:
    """Unscale -> clip -> step on FP32 masters -> copy back.  Returns grad norm."""
    masters = getattr(optimizer, "chronos_masters", None)
    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(params, grad_clip) if params else torch.zeros(())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    if masters is not None:
        masters.sync_to_model()
    evict = getattr(optimizer, "evict_expert_moments", None)
    if callable(evict):
        evict()
    return float(norm)


def apply_training_precision(trainer, *, models=(), requested=None) -> PrecisionPolicy:
    """Retrofit a stage trainer (``self.model`` + ``self.optimizer`` built)
    to the single-dtype policy.

    Moves the trainable model and any frozen companions (reference/teacher in
    ``models``) to the compute dtype, swaps the optimizer onto FP32 masters,
    disables autocast (the model is native) and picks the FP16-only scaler.
    Called with the trainer's ``args.dtype``; FP32 is a no-op apart from
    recording the policy.
    """
    requested = requested if requested is not None else getattr(trainer.args, "dtype", "auto")
    policy, trainer.autocast_ctx, trainer.scaler = attach_precision(
        trainer.model, trainer.optimizer, trainer.device, requested, models=models)
    trainer.precision = policy
    return policy


def attach_precision(model, optimizer, device, requested="auto", *, models=()):
    """Functional form of :func:`apply_training_precision` for custom loops.

    Returns ``(policy, autocast_ctx, scaler)``.  Must run before the first
    optimizer step (optimizer state is keyed by the FP32 masters).
    """
    import contextlib
    policy = resolve_torch(requested, device)
    dtype = policy.torch_dtype()
    model.to(device=device, dtype=dtype)
    for m in models:
        if m is not None:
            m.to(device=device, dtype=dtype)
    masters = _MasterWeights(model.parameters())
    _swap_params_to_masters(optimizer, masters)
    optimizer.chronos_masters = masters
    model._chronos_precision = policy
    return policy, contextlib.nullcontext(), grad_scaler_for(policy, device)
