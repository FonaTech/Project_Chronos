"""
chronos/trainer/optim_utils.py

Shared optimizer + LR-schedule helpers used by every Chronos training stage
(pretrain, SFT, DPO, ORPO, GRPO, distill) and the WebUI Train tab.

Two reasons to centralize this:

1. Decay groups: AdamW with weight_decay applied uniformly to all params
   destabilizes LayerNorm / RMSNorm scales and hurts embeddings. Standard
   recipe is to exclude norms, biases, and embeddings from decay. Doing
   this in one place means every trainer benefits from the same correct
   recipe and we don't get drift.

2. Warmup-then-cosine schedule. The pre-0.2 schedule was pure cosine (effective
   LR at step 0 is already 0.55·base_lr), which destabilizes the first few
   hundred steps with small batches. Linear warmup over the first ~5% of
   steps then cosine decay to 10% of base is the boring-and-correct choice.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from chronos.trainer.paged_store import PageKey, PagedOptimizerState


def _decay_groups(model: nn.Module) -> Tuple[list, list]:
    """Split trainable parameters into (decay, no_decay) lists.

    No-decay: any parameter whose owning module is a LayerNorm/RMSNorm,
    any 1-D parameter (typically bias / norm scale), and any embedding
    weight (regardless of dim — embeddings are looked-up rows, not
    rotated subspaces, so decay would push *all* tokens toward zero).
    """
    decay, no_decay = [], []
    seen = set()
    for mod in model.modules():
        is_norm = isinstance(mod, nn.LayerNorm) or mod.__class__.__name__.endswith("RMSNorm")
        is_embed = isinstance(mod, nn.Embedding)
        for p in mod.parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if is_norm or is_embed or p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    lr: float,
    *,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """AdamW with the standard transformer recipe applied to a Chronos model."""
    decay, no_decay = _decay_groups(model)
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr), betas=betas, eps=eps,
    )


class PagedAdamW(torch.optim.AdamW):
    """AdamW with crash-safe per-expert moment snapshots.

    PyTorch parameters remain optimizer-compatible; after every update, expert
    moments are streamed page-by-page to SSD. This is the safe first-stage
    paging implementation and provides resumable optimizer state without ever
    assembling an additional full CPU state dict. Accelerator-resident moment
    eviction can be layered on this contract by backend-specific trainers.
    """
    def __init__(self, named_parameters, *, page_dir: str, page_capacity: int = 8,
                 layer_to_group: dict[int, int] | None = None, **kwargs):
        named = [(name, p) for name, p in named_parameters if p.requires_grad]
        self._names = {id(p): name for name, p in named}
        self._param_by_name = {name: p for name, p in named}
        self._dirty_expert_params = set()
        self.page_store = PagedOptimizerState(page_dir, capacity=page_capacity)
        self._hydrated_pages = set()
        self._layer_to_group = {int(k): int(v) for k, v in (layer_to_group or {}).items()}
        # Keep the canonical two-group decay policy when callers provide a
        # model's named parameters.  Explicit ``params`` remains supported for
        # direct PyTorch use.
        if "params" not in kwargs:
            decay, no_decay = [], []
            for name, p in named:
                if p.ndim < 2 or "norm" in name.lower() or "bias" in name.lower() or "embed" in name.lower():
                    no_decay.append(p)
                else:
                    decay.append(p)
            super().__init__([
                {"params": decay, "weight_decay": float(kwargs.pop("weight_decay", 0.01))},
                {"params": no_decay, "weight_decay": 0.0},
            ], **kwargs)
        else:
            super().__init__([p for _, p in named], **kwargs)

    def _page_key(self, name: str):
        parts = name.split(".")
        try:
            li = int(parts[2]) if parts[:2] == ["model", "layers"] else None
            if li is None:
                return None
            expert_at = parts.index("experts")
            eid = int(parts[expert_at + 1])
        except (ValueError, IndexError):
            return None
        group = self._layer_to_group.get(li, li)
        return PageKey(li, group, eid, "adam")

    @torch.no_grad()
    def step(self, closure=None):
        # Resume expert moments lazily, one page at a time.  This avoids a
        # full CPU optimizer-state clone while preserving AdamW continuity.
        by_page = {}
        for name, p in self._param_by_name.items():
            key = self._page_key(name)
            if key is None or p.grad is None:
                continue
            self._dirty_expert_params.add(p)
            if key in self._hydrated_pages:
                continue
            suffix = name.split(f"experts.{key.expert}.", 1)[-1]
            by_page.setdefault(key, {})[suffix] = p
        for key, params in by_page.items():
            if not self.page_store.store.has(PageKey(key.layer, key.group, key.expert, "adam")):
                self._hydrated_pages.add(key)
                continue
            loaded = self.page_store.load(key, params)
            for suffix, param in params.items():
                state = self.state[param]
                state["step"] = torch.tensor(
                    float(loaded.get("step", 0)), device=param.device
                )
                state["exp_avg"] = loaded["exp_avg"][suffix].to(param.device, param.dtype)
                state["exp_avg_sq"] = loaded["exp_avg_sq"][suffix].to(param.device, param.dtype)
            self._hydrated_pages.add(key)
        # Expert pages may have moved between CPU and accelerator residency
        # since the previous microbatch.  Keep live Adam tensors colocated
        # with their parameter before invoking the fused/foreach update.
        for param in self._dirty_expert_params:
            state = self.state.get(param)
            if not state or "exp_avg" not in state:
                continue
            if state["exp_avg"].device != param.device:
                state["exp_avg"] = state["exp_avg"].to(param.device, param.dtype)
            if state["exp_avg_sq"].device != param.device:
                state["exp_avg_sq"] = state["exp_avg_sq"].to(param.device, param.dtype)
            if torch.is_tensor(state.get("step")) and state["step"].device != param.device:
                state["step"] = state["step"].to(param.device)
        loss = super().step(closure)
        grouped = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self._dirty_expert_params:
                    continue
                name = self._names.get(id(p), "")
                key = self._page_key(name)
                state = self.state.get(p)
                if key is None or not state or "exp_avg" not in state:
                    continue
                suffix = name.split(f"experts.{key.expert}.", 1)[-1]
                page = grouped.setdefault(key, {"exp_avg": {}, "exp_avg_sq": {}, "step": 0})
                page["exp_avg"][suffix] = state["exp_avg"]
                page["exp_avg_sq"][suffix] = state["exp_avg_sq"]
                page["step"] = max(int(page["step"]), int(state.get("step", 0)))
        for key, state in grouped.items():
            self.page_store.save(key, state, step=state.get("step", 0))
        self._dirty_expert_params.clear()
        return loss

    def state_dict(self):
        """Return the regular AdamW state plus paging metadata.

        Moments are already persisted per expert page.  Omitting those entries
        from the serialized PyTorch state is important: calling
        ``optimizer.state_dict()`` must not create a second full CPU copy of
        all expert moments, which would defeat SSD paging for large MoE
        checkpoints.  Non-expert state and the normal parameter-group layout
        remain fully PyTorch-compatible; expert pages are hydrated lazily on
        the first update after resume.
        """
        payload = super().state_dict()
        # ``Optimizer.state_dict`` replaces parameter objects with compact
        # integer ids in param-group order.  Reconstruct that mapping without
        # touching tensor data and remove only expert entries.
        param_ids = []
        for serialized_group, live_group in zip(
            payload.get("param_groups", []), self.param_groups
        ):
            param_ids.extend(zip(serialized_group.get("params", []), live_group.get("params", [])))
        expert_state_ids = set()
        for serialized_id, param in param_ids:
            name = self._names.get(id(param), "")
            if self._page_key(name) is not None:
                expert_state_ids.add(serialized_id)
        if expert_state_ids:
            payload["state"] = {
                key: value for key, value in payload.get("state", {}).items()
                if key not in expert_state_ids
            }
        payload["chronos_paging"] = {
            "page_dir": self.page_store.store.root_dir,
            "stats": self.page_store.store.summary(),
            "expert_pages": [
                key.filename() for key in self.page_store.store._cache.keys()
                if key.kind == "adam"
            ],
        }
        return payload

    @torch.no_grad()
    def evict_expert_moments(self) -> None:
        """Drop accelerator/CPU Adam moments already persisted as pages.

        This is called after an optimizer step in paged training.  The next
        step hydrates only experts that receive gradients, so optimizer memory
        follows the active working set instead of monotonically growing to all
        experts.  Non-expert state remains resident.
        """
        expert_params = []
        for param in list(self.state):
            name = self._names.get(id(param), "")
            if self._page_key(name) is not None:
                expert_params.append(param)
        for param in expert_params:
            self.state.pop(param, None)
        self._hydrated_pages.clear()
        self._dirty_expert_params.clear()

    def load_state_dict(self, state_dict):
        # Older checkpoints may not carry the optional paging envelope.
        payload = dict(state_dict)
        payload.pop("chronos_paging", None)
        return super().load_state_dict(payload)


def build_paged_optimizer(model: nn.Module, lr: float, *, page_dir: str,
                          weight_decay: float = 0.01, betas=(0.9, 0.95), eps=1e-8):
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "model"):
        cfg = getattr(model.model, "config", None)
    prelude = int(getattr(cfg, "prelude_layers", 0) or 0)
    groups = int(getattr(cfg, "recurrent_groups", 0) or 0)
    layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    layer_to_group = {
        li: (li - prelude if groups and prelude <= li < prelude + groups else -1)
        for li in range(layers)
    }
    # ``page_size_mb`` is a memory budget, not merely an informational field:
    # derive a bounded in-RAM page cache from the largest expert tensor set.
    page_mb = max(1, int(getattr(cfg, "page_size_mb", 64) or 64))
    expert_sizes = {}
    for name, param in model.named_parameters():
        if ".experts." in name:
            try:
                eid = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                eid = -1
            expert_sizes[eid] = expert_sizes.get(eid, 0) + int(param.numel() * 4)
    expert_bytes = max(expert_sizes.values(), default=1)
    page_capacity = max(1, int((page_mb * 1024 * 1024) / expert_bytes))
    # Keep a modest upper bound so an over-large page_size does not defeat
    # the purpose of bounded paging on machines with ample RAM.
    page_capacity = min(page_capacity, 256)
    return PagedAdamW(
        model.named_parameters(), page_dir=page_dir,
        page_capacity=page_capacity,
        layer_to_group=layer_to_group,
        lr=float(lr), weight_decay=float(weight_decay), betas=betas, eps=eps,
    )


def build_configured_optimizer(
    model: nn.Module,
    config,
    *,
    lr: float,
    weight_decay: float = 0.01,
    betas=(0.9, 0.95),
    eps=1e-8,
):
    """Build the optimizer selected by the v2 training configuration.

    ``optimizer_paging`` is the explicit switch for SSD-persisted Adam
    moments.  ``training_paging`` also enables it because a paged training
    run without paged optimizer state would immediately recreate a full
    accelerator/CPU moment footprint at the first step.  Keeping this policy
    in one function prevents the six stage trainers and the WebUI from
    silently diverging.
    """
    paged = bool(
        getattr(config, "optimizer_paging", False)
        or getattr(config, "training_paging", False)
    )
    if paged:
        return build_paged_optimizer(
            model,
            lr=lr,
            page_dir=str(getattr(config, "optimizer_page_dir", "./optimizer_pages")),
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    return build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def get_lr(
    current_step: int,
    total_steps: int,
    base_lr: float,
    *,
    warmup_steps: int | None = None,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup → cosine decay to `min_lr_ratio · base_lr`.

    `current_step` is 1-indexed (matches every trainer's `enumerate(loader, start=1)`).
    Falls back to a constant `base_lr` when `total_steps <= 0`.
    """
    if total_steps is None or total_steps <= 0:
        return float(base_lr)
    if warmup_steps is None:
        warmup_steps = max(1, min(500, total_steps // 20))
    step = max(1, int(current_step))
    if step <= warmup_steps:
        return float(base_lr) * (step / float(warmup_steps))
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate on every param-group in `optimizer`."""
    for g in optimizer.param_groups:
        g["lr"] = float(lr)
