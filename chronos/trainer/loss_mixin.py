"""
chronos/trainer/loss_mixin.py

Reusable loss-mixing helpers shared across all Chronos training stages
(pretrain, SFT, DPO, ORPO, GRPO). Centralizing this keeps each stage
trainer thin and guarantees lookahead supervision is preserved through
the whole 6-stage chain rather than being silently dropped by a stage
that forgot to add it.

Two helpers are exposed:

- ``chronos_loss_term(model, base_loss, lookahead_probs, config)`` —
  appends λ_balance·L_aux + λ_temporal·L_temporal + λ_lookahead·L_lookahead
  to any base loss (CE for pretrain/SFT, DPO loss, ORPO loss, etc.).

- ``router_kl_anchor(current_router_probs, ref_router_probs, lambda_anchor)`` —
  KL(current || reference) penalty. Captured at the start of an alignment
  stage by running the freshly-loaded model in eval mode on a calibration
  batch; prevents the alignment gradients from drifting routing away from
  the pretrained distribution that the cluster layout was optimized for.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from chronos.model.moe_chronos import ChronosMOEFeedForward
from chronos.model.temporal_loss import (
    cluster_locality_loss,
    iteration_cluster_spread,
    temporal_locality_loss,
    lookahead_supervision_loss,
    lookahead_topk_hit_loss,
    lookahead_union_loss,
    router_locality_loss,
    router_offload_metrics,
    segment_cluster_loss,
)


def collect_router_probs(model, *, with_grad: bool = False) -> Optional[torch.Tensor]:
    """Stack router probs from every ChronosMOEFeedForward layer.

    Args:
        with_grad: when True, returns the live tensor that still has an
                   autograd edge back to the gate (used by training loss
                   terms that need to shape the gate). When False, returns
                   the detached tensor (safe for logging / cluster layout
                   / KL anchor reference / lookahead teacher).

    Returns: ``[B, S, L, E]`` or ``None`` if no MoE layers fired in the
    last forward pass (e.g. before the first forward).
    """
    probs: List[torch.Tensor] = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, ChronosMOEFeedForward):
            p = mlp.last_router_probs_grad if with_grad else mlp.last_router_probs
            if p is not None:
                probs.append(p)
    if not probs:
        return None
    return torch.stack(probs, dim=2)


def real_token_mask_torch(labels: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` bool real-token mask of a right-padded batch (HF labels: the
    loss shifts).  A row's real tokens end at its last supervised label; later
    positions are padding.  Prompt tokens stay real.  Same rule as the MLX
    ``trainer.real_token_mask(labels)``."""
    pos = torch.arange(labels.shape[1], device=labels.device)
    last = torch.where(labels != -100, pos[None, :], torch.full_like(labels, -1)).max(dim=1).values
    return pos[None, :] <= last[:, None]


def cluster_membership_tensor(config, device=None) -> torch.Tensor:
    """``[E, C]`` one-hot of ``chronos.expert_clusters.training_clusters(config)``."""
    from chronos.expert_clusters import cluster_membership, training_clusters
    e = int(config.num_experts)
    return torch.tensor(cluster_membership(training_clusters(config), e), dtype=torch.float32, device=device)


def collect_iteration_router_probs(model, *, with_grad: bool = False) -> Optional[torch.Tensor]:
    """Every dispatch decision of the last forward as ``[B, S, V, E]``.

    A recurrent block routes once per loop iteration, so V counts virtual
    layers (prelude + groups·depth + coda for a full-depth forward).  Load
    balance and temporal locality must hold for each of them.
    """
    probs: List[torch.Tensor] = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, ChronosMOEFeedForward):
            probs.extend(mlp.iteration_router_probs_grad if with_grad else mlp.iteration_router_probs)
    if not probs:
        return None
    return torch.stack(probs, dim=2)


def first_iteration_router_probs(model) -> Optional[torch.Tensor]:
    """Step-0 routing per physical layer (lookahead teacher; prefetch key)."""
    probs = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, ChronosMOEFeedForward) and mlp.iteration_router_probs:
            probs.append(mlp.iteration_router_probs[0])
    return torch.stack(probs, dim=2) if probs else None


def recurrent_iteration_probs(model, config, *, with_grad: bool = True) -> Optional[torch.Tensor]:
    """``[B, S, G, D, E]``: router probabilities of each recurrent layer at each loop iteration (None unless
    every recurrent layer dispatched the same number, at least 2, of times in the last forward)."""
    start, groups = int(getattr(config, "prelude_layers", 0) or 0), int(getattr(config, "recurrent_groups", 0) or 0)
    rows = []
    for layer in list(model.model.layers)[start:start + groups]:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, ChronosMOEFeedForward):
            return None
        seq = mlp.iteration_router_probs_grad if with_grad else mlp.iteration_router_probs
        if not seq:
            return None
        rows.append(torch.stack(list(seq), dim=2))
    if not rows or rows[0].shape[2] < 2 or any(r.shape[2] != rows[0].shape[2] for r in rows):
        return None
    return torch.stack(rows, dim=2)


def iteration_mean_router_probs(model) -> Optional[torch.Tensor]:
    """Per physical layer, router probabilities averaged over its loop iterations ``[B, S, L, E]``
    (lookahead teacher covering every iteration; the step-0 routing for one-dispatch layers)."""
    probs = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, ChronosMOEFeedForward) and mlp.iteration_router_probs:
            probs.append(torch.stack(list(mlp.iteration_router_probs), dim=0).mean(dim=0))
    return torch.stack(probs, dim=2) if probs else None


def lookahead_teacher(router_4d, lookahead_probs, config):
    """Align prediction head g to physical layer prelude_layers + g."""
    if lookahead_probs.dim() == 4:
        return router_4d.mean(dim=2).detach()
    groups = lookahead_probs.shape[3]
    start = int(getattr(config, "prelude_layers", 0))
    teacher = router_4d[:, :, start:start + groups, :]
    if teacher.shape[2] != groups:
        raise ValueError("lookahead heads do not match the recurrent physical layers")
    return teacher.detach()


def chronos_loss_term(
    model,
    base_loss: torch.Tensor,
    lookahead_probs: Optional[torch.Tensor],
    config,
    *,
    aux_loss: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Combine `base_loss` with the Chronos-specific regularizers:

        L = L_base
            + λ_balance  · L_aux            (if aux_loss provided)
            + λ_temporal · L_temporal       (if seq has ≥2 tokens)
            + λ_lookahead · L_lookahead     (if lookahead_probs + λ_la > 0)

    The function is safe to call from any stage: it gracefully degrades
    when the necessary tensors are unavailable (e.g. lookahead_probs is
    None during a model that doesn't expose it).

    ``input_ids`` (and ``labels``, for the padding mask) let
    ``cluster_locality_mode="segment"`` find the documents of the batch;
    without them the window form is used.
    """
    loss = base_loss

    if aux_loss is not None:
        lambda_balance = float(getattr(config, "lambda_balance", 0.0))
        if lambda_balance > 0.0:
            # Prefer the un-scaled raw aux summed across MoE layers so we
            # don't double-multiply by router_aux_loss_coef. Falls back to
            # whatever the model handed us if aux_loss_raw isn't populated
            # (e.g. eval-mode forward).
            raw_total = base_loss.new_zeros(())
            saw_raw = False
            for layer in model.model.layers:
                mlp = getattr(layer, "mlp", None)
                if isinstance(mlp, ChronosMOEFeedForward) and mlp.aux_loss_raw is not None:
                    raw_total = raw_total + mlp.aux_loss_raw
                    saw_raw = True
            aux_term = raw_total if saw_raw else aux_loss
            loss = loss + lambda_balance * aux_term

    router_4d = collect_iteration_router_probs(model, with_grad=True)
    if router_4d is not None and router_4d.shape[1] > 1:
        # Locality must hold within each dispatch (virtual layer). Averaging
        # layers lets opposite routing changes cancel and hides page churn.
        router_mean = router_4d.permute(0, 2, 1, 3).flatten(0, 1)
        lambda_temporal = float(getattr(config, "lambda_temporal", 0.0))
        if lambda_temporal > 0.0:
            loss = loss + lambda_temporal * temporal_locality_loss(router_mean)
        lambda_router_locality = float(getattr(config, "lambda_router_locality", 0.0))
        if lambda_router_locality > 0.0:
            loss = loss + lambda_router_locality * router_locality_loss(
                router_mean,
                int(getattr(config, "num_experts_per_tok", 1) or 1),
            )
        lambda_cluster = float(getattr(config, "lambda_cluster_locality", 0.0) or 0.0)
        if lambda_cluster > 0.0:
            from chronos.expert_clusters import cluster_locality_mode, segment_ids
            membership = cluster_membership_tensor(config, router_4d.device)
            window = int(getattr(config, "cluster_locality_window", 0) or 0)
            if cluster_locality_mode(config) == "segment" and input_ids is not None:
                ids = input_ids.detach().cpu().numpy()
                mask = None if labels is None else real_token_mask_torch(labels)
                cluster_term = segment_cluster_loss(router_4d, membership, segment_ids(ids, window), mask)
            else:
                cluster_term = cluster_locality_loss(router_4d, membership, window)
            loss = loss + lambda_cluster * cluster_term

        lambda_iteration = float(getattr(config, "lambda_iteration_cluster", 0.0) or 0.0)
        if lambda_iteration > 0.0:
            iteration_probs = recurrent_iteration_probs(model, config)
            if iteration_probs is not None:
                mask = None if labels is None else real_token_mask_torch(labels)
                loss = loss + lambda_iteration * iteration_cluster_spread(iteration_probs, cluster_membership_tensor(config, iteration_probs.device), mask)

        lambda_lookahead = float(getattr(config, "lambda_lookahead", 0.0))
        lookahead_steps = int(getattr(config, "lookahead_steps", 0))
        if (
            lookahead_probs is not None
            and lambda_lookahead > 0.0
            and lookahead_steps > 0
        ):
            # Teacher must be detached: lookahead targets are stop-grad'd
            # by construction (we predict the *future* router state).
            source = (iteration_mean_router_probs(model) if getattr(config, "lookahead_teacher_scope", "first") == "iterations"
                    else first_iteration_router_probs(model))
            teacher = lookahead_teacher(source, lookahead_probs, config)
            la = lookahead_supervision_loss(lookahead_probs, teacher, lookahead_steps)
            loss = loss + lambda_lookahead * la
            lambda_lookahead_topk = float(getattr(config, "lambda_lookahead_topk", 0.0))
            if lambda_lookahead_topk > 0.0:
                topk_loss = lookahead_topk_hit_loss(
                    lookahead_probs,
                    teacher,
                    lookahead_steps,
                    int(getattr(config, "num_experts_per_tok", 1) or 1),
                )
                loss = loss + lambda_lookahead_topk * topk_loss
            lambda_lookahead_union = float(getattr(config, "lambda_lookahead_union", 0.0))
            if lambda_lookahead_union > 0.0:
                union_loss = lookahead_union_loss(
                    lookahead_probs,
                    teacher,
                    lookahead_steps,
                    int(getattr(config, "num_experts_per_tok", 1) or 1),
                )
                loss = loss + lambda_lookahead_union * union_loss
    # Recurrent-depth regularizers are exposed by ChronosModel without
    # changing the HuggingFace output tuple.  This keeps all six stage
    # trainers source-compatible while shaping the shared transition.
    recurrent_state = getattr(getattr(model, "model", None), "last_recurrent_state", None)
    if recurrent_state:
        ponder_coef = float(getattr(config, "ponder_loss_coef", 0.0))
        stability_coef = float(getattr(config, "recurrent_stability_coef", 0.0))
        distill_coef = float(getattr(config, "recurrent_distill_coef", 0.0))
        if ponder_coef > 0:
            loss = loss + ponder_coef * recurrent_state["ponder_loss"]
        if stability_coef > 0:
            loss = loss + stability_coef * recurrent_state["stability_loss"]
        if distill_coef > 0:
            loss = loss + distill_coef * recurrent_state["distill_loss"]
        # Penalize expansive fixed points while allowing the learned
        # contractive Euler step to approach one near convergence.
        contraction_budget = float(getattr(config, "recurrent_contraction_budget", 1.0))
        contraction_coef = float(getattr(config, "recurrent_contraction_coef", 0.0))
        if contraction_coef > 0 and "contraction" in recurrent_state:
            loss = loss + contraction_coef * (
                recurrent_state["contraction"] - contraction_budget
            ).clamp_min(0.0).pow(2)
    # Self-speculative draft exit: KL(full.detach() || draft-depth logits),
    # computed by ChronosForCausalLM.forward when draft_exit_kl_coef > 0.
    draft_kl = getattr(model, "last_draft_exit_kl", None)
    draft_coef = float(getattr(config, "draft_exit_kl_coef", 0.0) or 0.0)
    if draft_kl is not None and draft_coef > 0.0:
        loss = loss + draft_coef * draft_kl
    return loss


def collect_offload_training_metrics(
    model,
    lookahead_probs: Optional[torch.Tensor],
    config,
) -> dict[str, float]:
    router_4d = collect_router_probs(model, with_grad=False)
    if router_4d is None:
        return {}
    router_mean = router_4d.mean(dim=2)
    return router_offload_metrics(
        router_mean,
        lookahead_probs.detach() if lookahead_probs is not None else None,
        int(getattr(config, "lookahead_steps", 0) or 0),
        int(getattr(config, "num_experts_per_tok", 1) or 1),
    )


def router_kl_anchor(
    current_router_probs: torch.Tensor,
    ref_router_probs: torch.Tensor,
    lambda_anchor: float,
) -> torch.Tensor:
    """KL(current || reference) over routing distributions, scaled by λ.

    Both inputs are layer-averaged ``[B, S, E]`` distributions (post-softmax).
    Reference is treated as a fixed teacher (no grad needed; caller passes
    a detached tensor). Reduces to mean over ``B*S``.
    """
    if lambda_anchor <= 0.0:
        return current_router_probs.new_zeros(())
    eps = 1e-9
    kl = (current_router_probs * (
        current_router_probs.clamp_min(eps).log()
        - ref_router_probs.clamp_min(eps).log()
    )).sum(dim=-1).mean()
    return lambda_anchor * kl


def capture_reference_routing(model, calibration_batch, device, *, modality_inputs=None, modality_mask=None) -> torch.Tensor:
    """Run `model` in eval mode on the calibration batch, return the
    layer-averaged routing distribution as a detached reference for the
    `router_kl_anchor` term.

    The calibration batch is a single ``input_ids`` tensor (already on
    device-or-not is handled here). Returns ``[B, S, E]`` detached.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        ids = calibration_batch.to(device)
        if modality_inputs:
            modality_inputs = {k: v.to(device) for k, v in modality_inputs.items()}
        if modality_mask is not None:
            modality_mask = modality_mask.to(device)
        try:
            _ = model(ids, use_cache=False, modality_inputs=modality_inputs, modality_mask=modality_mask)
        except TypeError:
            _ = model(ids)  # plain HF-style models without the Chronos kwargs
        router_4d = collect_router_probs(model)
        if router_4d is None:
            ref = None
        else:
            ref = router_4d.mean(dim=2).detach()
    if was_training:
        model.train()
    return ref
