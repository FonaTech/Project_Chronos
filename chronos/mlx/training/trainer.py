"""Minimal but real MLX-native six-stage training support."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch

from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_controlled,
    save_state_dict_with_config,
)
from chronos.trainer.optim_utils import get_lr

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.moe import ChronosMLXMOE
    from chronos.mlx.multimodal import batch_modalities
    from chronos.mlx.training.io import mlx_state_to_torch
    from chronos.mlx.training.optim import MixedPrecisionAdamW
except ModuleNotFoundError:
    mx = None
    nn = None
    optim = None
    tree_map = None
    ChronosMLXModel = None
    ChronosMLXMOE = None
    batch_modalities = None
    mlx_state_to_torch = None


def _ids_to_mx(x) -> mx.array:
    if isinstance(x, mx.array):
        return x.astype(mx.int32)
    if isinstance(x, torch.Tensor):
        return mx.array(x.detach().cpu().numpy()).astype(mx.int32)
    return mx.array(x).astype(mx.int32)


def _f32(x: mx.array) -> mx.array:
    """FP32 view for loss math; a no-op when logits were born FP32."""
    return x if x.dtype == mx.float32 else x.astype(mx.float32)


def _masked_lm_loss(logits: mx.array, labels: mx.array) -> mx.array:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_logits = _f32(shift_logits.reshape(-1, shift_logits.shape[-1]))
    flat_labels = shift_labels.reshape(-1)
    mask = flat_labels != -100
    safe_labels = mx.where(mask, flat_labels, mx.zeros_like(flat_labels))
    losses = nn.losses.cross_entropy(flat_logits, safe_labels, reduction="none")
    return (losses * mask.astype(losses.dtype)).sum() / mx.maximum(mask.astype(mx.float32).sum(), 1.0)


def _gather_token_logprobs(logits: mx.array, labels: mx.array) -> mx.array:
    labels = labels.astype(mx.int32)
    logits = _f32(logits)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    flat = log_probs.reshape(-1, log_probs.shape[-1])
    idx = labels.reshape(-1)
    rows = mx.arange(idx.shape[0])
    return flat[rows, idx].reshape(labels.shape)


def _aligned_sequence_logprob(
    logits: mx.array,
    labels: mx.array,
    mask: mx.array,
    *,
    average: bool = False,
) -> mx.array:
    tok_lp = _gather_token_logprobs(logits, labels)
    mask = mask.astype(mx.float32)
    total = (tok_lp * mask).sum(axis=1)
    if average:
        return total / mx.maximum(mask.sum(axis=1), 1.0)
    return total


def _shifted_token_logprobs(logits: mx.array, ids: mx.array) -> mx.array:
    return _gather_token_logprobs(logits[:, :-1, :], ids[:, 1:])


def _collect_router_probs(model) -> mx.array | None:
    """Every dispatch of the last forward: ``[B, S, V, E]`` (V = virtual layers)."""
    probs = []
    for layer in getattr(model, "layers", []):
        moe = getattr(layer, "mlp", None)
        if isinstance(moe, ChronosMLXMOE):
            probs.extend(moe.iteration_router_probs)
    if not probs:
        return None
    return mx.stack(probs, axis=2)


def _first_iteration_router_probs(model) -> mx.array | None:
    """Step-0 routing per physical layer ``[B, S, L, E]`` (lookahead teacher)."""
    probs = [layer.mlp.iteration_router_probs[0] for layer in getattr(model, "layers", [])
             if isinstance(getattr(layer, "mlp", None), ChronosMLXMOE) and layer.mlp.iteration_router_probs]
    return mx.stack(probs, axis=2) if probs else None


def _masked_mean(values: mx.array, mask: mx.array | None) -> mx.array:
    """Mean of ``values`` over positions where ``mask`` (same shape, 1 = real) is set."""
    if mask is None:
        return values.mean()
    m = mask.astype(mx.float32)
    return (values * m).sum() / mx.maximum(m.sum(), 1.0)


def _pair_mask(token_mask: mx.array | None, k: int = 1) -> mx.array | None:
    """``[B, S-k]``: position t and t+k are both real tokens."""
    if token_mask is None:
        return None
    m = token_mask.astype(mx.float32)
    return m[:, :-k] * m[:, k:]


def _group_mask(token_mask: mx.array | None, groups: int) -> mx.array | None:
    """Repeat a ``[B, S]`` mask for the ``B*G`` rows of ``_flatten_lookahead_groups``."""
    if token_mask is None or groups <= 1:
        return token_mask
    return mx.repeat(token_mask, groups, axis=0)


def _temporal_locality_loss(router_mean: mx.array, token_mask: mx.array | None = None) -> mx.array:
    if router_mean.shape[1] <= 1:
        return mx.zeros(())
    diff = router_mean[:, 1:, :] - router_mean[:, :-1, :]
    return _masked_mean((diff * diff).sum(axis=-1), _pair_mask(token_mask))


def _router_locality_loss(router_mean: mx.array, num_experts_per_tok: int,
                          token_mask: mx.array | None = None) -> mx.array:
    if router_mean.shape[1] <= 1:
        return mx.zeros(())
    B, _S, E = router_mean.shape
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    prev = mx.stop_gradient(router_mean[:, :-1, :])
    cur = router_mean[:, 1:, :]
    target_ids = mx.argpartition(-prev, kth=top_k - 1, axis=-1)[:, :, :top_k]
    target_mask = (target_ids[..., None] == mx.arange(E)).astype(cur.dtype).sum(axis=-2)
    mass = (cur * target_mask).sum(axis=-1)
    return _masked_mean(-mx.log(mx.maximum(mass / float(top_k), 1e-9)), _pair_mask(token_mask))


_MEMBERSHIP_CACHE: dict = {}


def _cluster_membership(config) -> mx.array:
    """``[E, C]`` one-hot of ``chronos.expert_clusters.training_clusters(config)`` (cached)."""
    from chronos.expert_clusters import cluster_membership, resolve_cluster_count, training_clusters
    e = int(config.num_experts)
    key = (e, resolve_cluster_count(config))
    hit = _MEMBERSHIP_CACHE.get(key)
    if hit is None:
        hit = mx.array(cluster_membership(training_clusters(config), e), dtype=mx.float32)
        _MEMBERSHIP_CACHE[key] = hit
    return hit


def _cluster_locality_loss(router_4d: mx.array, membership: mx.array, window: int = 0,
                           token_mask: mx.array | None = None) -> mx.array:
    """MLX twin of ``chronos.model.temporal_loss.cluster_locality_loss``: ``1 − I(window; cluster)/log C``.

    ``token_mask`` ``[B, S]`` (1 = real token) drops right-padding: a window's
    cluster distribution averages its real tokens only, and windows count in
    proportion to them, so pad runs are neither a "domain" of their own nor
    dilute the windows they share.  ``None`` = every position is real.
    """
    B, S, V, E = router_4d.shape
    C = int(membership.shape[1])
    w = S if int(window or 0) <= 0 else min(int(window), S)
    n = S // w
    if C <= 1 or n <= 0:
        return mx.zeros(())
    mass = (router_4d.astype(mx.float32).reshape(-1, E) @ membership).reshape(B, S, V, C)
    mass = mass[:, : n * w].reshape(B, n, w, V, C)

    def entropy(p):
        p = mx.maximum(p, 1e-9)
        return -(p * mx.log(p)).sum(axis=-1)

    if token_mask is None:
        pw = mass.mean(axis=(2, 3)).reshape(B * n, C)
        mutual = entropy(pw.mean(axis=0)) - entropy(pw).mean()
        return 1.0 - mutual / math.log(C)
    m = token_mask[:, : n * w].astype(mx.float32).reshape(B, n, w)
    count = m.sum(axis=2)                                               # [B, n] real tokens per window
    pw = ((mass * m[..., None, None]).sum(axis=2).mean(axis=2)
          / mx.maximum(count, 1.0)[..., None]).reshape(B * n, C)
    weight = (count / mx.maximum(count.sum(), 1.0)).reshape(B * n)      # windows ∝ real tokens
    marginal = (pw * weight[:, None]).sum(axis=0)
    mutual = entropy(marginal) - (entropy(pw) * weight).sum()
    return 1.0 - mutual / math.log(C)


def _segment_cluster_loss(router_4d: mx.array, membership: mx.array, segments,
                          token_mask: mx.array | None = None) -> mx.array:
    """``1 − I(segment; cluster)/log C`` with documents as the unit (MLX twin of
    ``chronos.model.temporal_loss.segment_cluster_loss``).

    ``segments`` ``[B, S]`` comes from ``chronos.expert_clusters.segment_ids``:
    the token stream's own document separators, no label or task token.  Each
    segment's cluster mass is pooled over its real tokens and every dispatch;
    segments weigh by real-token count."""
    from chronos.expert_clusters import segment_index
    B, S, V, E = router_4d.shape
    C = int(membership.shape[1])
    inv, n = segment_index(segments)
    if C <= 1 or n <= 1:
        return mx.zeros(())
    mass = (router_4d.astype(mx.float32).reshape(-1, E) @ membership).reshape(B * S, V, C).mean(axis=1)
    m = (mx.ones((B * S,), mx.float32) if token_mask is None
         else token_mask.reshape(B * S).astype(mx.float32))
    idx = mx.array(inv)
    total = mx.zeros((n, C), mx.float32).at[idx].add(mass * m[:, None])
    count = mx.zeros((n,), mx.float32).at[idx].add(m)
    ps = total / mx.maximum(count, 1.0)[:, None]
    weight = count / mx.maximum(count.sum(), 1.0)

    def entropy(p):
        p = mx.maximum(p, 1e-9)
        return -(p * mx.log(p)).sum(axis=-1)

    marginal = (ps * weight[:, None]).sum(axis=0)
    mutual = entropy(marginal) - (entropy(ps) * weight).sum()
    return 1.0 - mutual / math.log(C)


def _cluster_objective(router_4d: mx.array, config, token_mask=None, input_ids=None) -> mx.array:
    """The configured cluster-locality term: ``window`` (fixed token windows)
    or ``segment`` (documents found from ``input_ids``; window form without them)."""
    from chronos.expert_clusters import cluster_locality_mode, segment_ids
    window = int(getattr(config, "cluster_locality_window", 0) or 0)
    membership = _cluster_membership(config)
    if cluster_locality_mode(config) == "segment" and input_ids is not None:
        return _segment_cluster_loss(router_4d, membership, segment_ids(input_ids, window), token_mask)
    return _cluster_locality_loss(router_4d, membership, window, token_mask)


def _load_balance_loss(router_4d: mx.array, num_experts_per_tok: int,
                       token_mask: mx.array | None = None) -> mx.array:
    """Switch-style ``E · Σ load·importance`` per virtual layer, summed.

    ``token_mask`` ``[B, S]`` (1 = real token) restricts load and importance
    to real tokens; ``None`` = every position counts (packed pretraining)."""
    B, S, L, E = router_4d.shape
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    total = mx.zeros(())
    w = None
    if token_mask is not None:
        w = token_mask.reshape(B * S).astype(router_4d.dtype)
        w = w / mx.maximum(w.sum(), 1.0)
    for li in range(L):
        scores = router_4d[:, :, li, :].reshape(B * S, E)
        topk_idx = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k]
        selected = (topk_idx[..., None] == mx.arange(E)).astype(scores.dtype)
        if w is None:
            load = selected.mean(axis=0)  # [K, E], matching the PyTorch trainer
            importance = scores.mean(axis=0)
        else:
            load = (selected * w[:, None, None]).sum(axis=0)
            importance = (scores * w[:, None]).sum(axis=0)
        total = total + (load * importance).sum() * E
    return total


def _flatten_lookahead_groups(prediction, teacher):
    """Flatten matching groups into batch; never average distinct routing heads."""
    if prediction is None or teacher is None:
        return prediction, teacher
    if prediction.ndim == 5:
        b, seq, steps, groups, experts = prediction.shape
        if teacher.shape != (b, seq, groups, experts):
            raise ValueError(f"MLX lookahead teacher {teacher.shape} does not match prediction {prediction.shape}")
        return (prediction.transpose(0, 3, 1, 2, 4).reshape(b*groups, seq, steps, experts),
                teacher.transpose(0, 2, 1, 3).reshape(b*groups, seq, experts))
    if prediction.ndim != 4 or teacher.ndim != 3:
        raise ValueError("MLX lookahead expects 4-D/3-D or 5-D/4-D prediction/teacher")
    return prediction, teacher


def _lookahead_teacher(router, prediction, config):
    if prediction is None or prediction.ndim == 4:
        return mx.stop_gradient(router.mean(axis=2))
    start = int(config.prelude_layers)
    return mx.stop_gradient(router[:, :, start:start + prediction.shape[3], :])


def _lookahead_supervision_loss(
    lookahead_probs: mx.array | None,
    teacher_router_probs: mx.array | None,
    lookahead_steps: int,
    token_mask: mx.array | None = None,
) -> mx.array:
    groups = lookahead_probs.shape[3] if lookahead_probs is not None and lookahead_probs.ndim == 5 else 1
    lookahead_probs, teacher_router_probs = _flatten_lookahead_groups(lookahead_probs, teacher_router_probs)
    token_mask = _group_mask(token_mask, groups)
    if lookahead_probs is None:
        return mx.zeros(())
    if teacher_router_probs is None:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    _B, S, Kp1, _E = lookahead_probs.shape
    K = min(int(lookahead_steps or 0), Kp1 - 1)
    if K <= 0 or S <= 1:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    teacher = mx.stop_gradient(teacher_router_probs.astype(mx.float32))
    total = mx.zeros((), dtype=mx.float32)
    terms = 0
    for k in range(1, K + 1):
        if S - k <= 0:
            continue
        teacher_k = teacher[:, k:, :]
        pred_k = lookahead_probs[:, :-k, k, :].astype(mx.float32)
        total = total + _masked_mean(-(teacher_k * mx.log(mx.maximum(pred_k, 1e-9))).sum(axis=-1),
                                     _pair_mask(token_mask, k))
        terms += 1
    return total / max(terms, 1)


def _lookahead_topk_hit_loss(
    lookahead_probs: mx.array | None,
    teacher_router_probs: mx.array | None,
    lookahead_steps: int,
    num_experts_per_tok: int,
    token_mask: mx.array | None = None,
) -> mx.array:
    groups = lookahead_probs.shape[3] if lookahead_probs is not None and lookahead_probs.ndim == 5 else 1
    lookahead_probs, teacher_router_probs = _flatten_lookahead_groups(lookahead_probs, teacher_router_probs)
    token_mask = _group_mask(token_mask, groups)
    if lookahead_probs is None:
        return mx.zeros(())
    if teacher_router_probs is None:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    _B, S, Kp1, E = lookahead_probs.shape
    K = min(int(lookahead_steps or 0), Kp1 - 1)
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    if K <= 0 or S <= 1:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    teacher = mx.stop_gradient(teacher_router_probs.astype(mx.float32))
    total = mx.zeros((), dtype=mx.float32)
    terms = 0
    for k in range(1, K + 1):
        if S - k <= 0:
            continue
        teacher_k = teacher[:, k:, :]
        pred_k = lookahead_probs[:, :-k, k, :].astype(mx.float32)
        target_ids = mx.argpartition(-teacher_k, kth=top_k - 1, axis=-1)[:, :, :top_k]
        target_mask = (target_ids[..., None] == mx.arange(E)).astype(pred_k.dtype).sum(axis=-2)
        hit_mass = (pred_k * target_mask).sum(axis=-1)
        total = total + _masked_mean(-mx.log(mx.maximum(hit_mass / float(top_k), 1e-9)), _pair_mask(token_mask, k))
        terms += 1
    return total / max(terms, 1)


def _lookahead_union_loss(
    lookahead_probs: mx.array | None,
    teacher_router_probs: mx.array | None,
    lookahead_steps: int,
    num_experts_per_tok: int,
    token_mask: mx.array | None = None,
) -> mx.array:
    groups = lookahead_probs.shape[3] if lookahead_probs is not None and lookahead_probs.ndim == 5 else 1
    lookahead_probs, teacher_router_probs = _flatten_lookahead_groups(lookahead_probs, teacher_router_probs)
    token_mask = _group_mask(token_mask, groups)
    if lookahead_probs is None:
        return mx.zeros(())
    if teacher_router_probs is None:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    B, S, Kp1, E = lookahead_probs.shape
    K = min(int(lookahead_steps or 0), Kp1 - 1)
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    if K <= 0 or S <= K:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    teacher = mx.stop_gradient(teacher_router_probs.astype(mx.float32))
    union_target = mx.zeros((B, S - K, E), dtype=mx.float32)
    for k in range(1, K + 1):
        target_ids = mx.argpartition(-teacher[:, k:k + S - K, :], kth=top_k - 1, axis=-1)[:, :, :top_k]
        target_mask = (target_ids[..., None] == mx.arange(E)).astype(mx.float32).sum(axis=-2)
        union_target = mx.maximum(union_target, target_mask)
    pred_union = lookahead_probs[:, :S - K, 1:K + 1, :].astype(mx.float32).max(axis=2)
    mass = (pred_union * union_target).sum(axis=-1)
    target_count = mx.maximum(union_target.sum(axis=-1), 1.0)
    # Union over t+1..t+K: position t counts when it and t+K are real (right padding).
    return _masked_mean(-mx.log(mx.maximum(mass / target_count, 1e-9)), _pair_mask(token_mask, K))


def _offload_training_metrics(
    router_mean: mx.array | None,
    lookahead_probs: mx.array | None,
    lookahead_steps: int,
    num_experts_per_tok: int,
) -> dict[str, float]:
    lookahead_probs, router_mean = _flatten_lookahead_groups(lookahead_probs, router_mean)
    if router_mean is None:
        return {}
    B, S, E = router_mean.shape
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    top_ids = mx.argpartition(-router_mean, kth=top_k - 1, axis=-1)[:, :, :top_k]
    mx.eval(top_ids)
    import numpy as np

    ids_np = np.array(top_ids)
    out = {
        "expert_working_set": float(len(set(int(v) for v in ids_np.reshape(-1)))),
        "router_adjacent_jaccard": 1.0,
        "lookahead_topk_recall": 0.0,
        "lookahead_union_recall": 0.0,
    }
    if S > 1:
        vals = []
        for b in range(B):
            for t in range(S - 1):
                a = set(int(v) for v in ids_np[b, t])
                c = set(int(v) for v in ids_np[b, t + 1])
                vals.append(len(a & c) / max(len(a | c), 1))
        out["router_adjacent_jaccard"] = float(sum(vals) / max(len(vals), 1))
    if lookahead_probs is not None and lookahead_steps > 0 and S > 1:
        K = min(int(lookahead_steps), int(lookahead_probs.shape[2]) - 1)
        pred = lookahead_probs.astype(mx.float32)
        mx.eval(pred)
        pred_np = np.array(pred)
        recalls = []
        union_recalls = []
        pred_budget = max(top_k, min(E, top_k * max(1, K)))
        for k in range(1, K + 1):
            if S - k <= 0:
                continue
            pred_ids = np.argpartition(-pred_np[:, :-k, k, :], kth=top_k - 1, axis=-1)[:, :, :top_k]
            for b in range(B):
                for t in range(S - k):
                    target = set(int(v) for v in ids_np[b, t + k])
                    got = set(int(v) for v in pred_ids[b, t])
                    recalls.append(len(target & got) / float(top_k))
        if K > 0 and S - K > 0:
            pred_union_ids = np.argpartition(-pred_np[:, :S - K, 1:K + 1, :].max(axis=2),
                                             kth=pred_budget - 1, axis=-1)[:, :, :pred_budget]
            for b in range(B):
                for t in range(S - K):
                    target = set()
                    for k in range(1, K + 1):
                        target.update(int(v) for v in ids_np[b, t + k])
                    got = set(int(v) for v in pred_union_ids[b, t])
                    union_recalls.append(len(target & got) / max(len(target), 1))
        if recalls:
            out["lookahead_topk_recall"] = float(sum(recalls) / len(recalls))
        if union_recalls:
            out["lookahead_union_recall"] = float(sum(union_recalls) / len(union_recalls))
    return out


def _router_kl_anchor(current: mx.array, reference: mx.array | None, lambda_anchor: float) -> mx.array:
    if reference is None or lambda_anchor <= 0.0:
        return mx.zeros((), dtype=current.dtype)
    B = min(current.shape[0], reference.shape[0])
    S = min(current.shape[1], reference.shape[1])
    cur = current[:B, :S, :].astype(mx.float32)
    ref = mx.stop_gradient(reference[:B, :S, :].astype(mx.float32))
    eps = 1e-9
    kl = (cur * (mx.log(mx.maximum(cur, eps)) - mx.log(mx.maximum(ref, eps)))).sum(axis=-1).mean()
    return float(lambda_anchor) * kl


def draft_exit_active(model, config) -> bool:
    if float(getattr(config, "draft_exit_kl_coef", 0.0) or 0.0) <= 0.0 or not getattr(model, "training", False):
        return False
    if not getattr(config, "recurrent_enabled", True) or int(config.recurrent_depth) <= 1:
        return False
    return 1 <= int(getattr(config, "draft_exit_depth", 1)) < int(config.recurrent_depth)


def draft_exit_logits(model, input_ids, config, media=None):
    """Draft-depth FP32 logits, or None when the KL hook is off.

    Run *before* the full-depth forward: the full pass then owns the routing
    and recurrent diagnostics the other regularizers read.
    """
    if not draft_exit_active(model, config):
        return None
    logits, _lookahead, _cache = model(input_ids, logits_dtype=mx.float32,
                                       recurrent_depth=int(config.draft_exit_depth), **(media or {}))
    return logits


def draft_exit_kl(draft_logits: mx.array, full_logits: mx.array, keep: mx.array | None) -> mx.array:
    """Mean KL(full.detach() || draft) in FP32; ``keep`` [B, S] marks supervised logits."""
    teacher = mx.stop_gradient(full_logits.astype(mx.float32))
    t_log = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    d32 = draft_logits.astype(mx.float32)
    s_log = d32 - mx.logsumexp(d32, axis=-1, keepdims=True)
    kl = (mx.exp(t_log) * (t_log - s_log)).sum(axis=-1)
    if keep is None:
        return kl.mean()
    keep = keep.astype(mx.float32)
    return (kl * keep).sum() / mx.maximum(keep.sum(), 1.0)


def real_token_mask(labels: mx.array, *, aligned: bool = False) -> mx.array:
    """``[B, S]`` bool: input positions holding real tokens of a right-padded batch.

    Padding carries label -100 after each row's last supervised token, so a
    row's real tokens end at that token: ``labels[t]`` is token ``t`` itself
    (``aligned=False``, HF convention, the loss shifts) or token ``t+1``
    (``aligned=True``, labels pre-shifted onto the logits).  Prompt tokens
    (-100 before the reply) stay real: they are routed context.  Packed
    pretraining (no -100) gives an all-true mask.
    """
    labels = mx.array(labels) if not isinstance(labels, mx.array) else labels
    pos = mx.arange(labels.shape[1])
    last = mx.max(mx.where(labels != -100, pos[None, :], -1), axis=1)
    return pos[None, :] <= (last + (1 if aligned else 0))[:, None]


def _shifted_keep(labels: mx.array) -> mx.array:
    """Logit t is supervised when label t+1 is not -100 (CE shift)."""
    nxt = labels[:, 1:] != -100
    return mx.concatenate([nxt, mx.zeros((labels.shape[0], 1), dtype=mx.bool_)], axis=1)


def _chronos_regularized_loss(model, base_loss, lookahead_probs, config, router_ref=None, metrics=None,
                              draft_kl=None, token_mask=None, input_ids=None):
    """Return total loss and retain the actual weighted components for logging.

    ``token_mask`` ``[B, S]`` (1 = real token, 0 = right padding) keeps padding
    out of every router regularizer; ``None`` = all positions are real.
    ``input_ids`` ``[B, S]`` lets ``cluster_locality_mode="segment"`` find the
    documents of the batch (without it the window form is used)."""
    object.__setattr__(model, "_last_lookahead_probs", lookahead_probs)
    terms = {"primary_loss": base_loss.astype(mx.float32)}
    def add(name, coefficient, value):
        if coefficient:
            terms[name] = float(coefficient) * value.astype(mx.float32)

    recurrent = getattr(model, "last_recurrent_state", None) or {}
    for name, coefficient in (("ponder_loss", "ponder_loss_coef"), ("distill_loss", "recurrent_distill_coef"), ("stability_loss", "recurrent_stability_coef")):
        if name in recurrent:
            add("recurrent_" + name, getattr(config, coefficient, 0), recurrent[name])
    if "contraction" in recurrent:
        add("recurrent_contraction_loss", config.recurrent_contraction_coef,
            mx.maximum(recurrent["contraction"]-config.recurrent_contraction_budget, 0)**2)

    router = _collect_router_probs(model)
    if router is not None:
        router = router.astype(mx.float32)
        top_k = config.num_experts_per_tok
        add("balance_loss", config.lambda_balance, _load_balance_loss(router, top_k, token_mask))
        if float(getattr(config, "lambda_cluster_locality", 0.0) or 0.0) > 0.0:
            add("cluster_locality_loss", config.lambda_cluster_locality,
                _cluster_objective(router, config, token_mask, input_ids))
        if router.shape[1] > 1:
            mean = router.mean(axis=2)
            add("temporal_loss", config.lambda_temporal, _temporal_locality_loss(mean, token_mask))
            add("locality_loss", config.lambda_router_locality, _router_locality_loss(mean, top_k, token_mask))
            if lookahead_probs is not None and config.lambda_lookahead > 0 and config.lookahead_steps > 0:
                teacher = _lookahead_teacher(_first_iteration_router_probs(model).astype(mx.float32),
                                             lookahead_probs, config)
                add("lookahead_loss", config.lambda_lookahead,
                    _lookahead_supervision_loss(lookahead_probs, teacher, config.lookahead_steps, token_mask))
                add("lookahead_topk_loss", config.lambda_lookahead_topk,
                    _lookahead_topk_hit_loss(lookahead_probs, teacher, config.lookahead_steps, top_k, token_mask))
                add("lookahead_union_loss", config.lambda_lookahead_union,
                    _lookahead_union_loss(lookahead_probs, teacher, config.lookahead_steps, top_k, token_mask))
            add("anchor_loss", 1, _router_kl_anchor(mean, router_ref, config.lambda_router_anchor))
    if draft_kl is not None:
        add("draft_exit_kl", getattr(config, "draft_exit_kl_coef", 0.0), draft_kl)
    loss = sum(terms.values())
    object.__setattr__(model, "_last_loss_components", terms)
    object.__setattr__(model, "_last_stage_metrics", metrics or {"primary": base_loss})
    return loss


def _ce_stage_loss(model, input_ids, labels, modality_inputs=None, modality_mask=None):
    media = {"modality_inputs": modality_inputs, "modality_mask": modality_mask} if modality_inputs else None
    draft = draft_exit_logits(model, input_ids, model.config, media)
    logits, lookahead, _cache = model(input_ids, logits_dtype=mx.float32, **(media or {}))
    base = _masked_lm_loss(logits, labels)
    draft_kl = None if draft is None else draft_exit_kl(draft, logits, _shifted_keep(labels))
    return _chronos_regularized_loss(model, base, lookahead, model.config, metrics={"ce": base, "primary": base},
                                     draft_kl=draft_kl, token_mask=real_token_mask(labels), input_ids=input_ids)


def _tile_media(batch, copies: int) -> dict:
    """Model kwargs with the batch's media repeated ``copies`` times along the
    batch axis (chosen‖rejected pairs, GRPO groups); ``{}`` for text-only."""
    media = batch.get("modality_inputs") if isinstance(batch, dict) else None
    if not media:
        return {}
    from chronos.mlx.multimodal import _mx
    tiled = {k: mx.concatenate([_mx(v)] * copies, axis=0) if copies > 1 else _mx(v)
             for k, v in media.items() if v is not None}
    mask = batch.get("modality_mask")
    if mask is not None:
        mask = _mx(mask, mx.float32)
        mask = mx.concatenate([mask] * copies, axis=0) if copies > 1 else mask
    return {"modality_inputs": tiled, "modality_mask": mask}


def _frozen_media(frozen, media: dict) -> dict:
    """A text-only reference/teacher is a valid source for a multimodal
    student (Torch distill parity): pass media only if it can fuse them."""
    return media if media and getattr(frozen, "multimodal_fusion", None) is not None else {}


def _dpo_stage_loss(model, reference, batch, beta: float = 0.1, router_ref=None):
    x_c = _ids_to_mx(batch["x_chosen"])
    x_r = _ids_to_mx(batch["x_rejected"])
    y_c = _ids_to_mx(batch["y_chosen"])
    y_r = _ids_to_mx(batch["y_rejected"])
    m_c = _ids_to_mx(batch["mask_chosen"])
    m_r = _ids_to_mx(batch["mask_rejected"])
    x = mx.concatenate([x_c, x_r], axis=0)
    media = _tile_media(batch, 2)            # chosen and rejected answer the same prompt media
    logits, lookahead, _cache = model(x, logits_dtype=mx.float32, **media)
    B = x_c.shape[0]
    lp_c = _aligned_sequence_logprob(logits[:B], y_c, m_c)
    lp_r = _aligned_sequence_logprob(logits[B:], y_r, m_r)
    if reference is not None:
        ref_logits, _ref_lp, _ref_cache = reference(x, logits_dtype=mx.float32, **_frozen_media(reference, media))
        ref_c = mx.stop_gradient(_aligned_sequence_logprob(ref_logits[:B], y_c, m_c))
        ref_r = mx.stop_gradient(_aligned_sequence_logprob(ref_logits[B:], y_r, m_r))
        base = -mx.mean(nn.log_sigmoid(beta * ((lp_c - lp_r) - (ref_c - ref_r))))
    else:
        base = -mx.mean(nn.log_sigmoid(beta * (lp_c - lp_r)))
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref)


def _orpo_stage_loss(model, batch, beta: float = 0.1, lambda_or: float = 0.1, router_ref=None):
    x_c = _ids_to_mx(batch["x_chosen"])
    x_r = _ids_to_mx(batch["x_rejected"])
    y_c = _ids_to_mx(batch["y_chosen"])
    y_r = _ids_to_mx(batch["y_rejected"])
    m_c = _ids_to_mx(batch["mask_chosen"])
    m_r = _ids_to_mx(batch["mask_rejected"])
    x = mx.concatenate([x_c, x_r], axis=0)
    logits, lookahead, _cache = model(x, logits_dtype=mx.float32, **_tile_media(batch, 2))
    B = x_c.shape[0]
    lp_c = _aligned_sequence_logprob(logits[:B], y_c, m_c, average=True)
    lp_r = _aligned_sequence_logprob(logits[B:], y_r, m_r, average=True)
    nll = -mx.mean(lp_c)
    p_c = mx.clip(mx.exp(lp_c), 1e-6, 1.0 - 1e-6)
    p_r = mx.clip(mx.exp(lp_r), 1e-6, 1.0 - 1e-6)
    odds = (mx.log(p_c) - mx.log1p(-p_c)) - (
        mx.log(p_r) - mx.log1p(-p_r)
    )
    or_term = -lambda_or * mx.mean(nn.log_sigmoid(beta * odds))
    base = nll + or_term
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref, metrics={"primary": nll, "secondary": or_term})


def _distill_stage_loss(
    model,
    teacher,
    input_ids,
    labels,
    teacher_input_ids=None,
    alpha: float = 0.7,
    temperature: float = 4.0,
    router_ref=None,
    modality_inputs=None,
    modality_mask=None,
):
    media = {"modality_inputs": modality_inputs, "modality_mask": modality_mask} if modality_inputs else {}
    logits, lookahead, _cache = model(input_ids, logits_dtype=mx.float32, **media)
    t_logits, _tlp, _tcache = teacher(teacher_input_ids if teacher_input_ids is not None else input_ids,
                                      logits_dtype=mx.float32, **_frozen_media(teacher, media))
    V = min(logits.shape[-1], t_logits.shape[-1])
    S = min(logits.shape[1], t_logits.shape[1])
    logits_kd = logits[:, :S, :V]
    t_logits = mx.stop_gradient(t_logits[:, :S, :V])
    logits_kd = _f32(logits_kd)
    t_logits = _f32(t_logits)
    s_logp = logits_kd / temperature - mx.logsumexp(logits_kd / temperature, axis=-1, keepdims=True)
    t_prob = mx.softmax(t_logits / temperature, axis=-1)
    kd_tok = mx.sum(t_prob * (mx.log(mx.maximum(t_prob, 1e-12)) - s_logp), axis=-1)
    mask = (labels[:, :S] != -100).astype(mx.float32)
    kd = (temperature * temperature) * (kd_tok * mask).sum() / mx.maximum(mask.sum(), 1.0)
    ce = _masked_lm_loss(logits, labels)
    base = alpha * kd + (1.0 - alpha) * ce
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref,
                                     metrics={"primary": kd, "secondary": ce, "ce": ce},
                                     token_mask=real_token_mask(labels))


def _fit_teacher_input(ids: mx.array, teacher_vocab_size: int) -> mx.array:
    return mx.where(ids >= int(teacher_vocab_size), ids % int(teacher_vocab_size), ids)


def _grpo_stage_loss(model, reference, batch, beta: float = 0.04, router_ref=None):
    ids = _ids_to_mx(batch["ids"])
    response_mask = mx.array(batch["response_mask"]).astype(mx.float32)
    advantages = mx.array(batch["advantages"]).astype(mx.float32)
    media = _tile_media(batch, int(ids.shape[0]))   # one prompt's media for every trajectory
    logits, lookahead, _cache = model(ids, logits_dtype=mx.float32, **media)
    lp_tok = _shifted_token_logprobs(logits, ids)
    mask = response_mask[:, 1:]
    seq_lp = (lp_tok * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
    pg_loss = -mx.mean(mx.stop_gradient(advantages) * seq_lp)

    kl = mx.zeros((), dtype=logits.dtype)
    if reference is not None:
        ref_logits, _ref_lp, _ref_cache = reference(ids, logits_dtype=mx.float32, **_frozen_media(reference, media))
        ref_lp_tok = mx.stop_gradient(_shifted_token_logprobs(ref_logits, ids))
        log_ratio = ref_lp_tok - lp_tok
        kl = ((mx.exp(log_ratio) - log_ratio - 1.0) * mask).sum() / mx.maximum(mask.sum(), 1.0)
    base = pg_loss + float(beta) * kl
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref, metrics={"primary": pg_loss, "secondary": kl})


def _capture_router_ref(model, batch, stage: str) -> mx.array | None:
    if stage in {"pretrain", "grpo"}:
        return None
    media, media_mask = None, None
    if stage in {"sft", "distill"}:
        raw_ids, _labels, media, media_mask = batch_modalities(batch)
        ids = _ids_to_mx(raw_ids)
    elif stage in {"dpo", "orpo"}:
        ids = mx.concatenate([_ids_to_mx(batch["x_chosen"]), _ids_to_mx(batch["x_rejected"])], axis=0)
        tiled = _tile_media(batch, 2)
        media, media_mask = tiled.get("modality_inputs"), tiled.get("modality_mask")
    else:
        return None
    _logits, _lookahead, _cache = model(ids, modality_inputs=media, modality_mask=media_mask)
    router_4d = _collect_router_probs(model)
    if router_4d is None:
        return None
    ref = mx.stop_gradient(router_4d.mean(axis=2))
    mx.eval(ref)
    return ref


def _sample_next(logits: mx.array, temperature: float) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    probs = mx.softmax(logits / max(float(temperature), 1e-6), axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-9)).astype(mx.int32)


def _encoded_media(model, media: dict) -> dict:
    """Rollout kwargs with raw image/audio encoded once into ``*_embeds``.

    Every decode step re-applies the fusion (Torch ``generate`` parity); the
    encoder output is constant, so it is computed once and passed through
    the pre-encoded path (same tokens, same offsets, same validity)."""
    if not media:
        return {}
    fusion = getattr(model, "multimodal_fusion", None)
    inputs = dict(media["modality_inputs"])
    if fusion is not None:
        from chronos.mlx.multimodal import _mx
        for name, encoder in (("image", fusion.image_encoder), ("audio", fusion.audio_encoder)):
            raw = inputs.get(name)
            if raw is None or encoder is None or inputs.get(f"{name}_embeds") is not None:
                continue
            inputs.pop(name)
            inputs[f"{name}_embeds"] = mx.stop_gradient(encoder(_mx(raw, mx.float32)))
            mask = inputs.pop(f"{name}_mask", None)
            if mask is not None:
                inputs[f"{name}_embeds_mask"] = mask
    return {"modality_inputs": inputs, "modality_mask": media.get("modality_mask")}


def _rollout_one(model, prompt_ids: mx.array, max_gen_len: int, temperature: float, eos_token_id: int | None,
                 media: dict | None = None) -> tuple[mx.array, list[int]]:
    ids = prompt_ids
    cache = None
    generated: list[int] = []
    for _ in range(max(1, int(max_gen_len))):
        token_in = ids if cache is None else ids[:, -1:]
        logits, _lookahead, cache = model(token_in, cache=cache, **(media or {}))
        next_id = _sample_next(logits[:, -1, :], temperature).reshape(1, 1)
        mx.eval(next_id)
        tok = int(next_id.item())
        generated.append(tok)
        ids = mx.concatenate([ids, next_id], axis=1)
        if eos_token_id is not None and tok == int(eos_token_id):
            break
    mx.eval(ids)
    return ids, generated


def _grpo_prompt_media(item, args):
    """``(prompt_text, media)`` for a GRPO prompt: a string, or the
    ``{"prompt", "record"}`` dict of ``load_grpo_prompt_records`` (its
    image/audio resolved here, once per step, with the dataset loaders)."""
    if not isinstance(item, dict):
        return str(item), None
    from chronos.data.flexible_dataset import record_media
    get = (lambda k, d: getattr(args, k, d)) if args is not None else (lambda _k, d: d)
    found = record_media(item.get("record") or {}, image_size=int(get("image_size", 224)),
                         audio_sample_rate=int(get("audio_sample_rate", 16000)),
                         max_audio_seconds=float(get("max_audio_seconds", 30.0)),
                         media_error_policy=str(get("media_error_policy", "warn")))
    if found is None:
        return str(item["prompt"]), None
    media, mask = found
    batched = {k: v.unsqueeze(0) for k, v in media.items()}      # one prompt → batch of 1
    return str(item["prompt"]), {"modality_inputs": batched, "modality_mask": mask.unsqueeze(0)}


def _build_grpo_batch(model, prompt, tokenizer, reward_fn, args) -> dict:
    prompt, media = _grpo_prompt_media(prompt, args)
    rollout_media = _encoded_media(model, media)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    max_seq_len = int(getattr(args, "max_seq_len", 96) if args is not None else 96)
    max_gen_len = int(getattr(args, "max_gen_len", 24) if args is not None else 24)
    num_generations = int(getattr(args, "num_generations", 4) if args is not None else 4)
    temperature = float(getattr(args, "temperature", 1.0) if args is not None else 1.0)
    enc = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_seq_len)
    prompt_ids = _ids_to_mx(enc.input_ids)
    prompt_len = int(prompt_ids.shape[1])
    trajectories = []
    rewards = []
    for _ in range(max(1, num_generations)):
        full_ids, response_tokens = _rollout_one(
            model,
            prompt_ids,
            max_gen_len=max_gen_len,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            media=rollout_media,
        )
        trajectories.append(full_ids)
        rewards.append(float(reward_fn(prompt, response_tokens, tokenizer)))

    max_len = max(int(t.shape[1]) for t in trajectories)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ids_rows = []
    mask_rows = []
    for full_ids in trajectories:
        row = [int(v) for v in full_ids[0].tolist()]
        pad = [pad_id] * (max_len - len(row))
        ids_rows.append(row + pad)
        mask_rows.append(([0] * prompt_len + [1] * max(0, len(row) - prompt_len)) + [0] * len(pad))
    rewards_mx = mx.array(rewards, dtype=mx.float32)
    adv = rewards_mx - rewards_mx.mean()
    denom = mx.maximum(mx.sqrt(((adv * adv).mean())), 1e-6)
    advantages = adv / denom
    mx.eval(advantages)
    out = {
        "ids": ids_rows,
        "response_mask": mask_rows,
        "advantages": advantages.tolist(),
    }
    if media:      # the policy update re-encodes (encoder gradients), tiled per trajectory
        out.update(media)
    return out


@dataclass
class MLXStageResult:
    checkpoint_path: str
    steps: int
    last_loss: float
    dtype: str
    elapsed_s: float = 0.0
    total_steps: int = 0
    stopped: bool = False
    checkpoint_saved: bool = False
    rollbacks: int = 0
    failure_reason: str = ""


def _is_mx_array(value) -> bool:
    return isinstance(value, mx.array)


def _tree_copy(tree):
    if isinstance(tree, dict):
        return {k: _tree_copy(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_copy(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_copy(v) for v in tree)
    if _is_mx_array(tree):
        # MLX arrays are immutable; copy the tree without synchronizing once
        # per tensor. Snapshots retain already evaluated arrays until rollback.
        return mx.array(tree)
    return tree


def _tree_arrays(tree) -> list[mx.array]:
    arrays: list[mx.array] = []
    if isinstance(tree, dict):
        for value in tree.values():
            arrays.extend(_tree_arrays(value))
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            arrays.extend(_tree_arrays(value))
    elif _is_mx_array(tree):
        arrays.append(tree)
    return arrays


def _tree_finite_check(tree) -> mx.array:
    """Lazy device boolean: every floating array in ``tree`` is finite.

    Evaluate it together with the arrays it checks (one ``mx.eval`` for
    values + guard) instead of a separate sync per check.
    """
    checks = [mx.all(mx.isfinite(arr)) for arr in _tree_arrays(tree)
              if mx.issubdtype(arr.dtype, mx.inexact)]
    if not checks:
        return mx.array(True)
    return mx.all(mx.stack(checks)) if len(checks) > 1 else checks[0]


def _tree_all_finite(tree) -> bool:
    check = _tree_finite_check(tree)
    mx.eval(check)
    return bool(check.item())


def _scalar_is_finite(value: mx.array) -> bool:
    check = mx.all(mx.isfinite(value))
    mx.eval(check)
    return bool(check.item())


def _add_grads(a, b):
    if a is None:
        return tree_map(lambda g: g.astype(mx.float32), b)
    return tree_map(lambda x, y: x + y.astype(mx.float32), a, b)


def _scale_grads(grads, scale: float):
    return tree_map(lambda g: g * float(scale), grads)


def _planned_total_steps(data_iter, epochs: int, max_steps) -> int:
    planned = 0
    try:
        planned = max(1, int(len(data_iter)) * max(1, int(epochs)))
    except Exception:
        planned = 0
    if max_steps is not None:
        cap = max(1, int(max_steps))
        return min(planned, cap) if planned else cap
    return planned


def _normalize_mlx_dtype_name(dtype_name: str | None) -> str:
    value = (dtype_name or "auto").strip().lower()
    if value in {"auto", ""}:
        return "bfloat16" if mx is not None and hasattr(mx, "bfloat16") else "float32"
    if value in {"fp16", "float16", "half"}:
        return "float16"
    if value in {"bf16", "bfloat16"}:
        return "bfloat16"
    return "float32"


def _mlx_dtype_from_name(dtype_name: str | None):
    name = _normalize_mlx_dtype_name(dtype_name)
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16 if hasattr(mx, "bfloat16") else mx.float32
    return mx.float32


def _load_mlx_from_checkpoint(config, checkpoint_path: str | None, dtype_name: str | None = None):
    """Build the MLX model directly in the policy compute dtype.

    ``chronos.precision.resolve_mlx`` decides the dtype once (GPU: requested
    or BF16; CPU: FP32 because CPU GatherMM is FP32-only).  Each tensor is
    converted once from the checkpoint dtype; no FP32 staging copy.
    """
    from chronos.precision import resolve_mlx
    policy = resolve_mlx(dtype_name or "auto")
    pt = ChronosForCausalLM(config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
        load_state_dict_controlled(pt, state)
    model = ChronosMLXModel.from_chronos_pytorch(pt, config, dtype=policy.compute)
    mx.eval(model.parameters())
    model.config = config
    model._chronos_precision = policy
    return model


def _save_mlx_checkpoint(model, config, out_path: str, stage: str) -> None:
    """Save in the model's compute dtype (BF16 stays BF16, bit-exact)."""
    pt = ChronosForCausalLM(config)
    state = mlx_state_to_torch(model, config)
    dtype = next(iter(state.values())).dtype
    pt.to(dtype).load_state_dict(state, strict=True)
    save_state_dict_with_config(pt, out_path, config, stage=stage, half=False)


def run_mlx_stage(
    *,
    stage: str,
    config,
    checkpoint_path: str | None,
    save_dir: str,
    loader=None,
    validation_loader=None,
    prompts=None,
    teacher_path: str | None = None,
    args=None,
    progress_callback=None,
    stop_event=None,
) -> MLXStageResult:
    dtype_name = _normalize_mlx_dtype_name(getattr(args, "dtype", "auto") if args is not None else "auto")
    model = _load_mlx_from_checkpoint(config, checkpoint_path, dtype_name=dtype_name)
    reference = None
    if stage in {"dpo", "grpo"} and checkpoint_path:
        reference = _load_mlx_from_checkpoint(config, checkpoint_path, dtype_name=dtype_name)
    teacher = None
    if stage == "distill":
        if not teacher_path:
            raise ValueError("MLX distill requires teacher_path")
        try:
            teacher_config, _teacher_sources = chronos_config_from_checkpoint(
                teacher_path,
                require_unsniffable=True,
            )
        except Exception:
            teacher_config = config
        teacher = _load_mlx_from_checkpoint(teacher_config, teacher_path, dtype_name=dtype_name)

    base_lr = float(getattr(args, "learning_rate", 5e-5) if args is not None else 5e-5)
    default_eps = 1e-8
    optimizer = MixedPrecisionAdamW(
        learning_rate=base_lr,
        weight_decay=float(getattr(args, "weight_decay", 0.01) if args is not None else 0.01),
        eps=float(getattr(args, "adam_eps", default_eps) if args is not None else default_eps),
    )
    optimizer.init(model.trainable_parameters())
    model.train()
    for frozen in (reference, teacher):
        if frozen is not None:
            frozen.eval()
    max_steps = getattr(args, "steps", None) if args is not None else None
    if max_steps is not None and int(max_steps) <= 0:
        max_steps = None
    epochs = int(getattr(args, "epochs", 1) if args is not None else 1)
    accum = max(1, int(getattr(args, "accumulation_steps", 1) if args is not None else 1))
    grad_clip = float(getattr(args, "grad_clip", 0.0) if args is not None else 0.0)
    save_interval = max(1, int(getattr(args, "save_interval", 1000) if args is not None else 1000))
    rollback_limit = max(1, int(getattr(args, "mlx_rollback_limit", 3) if args is not None else 3))
    beta = float(getattr(args, "beta", 0.1) if args is not None else 0.1)
    lambda_or = float(getattr(args, "lambda_or", 0.1) if args is not None else 0.1)
    alpha = float(getattr(args, "alpha", 0.7) if args is not None else 0.7)
    temperature = float(getattr(args, "temperature", 4.0) if args is not None else 4.0)
    log_interval = max(1, int(getattr(args, "log_interval", 10) if args is not None else 10))
    steps = 0
    last_loss = 0.0
    best_loss = float("inf")
    failure_reason = ""
    grad_norm = 0.0
    token_count = 0
    last_log_tokens = 0
    t0 = time.monotonic()
    last_log_t = t0
    last_log_step = 0
    router_ref = None

    def _should_stop() -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    if stage == "grpo":
        from transformers import AutoTokenizer
        import chronos.deps
        from chronos.trainer.reward import build_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
        reward_fn = build_reward_fn(getattr(args, "reward", "toy") if args is not None else "toy")
        data_iter = prompts or []
    else:
        data_iter = loader
    if data_iter is None:
        raise ValueError(f"MLX {stage} requires loader/prompts")
    total_steps = _planned_total_steps(data_iter, epochs, max_steps)
    total_updates = max(1, (total_steps + accum - 1) // accum)
    optimizer.learning_rate = get_lr(1, total_updates, base_lr)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"{stage}_{config.hidden_size}_moe.pth" if stage != "pretrain" else f"chronos_{config.hidden_size}_moe.pth",
    )
    checkpoint_stage = "chronos" if stage == "pretrain" else stage
    checkpoint_saved = False
    rollback_count = 0
    consecutive_rollbacks = 0
    lr_scale = 1.0
    pending_grads = None
    pending_count = 0
    rollback_model_state = _tree_copy(model.parameters())
    rollback_optimizer_state = _tree_copy(optimizer.state)
    rollback_loss = 0.0
    best_validation = float("inf")

    def _emit(event: dict) -> None:
        if progress_callback is not None:
            progress_callback(event)

    def _discard_pending() -> None:
        nonlocal pending_grads, pending_count
        pending_grads = None
        pending_count = 0

    def _restore_rollback_state() -> None:
        nonlocal last_loss
        _discard_pending()
        last_loss = rollback_loss
        model.update(_tree_copy(rollback_model_state))
        optimizer.state = _tree_copy(rollback_optimizer_state)
        mx.eval(model.parameters(), optimizer.state)

    def _save_checkpoint(reason: str) -> None:
        nonlocal checkpoint_saved
        _save_mlx_checkpoint(model, config, out_path, stage=checkpoint_stage)
        checkpoint_saved = True
        _emit({
            "event": "save",
            "stage": stage,
            "steps": total_steps,
            "step": steps,
            "loss": last_loss,
            "elapsed_s": time.monotonic() - t0,
            "dtype": dtype_name,
            "checkpoint_path": out_path,
            "reason": reason,
        })

    def loss_fn(m, batch):
        if stage in {"pretrain", "sft"}:
            ids, labels, media, media_mask = batch_modalities(batch)
            if media:
                return _ce_stage_loss(m, _ids_to_mx(ids), _ids_to_mx(labels),
                                      modality_inputs=media, modality_mask=media_mask)
            return _ce_stage_loss(m, _ids_to_mx(ids), _ids_to_mx(labels))
        if stage == "dpo":
            return _dpo_stage_loss(m, reference, batch, beta=beta, router_ref=router_ref)
        if stage == "orpo":
            return _orpo_stage_loss(m, batch, beta=beta, lambda_or=lambda_or, router_ref=router_ref)
        if stage == "distill":
            ids, labels, media, media_mask = batch_modalities(batch)
            ids = _ids_to_mx(ids)
            labels = _ids_to_mx(labels)
            teacher_vocab = int(getattr(getattr(teacher, "config", config), "vocab_size", config.vocab_size))
            return _distill_stage_loss(
                m,
                teacher,
                ids,
                labels,
                teacher_input_ids=_fit_teacher_input(ids, teacher_vocab),
                alpha=alpha,
                temperature=temperature,
                router_ref=router_ref,
                modality_inputs=media,
                modality_mask=media_mask,
            )
        if stage == "grpo":
            return _grpo_stage_loss(m, reference, batch, beta=beta, router_ref=router_ref)
        raise ValueError(f"unknown MLX stage: {stage}")

    eager_grad = nn.value_and_grad(model, loss_fn)
    text_grad = None
    if stage in {"pretrain", "sft"}:
        # Text batches take the compiled micro step (``CHRONOS_MLX_COMPILE``,
        # default auto); media batches and other stages stay eager.
        from chronos.mlx.training.compiled import make_grad_fn
        text_grad = make_grad_fn(model, lambda m, ids, labels: _ce_stage_loss(m, ids, labels),
                                 mode=getattr(args, "mlx_compile", None) if args is not None else None,
                                 log=print)

    def grad_fn(m, batch):
        if text_grad is not None:
            ids, labels, media, _mask = batch_modalities(batch)
            if not media:
                return text_grad(m, _ids_to_mx(ids), _ids_to_mx(labels))
        return eager_grad(m, batch)

    _emit({
        "event": "start",
        "stage": stage,
        "dtype": dtype_name,
        "step": 0,
        "steps": total_steps,
        "loss": 0.0,
        "elapsed_s": 0.0,
        "lr": base_lr,
        "accumulation_steps": accum,
        "save_interval": save_interval,
        "optimizer": "AdamW FP32 masters/moments, betas=(0.9,0.95), bias correction",
        "optimizer_updates": total_updates,
    })

    def rollback(reason, epoch):
        nonlocal rollback_count, consecutive_rollbacks, lr_scale, failure_reason
        rollback_count += 1
        consecutive_rollbacks += 1
        lr_scale *= .5
        _restore_rollback_state()
        if consecutive_rollbacks >= rollback_limit:
            failure_reason = reason
        _emit({"event": "rollback", "stage": stage, "epoch": epoch, "epochs": epochs,
               "step": steps, "steps": total_steps, "loss": last_loss,
               "elapsed_s": time.monotonic()-t0, "dtype": dtype_name,
               "lr": base_lr*lr_scale, "rollbacks": rollback_count, "reason": reason})

    def apply_pending(epoch):
        nonlocal rollback_model_state, rollback_optimizer_state, rollback_loss
        nonlocal consecutive_rollbacks, grad_norm
        if pending_grads is None:
            return True
        update_grads = _scale_grads(pending_grads, 1.0 / max(1, pending_count))
        # Accumulation and this norm are FP32 even when forward weights are BF16.
        update_grads, norm = optim.clip_grad_norm(update_grads, grad_clip if grad_clip > 0 else float("inf"))
        grads_ok = _tree_finite_check((norm, update_grads))
        optimizer.learning_rate = get_lr(int(optimizer.step.item()) + 1, total_updates, base_lr * lr_scale)
        previous_snapshot = (rollback_model_state, rollback_optimizer_state, rollback_loss)
        rollback_model_state = _tree_copy(model.parameters())
        rollback_optimizer_state = _tree_copy(optimizer.state)
        rollback_loss = last_loss
        optimizer.update(model, update_grads)
        # One device sync for clip + guard + update + post-update guard. A bad
        # gradient only wastes this update's compute: both guards are read
        # before the result is kept, and the rollback targets are unchanged.
        state_ok = _tree_finite_check((model.parameters(), optimizer.state))
        mx.eval(model.parameters(), optimizer.state, norm, grads_ok, state_ok)
        grad_norm = float(norm.item())
        if not bool(grads_ok.item()):
            rollback_model_state, rollback_optimizer_state, rollback_loss = previous_snapshot
            rollback("non_finite_accumulated_grad", epoch)
            return False
        if not bool(state_ok.item()):
            rollback("non_finite_parameters_or_optimizer", epoch)
            return False
        consecutive_rollbacks = 0
        _discard_pending()
        return True

    def evaluate_validation(epoch):
        nonlocal best_validation
        if validation_loader is None or stage not in {"pretrain", "sft"}:
            return
        total_nll, valid_tokens = 0.0, 0
        was_training = model.training
        model.eval()
        try:
            for vbatch in validation_loader:
                ids, labels, media, media_mask = batch_modalities(vbatch)
                labels = _ids_to_mx(labels)
                logits, _, _ = model(_ids_to_mx(ids), logits_dtype=mx.float32,
                                     modality_inputs=media, modality_mask=media_mask)
                ce = _masked_lm_loss(logits, labels)
                count = mx.sum(labels[:, 1:] != -100)
                mx.eval(ce, count)
                total_nll += float(ce.item()) * int(count.item())
                valid_tokens += int(count.item())
        finally:
            model.train(was_training)
        if valid_tokens:
            value = total_nll / valid_tokens
            _emit({"event": "validation", "stage": stage, "epoch": epoch,
                   "step": steps, "steps": total_steps, "ce": value,
                   "tokens": valid_tokens, "elapsed_s": time.monotonic()-t0})
            if value < best_validation:
                best_validation = value
                _save_mlx_checkpoint(model, config, out_path.replace(".pth", ".best.pth"), checkpoint_stage)

    stop_outer = False
    for epoch_idx in range(epochs):
        for batch in data_iter:
            if _should_stop():
                break
            if router_ref is None and stage in {"sft", "dpo", "orpo", "distill"}:
                router_ref = _capture_router_ref(model, batch, stage)
            train_batch = batch
            if stage == "grpo":
                if _should_stop():
                    break
                train_batch = _build_grpo_batch(
                    model,
                    batch if isinstance(batch, dict) else str(batch),
                    tokenizer,
                    reward_fn,
                    args,
                )
            model.recurrent_controller.set_training_progress(steps + 1, total_steps)
            loss, grads = grad_fn(model, train_batch)
            supervised = stage in {"pretrain", "sft", "distill"}
            # Loss, gradients, their finite guard, the FP32 accumulation and the
            # supervised-token count are materialised by one device sync.
            finite_check = _tree_finite_check((loss, grads))
            candidate = _add_grads(pending_grads, grads)
            n_supervised = (mx.sum(_ids_to_mx(batch_modalities(train_batch)[1])[:, 1:] != -100)
                            if supervised else None)
            mx.eval(loss, finite_check, candidate, *([n_supervised] if supervised else []))
            steps += 1
            finite = bool(finite_check.item())
            if not finite:
                rollback("non_finite_loss_or_grad", epoch_idx + 1)
                if consecutive_rollbacks >= rollback_limit:
                    stop_outer = True
                    break
                if max_steps is not None and steps >= int(max_steps):
                    break
                continue

            observed_loss = float(loss.item())
            # A catastrophe guard, not a convergence/quality criterion. Only
            # nonnegative supervised objectives have a meaningful ratio here.
            if stage in {"pretrain", "sft", "distill"} and observed_loss > max(100.0, 20.0*best_loss):
                rollback("finite_loss_divergence", epoch_idx + 1)
                failure_reason = "finite_loss_divergence"
                stop_outer = True
                break
            last_loss = observed_loss
            best_loss = min(best_loss, observed_loss)
            if supervised:
                token_count += int(n_supervised.item())
            pending_grads = candidate          # adopted only after the guard passed
            pending_count += 1
            should_update = pending_count >= accum or (max_steps is not None and steps >= int(max_steps))
            if should_update:
                apply_pending(epoch_idx + 1)
                if consecutive_rollbacks >= rollback_limit:
                    stop_outer = True
                    break

            if steps % save_interval == 0:
                _save_checkpoint("interval")

            if steps == 1 or steps % log_interval == 0:
                now = time.monotonic()
                delta_steps = steps - last_log_step
                interval_s = max(now - last_log_t, 1e-6)
                tps = delta_steps / interval_s
                tokens_per_s = (token_count-last_log_tokens) / interval_s
                last_log_tokens = token_count
                last_log_t = now
                last_log_step = steps
                router_4d = _collect_router_probs(model)
                prediction = getattr(model, "_last_lookahead_probs", None)
                router_mean = _lookahead_teacher(router_4d, prediction, config) if router_4d is not None else None
                offload_metrics = _offload_training_metrics(
                    router_mean,
                    getattr(model, "_last_lookahead_probs", None),
                    int(getattr(config, "lookahead_steps", 0) or 0),
                    int(getattr(config, "num_experts_per_tok", 1) or 1),
                )
                components = {k: float(v.item()) for k, v in getattr(model, "_last_loss_components", {}).items()}
                stage_metrics = {k: float(v.item()) for k, v in getattr(model, "_last_stage_metrics", {}).items()}
                lookahead_loss = sum(v for k, v in components.items() if k.startswith("lookahead_"))
                recurrent_loss = sum(v for k, v in components.items() if k.startswith("recurrent_"))
                defaults = {
                    "pretrain": (last_loss-components.get("primary_loss", last_loss), components.get("temporal_loss", 0)),
                    "sft": (lookahead_loss, components.get("anchor_loss", 0)),
                    "dpo": (components.get("anchor_loss", 0), lookahead_loss),
                }
                secondary, tertiary = defaults.get(stage, (0, components.get("anchor_loss", 0)))
                _emit({
                    "event": "step",
                    "stage": stage,
                    "epoch": epoch_idx + 1,
                    "epochs": epochs,
                    "step": steps,
                    "steps": total_steps,
                    "loss": last_loss,
                    **stage_metrics,
                    "secondary": stage_metrics.get("secondary", secondary),
                    "tertiary": tertiary,
                    "aux": last_loss-components.get("primary_loss", last_loss),
                    "temporal": components.get("temporal_loss", 0),
                    "loss_components": components,
                    "recurrent_loss": recurrent_loss,
                    "lookahead_loss": lookahead_loss,
                    "grad_norm": grad_norm,
                    "optimizer_step": int(optimizer.step.item()),
                    "tps": tps,
                    "tokens_per_s": tokens_per_s,
                    "elapsed_s": now - t0,
                    "dtype": dtype_name,
                    "lr": float(optimizer.learning_rate.item()) if "learning_rate" in optimizer.state else base_lr * lr_scale,
                    "rollbacks": rollback_count,
                    **offload_metrics,
                })
            if max_steps is not None and steps >= int(max_steps):
                break
        if not stop_outer and not _should_stop():
            evaluate_validation(epoch_idx+1)
        if stop_outer or _should_stop() or (max_steps is not None and steps >= int(max_steps)):
            break

    stopped = _should_stop()
    if pending_grads is not None and pending_count > 0 and not stop_outer:
        if not apply_pending(epochs):
            stop_outer = True
            failure_reason = "final_update_failed"
        elif not stopped:
            evaluate_validation(epochs)
    if stop_outer:
        # Keep a failed run's recovery artifact distinct from a successful model.
        out_path = out_path.replace(".pth", ".recovery.pth")
    _save_checkpoint("stop" if stopped else (failure_reason if stop_outer else "final"))
    elapsed = time.monotonic() - t0
    _emit({
        "event": "stopped" if stopped else ("failed" if stop_outer else "complete"),
        "stage": stage,
        "step": steps,
        "steps": total_steps,
        "loss": last_loss,
        "elapsed_s": elapsed,
        "dtype": dtype_name,
        "checkpoint_path": out_path if checkpoint_saved else "",
        "checkpoint_saved": checkpoint_saved,
        "rollbacks": rollback_count,
        "reason": failure_reason,
    })
    if stopped:
        print(
            f"[MLX {stage}] stopped steps={steps} loss={last_loss:.4f} "
            f"elapsed={elapsed:.2f}s saved={out_path}"
        )
    elif stop_outer:
        print(
            f"[MLX {stage}] stopped after rollback limit steps={steps} loss={last_loss:.4f} "
            f"elapsed={elapsed:.2f}s saved={out_path if checkpoint_saved else 'n/a'}"
        )
    else:
        print(
            f"[MLX {stage}] dtype={dtype_name} steps={steps} loss={last_loss:.4f} "
            f"elapsed={elapsed:.2f}s saved={out_path}"
        )
    return MLXStageResult(
        out_path,
        steps,
        last_loss,
        dtype_name,
        elapsed,
        total_steps=total_steps,
        stopped=stopped or stop_outer,
        checkpoint_saved=checkpoint_saved,
        rollbacks=rollback_count,
        failure_reason=failure_reason,
    )
