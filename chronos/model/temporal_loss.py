import torch
import torch.nn.functional as F


def temporal_locality_loss(expert_probs_seq: torch.Tensor) -> torch.Tensor:
    """
    Penalizes large routing distribution shifts between adjacent time steps.

    L_temporal = (1/T) * sum_{t=2}^{T} ||E_t - E_{t-1}||_2^2

    Args:
        expert_probs_seq: [B, T, num_experts]  per-step routing probabilities
    Returns:
        scalar loss
    """
    diff = expert_probs_seq[:, 1:, :] - expert_probs_seq[:, :-1, :]
    return (diff ** 2).sum(dim=-1).mean()


def router_locality_loss(
    expert_probs_seq: torch.Tensor,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Encourage adjacent tokens to keep overlapping routed expert sets.

    The loss maximizes probability mass at time t+1 assigned to the top-k set
    from time t. It is a differentiable proxy for expert working-set locality,
    which directly reduces lazy/offload cache churn.
    """
    if expert_probs_seq is None or expert_probs_seq.shape[1] <= 1:
        return expert_probs_seq.new_zeros(()) if expert_probs_seq is not None else torch.tensor(0.0)
    B, S, E = expert_probs_seq.shape
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    prev = expert_probs_seq[:, :-1, :].detach()
    cur = expert_probs_seq[:, 1:, :]
    target_ids = torch.topk(prev, k=top_k, dim=-1).indices
    target_mask = F.one_hot(target_ids, num_classes=E).sum(dim=-2).to(cur.dtype)
    overlap_mass = (cur * target_mask).sum(dim=-1).clamp_min(1e-9)
    return (-torch.log(overlap_mass / float(top_k))).mean()


def router_offload_metrics(
    router_probs: torch.Tensor,
    lookahead_probs: torch.Tensor = None,
    lookahead_steps: int = 0,
    num_experts_per_tok: int = 1,
) -> dict[str, float]:
    """Return detached routing metrics used by logs and checkpoint verify."""
    if router_probs is None or router_probs.shape[1] <= 0:
        return {}
    with torch.no_grad():
        probs = router_probs.detach()
        B, S, E = probs.shape
        top_k = max(1, min(int(num_experts_per_tok or 1), E))
        top_ids = torch.topk(probs, k=top_k, dim=-1).indices
        flat = top_ids.reshape(-1)
        working_set = float(torch.unique(flat).numel())
        out = {
            "expert_working_set": working_set,
            "router_adjacent_jaccard": 1.0,
            "lookahead_topk_recall": 0.0,
            "lookahead_union_recall": 0.0,
        }
        if S > 1:
            prev = F.one_hot(top_ids[:, :-1, :], num_classes=E).sum(dim=-2).bool()
            cur = F.one_hot(top_ids[:, 1:, :], num_classes=E).sum(dim=-2).bool()
            inter = (prev & cur).sum(dim=-1).float()
            union = (prev | cur).sum(dim=-1).float().clamp_min(1.0)
            out["router_adjacent_jaccard"] = float((inter / union).mean().item())
        if lookahead_probs is not None and lookahead_steps > 0 and S > 1:
            K = min(int(lookahead_steps), int(lookahead_probs.shape[2]) - 1)
            recalls = []
            union_recalls = []
            pred_budget = max(top_k, min(E, top_k * max(1, K)))
            for k in range(1, K + 1):
                if S - k <= 0:
                    continue
                target = F.one_hot(top_ids[:, k:, :], num_classes=E).sum(dim=-2).bool()
                pred_ids = torch.topk(lookahead_probs[:, :-k, k, :], k=top_k, dim=-1).indices
                pred = F.one_hot(pred_ids, num_classes=E).sum(dim=-2).bool()
                recalls.append(((pred & target).sum(dim=-1).float() / float(top_k)).mean())
            if K > 0 and S - K > 0:
                union_target = torch.zeros(B, S - K, E, dtype=torch.bool, device=probs.device)
                for k in range(1, K + 1):
                    union_target |= F.one_hot(top_ids[:, k:k + S - K, :], num_classes=E).sum(dim=-2).bool()
                union_target_count = union_target.sum(dim=-1).float().clamp_min(1.0)
                pred_ids = torch.topk(lookahead_probs[:, :S - K, 1:, :].amax(dim=2), k=pred_budget, dim=-1).indices
                pred_union = F.one_hot(pred_ids, num_classes=E).sum(dim=-2).bool()
                union_recalls.append(((pred_union & union_target).sum(dim=-1).float() / union_target_count).mean())
            if recalls:
                out["lookahead_topk_recall"] = float(torch.stack(recalls).mean().item())
            if union_recalls:
                out["lookahead_union_recall"] = float(torch.stack(union_recalls).mean().item())
        return out


def load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """
    Standard auxiliary load-balancing loss (same formulation as minimind).

    Args:
        router_logits: [N, num_experts]  (N = B*S flattened)
    Returns:
        scalar loss
    """
    scores = F.softmax(router_logits, dim=-1)
    _, topk_idx = torch.topk(scores, k=num_experts_per_tok, dim=-1, sorted=False)
    load = F.one_hot(topk_idx, num_experts).float().mean(0)
    return (load * scores.mean(0)).sum() * num_experts


def lookahead_supervision_loss(
    lookahead_probs: torch.Tensor,
    teacher_router_probs: torch.Tensor,
    lookahead_steps: int,
) -> torch.Tensor:
    """
    Teacher-forcing loss for the LookaheadRouter.

    At time t the router predicts the routing distribution at t+k for
    k ∈ {1..lookahead_steps}. We supervise those predictions with the
    *actual* layer-averaged router probabilities from step t+k (stop-grad).

    Formulation (soft-target cross-entropy, equivalent to forward KL
    up to a teacher entropy constant):

        L = (1 / K) · Σ_{k=1..K} [ - mean_{t} Σ_e P_{t+k}[e] · log Q_t^(k)[e] ]

    Args:
        lookahead_probs:       [B, S, K+1, E] — softmax'd output of
                                LookaheadRouter. Index 0 = current step,
                                indices 1..K = future-step predictions.
        teacher_router_probs:  [B, S, E]       — layer-averaged real router
                                probabilities. Caller MUST pass a detached
                                (stop-grad) tensor.
        lookahead_steps:       K, how many future steps to supervise.

    Returns:
        scalar loss. Zero when the sequence is too short to form a pair.
    """
    if lookahead_probs is None or teacher_router_probs is None:
        return lookahead_probs.new_zeros(()) if lookahead_probs is not None else \
               teacher_router_probs.new_zeros(())

    B, S, Kp1, E = lookahead_probs.shape
    K = min(lookahead_steps, Kp1 - 1)
    if K <= 0 or S <= 1:
        return lookahead_probs.new_zeros(())

    teacher = teacher_router_probs  # caller is responsible for .detach()
    total = lookahead_probs.new_zeros(())
    valid_terms = 0
    for k in range(1, K + 1):
        if S - k <= 0:
            continue
        teacher_k = teacher[:, k:, :]                  # [B, S-k, E]
        pred_k = lookahead_probs[:, :-k, k, :]         # [B, S-k, E]
        # Soft-target CE: −Σ_e P · log Q
        log_pred = pred_k.clamp_min(1e-9).log()
        ce = -(teacher_k * log_pred).sum(dim=-1)        # [B, S-k]
        total = total + ce.mean()
        valid_terms += 1
    if valid_terms == 0:
        return lookahead_probs.new_zeros(())
    return total / valid_terms


def lookahead_topk_hit_loss(
    lookahead_probs: torch.Tensor,
    teacher_router_probs: torch.Tensor,
    lookahead_steps: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """
    Differentiable top-k recall proxy for prefetch quality.

    For each future offset k, take the real future router top-k set as a
    stop-grad target and maximize the predicted probability mass assigned to
    that set. This complements the soft CE objective with the operational
    metric that matters to offloading: did the predicted buffer include the
    experts that will be needed soon?
    """
    if lookahead_probs is None or teacher_router_probs is None:
        return lookahead_probs.new_zeros(()) if lookahead_probs is not None else \
               teacher_router_probs.new_zeros(())

    B, S, Kp1, E = lookahead_probs.shape
    K = min(int(lookahead_steps), Kp1 - 1)
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    if K <= 0 or S <= 1:
        return lookahead_probs.new_zeros(())

    teacher = teacher_router_probs.detach()
    total = lookahead_probs.new_zeros(())
    valid_terms = 0
    for k in range(1, K + 1):
        if S - k <= 0:
            continue
        teacher_k = teacher[:, k:, :]
        pred_k = lookahead_probs[:, :-k, k, :]
        target_ids = torch.topk(teacher_k, k=top_k, dim=-1).indices
        target_mask = F.one_hot(target_ids, num_classes=E).sum(dim=-2).to(pred_k.dtype)
        hit_mass = (pred_k * target_mask).sum(dim=-1).clamp_min(1e-9)
        # Divide by top_k so a uniform predictor has comparable scale across
        # top-k values; minimizing -log mass rewards covering the set.
        total = total + (-torch.log(hit_mass / float(top_k))).mean()
        valid_terms += 1
    if valid_terms == 0:
        return lookahead_probs.new_zeros(())
    return total / valid_terms


def lookahead_union_loss(
    lookahead_probs: torch.Tensor,
    teacher_router_probs: torch.Tensor,
    lookahead_steps: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """Predict the union of experts needed over the future prefetch window."""
    if lookahead_probs is None or teacher_router_probs is None:
        return lookahead_probs.new_zeros(()) if lookahead_probs is not None else \
               teacher_router_probs.new_zeros(())
    B, S, Kp1, E = lookahead_probs.shape
    K = min(int(lookahead_steps), Kp1 - 1)
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    if K <= 0 or S <= K:
        return lookahead_probs.new_zeros(())
    teacher = teacher_router_probs.detach()
    union_target = torch.zeros(B, S - K, E, dtype=lookahead_probs.dtype, device=lookahead_probs.device)
    for k in range(1, K + 1):
        target_ids = torch.topk(teacher[:, k:k + S - K, :], k=top_k, dim=-1).indices
        union_target = torch.maximum(
            union_target,
            F.one_hot(target_ids, num_classes=E).sum(dim=-2).to(union_target.dtype),
        )
    pred_union = lookahead_probs[:, :S - K, 1:K + 1, :].amax(dim=2)
    hit_mass = (pred_union * union_target).sum(dim=-1).clamp_min(1e-9)
    target_count = union_target.sum(dim=-1).clamp_min(1.0)
    return (-torch.log(hit_mass / target_count)).mean()


def total_loss(
    ce_loss: torch.Tensor,
    balance_loss: torch.Tensor,
    expert_probs_seq: torch.Tensor,
    lambda1: float,
    lambda2: float,
    *,
    lookahead_probs: torch.Tensor = None,
    teacher_probs: torch.Tensor = None,
    lookahead_steps: int = 0,
    lambda_lookahead: float = 0.0,
    lambda_lookahead_topk: float = 0.0,
    lambda_lookahead_union: float = 0.0,
    lambda_router_locality: float = 0.0,
    num_experts_per_tok: int = 1,
) -> torch.Tensor:
    """
    L_total = L_CE + λ1·L_balance + λ2·L_temporal + λ_lookahead·L_lookahead

    The lookahead term is only applied when `lookahead_probs`, `teacher_probs`
    and `lambda_lookahead > 0` are all provided — preserves the M1-era
    signature for legacy callers.
    """
    loss = ce_loss + lambda1 * balance_loss + lambda2 * temporal_locality_loss(expert_probs_seq)
    if lambda_router_locality > 0.0:
        loss = loss + lambda_router_locality * router_locality_loss(
            expert_probs_seq, num_experts_per_tok,
        )
    if (
        lookahead_probs is not None
        and teacher_probs is not None
        and lambda_lookahead > 0.0
        and lookahead_steps > 0
    ):
        loss = loss + lambda_lookahead * lookahead_supervision_loss(
            lookahead_probs, teacher_probs, lookahead_steps,
        )
    if (
        lookahead_probs is not None
        and teacher_probs is not None
        and lambda_lookahead_topk > 0.0
        and lookahead_steps > 0
    ):
        loss = loss + lambda_lookahead_topk * lookahead_topk_hit_loss(
            lookahead_probs, teacher_probs, lookahead_steps, num_experts_per_tok,
        )
    if (
        lookahead_probs is not None
        and teacher_probs is not None
        and lambda_lookahead_union > 0.0
        and lookahead_steps > 0
    ):
        loss = loss + lambda_lookahead_union * lookahead_union_loss(
            lookahead_probs, teacher_probs, lookahead_steps, num_experts_per_tok,
        )
    return loss
