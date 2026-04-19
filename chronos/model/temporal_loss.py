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


def total_loss(
    ce_loss: torch.Tensor,
    balance_loss: torch.Tensor,
    expert_probs_seq: torch.Tensor,
    lambda1: float,
    lambda2: float,
) -> torch.Tensor:
    """
    L_total = L_CE + λ1 * L_balance + λ2 * L_temporal

    Args:
        ce_loss:          standard cross-entropy loss
        balance_loss:     load-balancing auxiliary loss
        expert_probs_seq: [B, T, num_experts] routing probs over the sequence
        lambda1:          weight for balance loss
        lambda2:          weight for temporal locality penalty
    """
    return ce_loss + lambda1 * balance_loss + lambda2 * temporal_locality_loss(expert_probs_seq)
