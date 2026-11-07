"""Expert-cluster partition shared by training and lazy inference (no torch / mlx imports).

A cluster is a fixed set of expert ids that is stored and paged together
(one ``cluster_<c>.ctsr`` file in the MLX store).  The training-time
cluster-locality constraint and the inference-time cluster prefetch use the
same partition, so routing that training concentrates on one cluster is the
routing that one SSD read serves at inference.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

# Default strength of the training constraint (on by default).  Measured on
# the synthetic multi-topic A/B (tools/bench_cluster_locality.py, 300 steps,
# reports/residency_20261107/sweep/): λ = 0.03 / 0.1 / 0.3 / 1.0 cut lazy
# cluster reads per token from 1.48 to 0.51 / 0.15 / 0.023 / 0.026, with
# held-out CE within +0.3–0.6 % of λ = 0 at every value.  0.3 is the smallest
# value that reaches the plateau.
DEFAULT_LAMBDA = 0.3
# Window used when the data gives no record length (tokens).  The same sweep
# put the best window at the stream segment length (16: 0.044, 32: 0.023,
# 64: 0.11, whole sequence: 0.11 reads per token at a 32-token segment).
DEFAULT_WINDOW = 32
PAGING_MODES = ("auto", "cluster", "expert")


def resolve_cluster_count(config) -> int:
    """Number of clusters: ``cluster_locality_clusters`` or ``⌈E / (2k)⌉``, at least 1."""
    e = int(getattr(config, "num_experts", 0) or 0)
    if e <= 0:
        return 1
    n = int(getattr(config, "cluster_locality_clusters", 0) or 0)
    if n <= 0:
        k = max(1, int(getattr(config, "num_experts_per_tok", 1) or 1))
        n = -(-e // (2 * k))
    return max(1, min(n, e))


def contiguous_clusters(num_experts: int, num_clusters: int) -> List[List[int]]:
    """Split ``0..E-1`` into ``num_clusters`` contiguous, near-equal groups."""
    e, c = int(num_experts), max(1, min(int(num_clusters), int(num_experts)))
    base, extra = divmod(e, c)
    out, start = [], 0
    for i in range(c):
        size = base + (1 if i < extra else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


def expert_to_cluster(clusters: Sequence[Sequence[int]], num_experts: int) -> List[int]:
    """``assign[e]`` = cluster id of expert ``e``; every expert must appear exactly once."""
    assign = [-1] * int(num_experts)
    for cid, members in enumerate(clusters):
        for eid in members:
            eid = int(eid)
            if not 0 <= eid < int(num_experts) or assign[eid] != -1:
                raise ValueError(f"expert {eid} is out of range or in two clusters")
            assign[eid] = cid
    if -1 in assign:
        raise ValueError(f"experts {[i for i, c in enumerate(assign) if c == -1]} have no cluster")
    return assign


def training_clusters(config) -> List[List[int]]:
    """The partition the training constraint uses (and that inference should page by)."""
    e = int(getattr(config, "num_experts", 0) or 0)
    return contiguous_clusters(e, resolve_cluster_count(config))


def cluster_membership(clusters: Sequence[Sequence[int]], num_experts: int) -> List[List[float]]:
    """Row-per-expert one-hot ``[E, C]`` (nested lists; callers wrap it as a tensor)."""
    assign = expert_to_cluster(clusters, num_experts)
    c = len(clusters)
    return [[1.0 if assign[e] == j else 0.0 for j in range(c)] for e in range(int(num_experts))]


def trained_with_clusters(config) -> bool:
    """True when the weights were trained with the cluster-locality constraint (λ > 0)."""
    try:
        return float(getattr(config, "lambda_cluster_locality", 0.0) or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def inference_clusters(config, paging: str = "auto") -> Optional[List[List[int]]]:
    """Page layout for lazy inference, adapted to how the model was trained.

    ``paging``: ``"auto"`` pages by the training partition when the weights
    were trained with the constraint and per expert otherwise; ``"cluster"``
    forces the training partition; ``"expert"`` forces one page per expert.
    Returns the cluster list (singletons for per-expert paging).
    """
    if paging not in PAGING_MODES:
        raise ValueError(f"paging must be one of {PAGING_MODES}, not {paging!r}")
    e = int(getattr(config, "num_experts", 0) or 0)
    if e <= 0:
        return []
    if paging == "cluster" or (paging == "auto" and trained_with_clusters(config)):
        return training_clusters(config)
    return [[i] for i in range(e)]


def paging_tag(clusters: Sequence[Sequence[int]]) -> str:
    """Short cache-directory suffix that identifies a page layout."""
    sizes = sorted({len(c) for c in clusters})
    if sizes == [1]:
        return "e"
    return f"c{len(clusters)}"


def recommended_window(median_record_tokens: Optional[float], seq_len: int) -> int:
    """Cluster-locality window for a dataset: the power of two nearest the median
    record length (one record ≈ one stream segment), clamped to ``[16, seq_len]``.
    Falls back to ``DEFAULT_WINDOW`` when no record length is known."""
    seq_len = max(1, int(seq_len))
    if not median_record_tokens or median_record_tokens <= 0:
        return min(DEFAULT_WINDOW, seq_len)
    w = 1 << int(round(math.log2(float(median_record_tokens))))
    return int(max(min(16, seq_len), min(w, seq_len)))
