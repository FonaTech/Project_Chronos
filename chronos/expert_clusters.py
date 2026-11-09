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


def median_record_tokens(shards_dir: str, max_files: int = 8) -> float | None:
    """Median SFT record length (tokens) from the ``.idx`` offsets under ``<shards>/sft/train``.

    Reads at most ``max_files`` index files (offsets only, no token data).
    Returns None when the shards have no SFT records (packed pretrain text
    has no record boundaries)."""
    import glob
    import os

    import numpy as np
    files = sorted(glob.glob(os.path.join(shards_dir, "sft", "train", "*.idx")))[:max_files]
    lengths = [np.diff(np.fromfile(f, dtype=np.int64)) for f in files]
    lengths = [x for x in lengths if x.size]
    if not lengths:
        return None
    return float(np.median(np.concatenate(lengths)))


def cluster_locality_config(model_config: dict, shards_dir: str | None, *, enabled: bool = True,
                            lambda_cluster: float | None = None, window: int | None = None,
                            clusters: int | None = None, seq_len: int | None = None) -> dict:
    """Cluster-locality fields for a fine-tune and where each value comes from.

    On by default at ``DEFAULT_LAMBDA`` (measured sweep on a synthetic
    multi-topic task, from-scratch training; its effect on a fine-tune of
    weights trained without the constraint has not been measured).  Off for
    dense models and when there are fewer than two clusters."""
    from chronos.model.config import ChronosConfig      # lazy: this module stays torch/mlx-free
    # The values training will use: longrun builds ChronosConfig(**model_config).
    cfg = ChronosConfig(**{k: v for k, v in model_config.items() if v is not None})
    e = int(getattr(cfg, "num_experts", 0) or 0)
    k = int(getattr(cfg, "num_experts_per_tok", 1) or 1)
    seq = int(seq_len or getattr(cfg, "max_position_embeddings", 512) or 512)
    usable = bool(getattr(cfg, "use_moe", True)) and e > 2 * k
    lam = 0.0 if not (enabled and usable) else float(DEFAULT_LAMBDA if lambda_cluster is None else lambda_cluster)
    if lam < 0:
        raise ValueError("lambda_cluster must be >= 0")
    median = median_record_tokens(shards_dir) if (shards_dir and window is None) else None
    win = int(window) if window is not None else recommended_window(median, seq)
    if win < 0:
        raise ValueError("cluster_window must be >= 0 (0 = whole sequence)")
    values = {"lambda_cluster_locality": lam, "cluster_locality_window": win,
              "cluster_locality_clusters": int(clusters or 0)}
    if not enabled:
        why = "off (disabled by the caller)"
    elif not usable:
        why = "off: dense model or fewer than two clusters"
    elif lambda_cluster is None:
        why = f"default {DEFAULT_LAMBDA} (measured sweep, synthetic task; not measured for fine-tuning)"
    else:
        why = "user value"
    if window is not None:
        wsrc = "user value"
    elif median is not None:
        wsrc = f"rule: power of two nearest the median SFT record ({median:.0f} tokens), clamped to [16, {seq}]"
    else:
        wsrc = "default (no SFT record lengths in the data)"
    return {"values": values, "source": {"lambda_cluster_locality": why, "cluster_locality_window": wsrc,
                                         "cluster_locality_clusters": "user value" if clusters else "ceil(E / 2k)"}}


# ---- cluster objective: unit of "one stream" --------------------------------
# "window": fixed ``cluster_locality_window``-token windows (the original form).
# "segment": the documents of the batch, found from the token stream itself
# (``SEGMENT_BOUNDARY_ID`` separators of packed pretraining, one record per SFT
# row); documents longer than ``cluster_locality_window`` are cut into parts.
CLUSTER_MODES = ("window", "segment")
SEGMENT_BOUNDARY_ID = 0   # ``<|endoftext|>`` in Chronos tokenizers: document separator / padding


def cluster_locality_mode(config) -> str:
    mode = str(getattr(config, "cluster_locality_mode", "window") or "window")
    if mode not in CLUSTER_MODES:
        raise ValueError(f"cluster_locality_mode must be one of {CLUSTER_MODES}, got {mode!r}")
    return mode


def segment_ids(input_ids, chunk: int = 0, boundary_id: int = SEGMENT_BOUNDARY_ID):
    """``[B, S]`` int64 document index of every position, from the tokens alone.

    A separator ``boundary_id`` closes the document it ends (so does each pad
    of a right-padded row; ``token_mask`` removes those from the loss).
    ``chunk`` > 0 cuts a document into ``chunk``-token parts.  Ids increase
    along each row; rows are told apart by ``segment_index``.  No label,
    prompt template or task token is consulted: what a document contains is
    left for the router to discover."""
    import numpy as np
    ids = np.asarray(input_ids)
    B, S = ids.shape
    is_bnd = ids == int(boundary_id)
    doc = np.zeros((B, S), np.int64)
    if S > 1:
        doc[:, 1:] = np.cumsum(is_bnd[:, :-1], axis=1)
    if int(chunk or 0) <= 0:
        return doc
    pos = np.arange(S)[None, :]
    first = np.ones((B, S), bool)
    first[:, 1:] = doc[:, 1:] != doc[:, :-1]
    start = np.maximum.accumulate(np.where(first, pos, 0), axis=1)
    key = doc * S + (pos - start) // int(chunk)
    new = np.zeros((B, S), np.int64)
    new[:, 1:] = key[:, 1:] != key[:, :-1]
    return np.cumsum(new, axis=1)


def segment_ids_mx(input_ids, boundary_id: int = SEGMENT_BOUNDARY_ID):
    """``segment_ids`` (chunk = 0) for an MLX array, traceable under ``mx.compile``.

    Same rule: a separator token closes the document it ends, so the id of position t is the
    number of separators strictly before t.  Returns an int32 ``[B, S]`` array."""
    import mlx.core as mx
    is_bnd = (input_ids == int(boundary_id)).astype(mx.int32)
    before = mx.cumsum(is_bnd, axis=1) - is_bnd
    return mx.stop_gradient(before)


def segment_index(segments):
    """``(inverse [B*S] int32, n)``: dense batch-wide segment number of each position."""
    import numpy as np
    seg = np.asarray(segments, np.int64)
    B, S = seg.shape
    flat = (seg + np.arange(B, dtype=np.int64)[:, None] * (S + 1)).reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    return inv.astype(np.int32).reshape(-1), int(uniq.size)
