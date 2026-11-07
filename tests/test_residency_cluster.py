"""Next-use residency planning and the cluster-locality training constraint."""
from __future__ import annotations

import math
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.expert_clusters import (  # noqa: E402
    cluster_membership, contiguous_clusters, expert_to_cluster, resolve_cluster_count,
)
from chronos.residency_planner import NextUsePlanner  # noqa: E402


# ── planner ───────────────────────────────────────────────────────

def _run(planner, kind, trace, verify=False):
    planner.begin_forward(kind)
    for layer, it, eid in trace:
        planner.access(layer, it, eid)
    planner.end_forward(verify=verify)


def test_planner_evicts_the_page_needed_latest():
    p = NextUsePlanner(8)
    trace = [(0, 0, 1), (1, 0, 2), (1, 1, 3), (2, 0, 4)]
    _run(p, "single", trace)
    p.begin_forward("single")
    p.access(0, 0, 1)
    # 2 is needed next, 4 last, 7 never: never-used first, then the latest one
    assert p.victim([2, 4, 7]) == 7
    assert p.victim([2, 3, 4]) == 4
    p.access(1, 0, 2)
    assert p.victim([1, 3]) == 1            # 1 is done for this forward, 3 still ahead


def test_planner_never_evicts_pending_experts_of_the_running_dispatch():
    p = NextUsePlanner(8)
    p.begin_forward("single")
    p.access(0, 0, 2, pending=[5, 6])
    assert p.victim([5, 6]) in (5, 6)       # forced choice still returns a candidate
    assert p.victim([5, 6, 7]) == 7


def test_planner_uses_the_draft_queue_for_verify():
    p = NextUsePlanner(8)
    _run(p, "multi", [(0, 0, 0)])                        # prefill
    _run(p, "draft", [(0, 0, 3), (1, 0, 4)])             # two draft tokens
    _run(p, "draft", [(0, 0, 5), (1, 0, 4)])
    p.begin_forward("multi")                             # verify
    assert p.stats["queue_verifies"] == 1
    assert set(p.upcoming()) >= {3, 4, 5}
    assert p.victim([3, 4, 5, 6]) == 6
    p.end_forward(verify=True)
    p.begin_forward("multi")                             # queue cleared after verify
    assert p.stats["queue_verifies"] == 1


def test_planner_prefetch_is_bounded_by_spare_slots():
    p = NextUsePlanner(8)
    _run(p, "single", [(0, 0, 1), (1, 0, 2), (2, 0, 3)])
    p.begin_forward("single")
    # resident {1, 6, 7}: 6 and 7 are never used → two spare slots at capacity 3
    assert p.prefetch_plan({1, 6, 7}, capacity=3) == [2, 3]
    assert p.prefetch_plan({1, 2, 3}, capacity=3) == []
    assert p.prefetch_plan({1}, capacity=2) == [2]


def test_planner_dedupes_loader_and_touch_reports():
    p = NextUsePlanner(4)
    p.begin_forward("single")
    p.access(0, 0, 1)
    p.access(0, 0, 1)
    p.end_forward()
    assert p.stats["accesses"] == 1


# ── cluster partition ─────────────────────────────────────────────

def test_contiguous_clusters_cover_every_expert_once():
    for e, c in ((36, 9), (16, 4), (10, 3), (5, 8)):
        clusters = contiguous_clusters(e, c)
        assign = expert_to_cluster(clusters, e)
        assert sorted(x for cl in clusters for x in cl) == list(range(e))
        assert len(set(assign)) == min(c, e)
        rows = cluster_membership(clusters, e)
        assert all(sum(r) == 1 for r in rows)
    with pytest.raises(ValueError):
        expert_to_cluster([[0, 1], [1, 2]], 3)


def test_default_cluster_count_is_e_over_2k():
    class Cfg:
        num_experts, num_experts_per_tok, cluster_locality_clusters = 36, 2, 0
    assert resolve_cluster_count(Cfg) == 9
    Cfg.cluster_locality_clusters = 4
    assert resolve_cluster_count(Cfg) == 4


# ── cluster-locality loss ─────────────────────────────────────────

torch = pytest.importorskip("torch")


def _membership(e=8, c=4):
    return torch.tensor(cluster_membership(contiguous_clusters(e, c), e), dtype=torch.float32)


def test_cluster_locality_loss_range_and_extremes():
    from chronos.model.temporal_loss import cluster_locality_loss
    m = _membership()
    B, S, V, E = 2, 16, 3, 8
    uniform = torch.full((B, S, V, E), 1.0 / E)
    assert cluster_locality_loss(uniform, m, window=4).item() == pytest.approx(1.0, abs=1e-5)
    # every window inside one cluster, the four windows of a row on four clusters
    local = torch.zeros(B, S, V, E)
    for j in range(4):
        local[:, j * 4:(j + 1) * 4, :, 2 * j:2 * j + 2] = 0.5
    assert cluster_locality_loss(local, m, window=4).item() == pytest.approx(0.0, abs=1e-4)
    # collapse onto one cluster everywhere: windows are pure but carry no information
    collapsed = torch.zeros(B, S, V, E)
    collapsed[..., 0:2] = 0.5
    assert cluster_locality_loss(collapsed, m, window=4).item() == pytest.approx(1.0, abs=1e-4)
    rand = torch.softmax(torch.randn(B, S, V, E), dim=-1)
    val = cluster_locality_loss(rand, m, window=4).item()
    assert 0.0 <= val <= 1.0


def test_cluster_locality_loss_counts_every_recurrent_iteration():
    """A window that switches cluster at a later loop iteration is not local."""
    from chronos.model.temporal_loss import cluster_locality_loss
    m = _membership()
    probs = torch.zeros(1, 16, 2, 8)
    for j in range(4):                              # window j: cluster j in both iterations
        probs[:, j * 4:(j + 1) * 4, :, 2 * j:2 * j + 2] = 0.5
    same = cluster_locality_loss(probs, m, window=4).item()
    probs[:, :4, 1] = 0.0
    probs[:, :4, 1, 2:4] = 0.5                      # iteration 1 of window 0 moves to cluster 1
    split = cluster_locality_loss(probs, m, window=4).item()
    assert same == pytest.approx(0.0, abs=1e-4) and split > same + 0.1


def test_cluster_locality_gradient_moves_mass_into_the_window_cluster():
    from chronos.model.temporal_loss import cluster_locality_loss
    torch.manual_seed(0)
    m = _membership()
    logits = torch.randn(2, 16, 2, 8, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    first = None
    for _ in range(200):
        loss = cluster_locality_loss(torch.softmax(logits, -1), m, window=4)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_torch_and_mlx_cluster_locality_losses_match():
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.training.trainer import _cluster_locality_loss
    from chronos.model.temporal_loss import cluster_locality_loss
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(2, 24, 5, 12)).astype(np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    member = cluster_membership(contiguous_clusters(12, 3), 12)
    for window in (0, 8, 5):
        ref = cluster_locality_loss(torch.tensor(probs), torch.tensor(member), window=window).item()
        with mx.stream(mx.cpu):
            got = float(_cluster_locality_loss(mx.array(probs), mx.array(member, dtype=mx.float32), window))
        assert got == pytest.approx(ref, abs=1e-5)


def test_cluster_term_reaches_both_training_losses():
    """``lambda_cluster_locality`` adds the term in the Torch and MLX trainers (0 = absent)."""
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.trainer import _ce_stage_loss
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.trainer.loss_mixin import chronos_loss_term

    def cfg(lam):
        return ChronosConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             vocab_size=48, num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
                             moe_intermediate_size=32, max_position_embeddings=64, recurrent_depth=2,
                             prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                             lambda_cluster_locality=lam, cluster_locality_window=8)
    ids = torch.randint(3, 48, (2, 16))
    totals = {}
    for lam in (0.0, 0.5):
        torch.manual_seed(3)
        c = cfg(lam)
        pt = ChronosForCausalLM(c)
        out = pt(ids)
        base = out.logits.sum() * 0.0
        totals[("torch", lam)] = chronos_loss_term(pt, base, None, c).item()
        with mx.stream(mx.cpu):
            m = ChronosMLXModel.from_chronos_pytorch(pt, c)
            m.train()
            _ce_stage_loss(m, mx.array(ids.numpy(), dtype=mx.int32), mx.array(ids.numpy(), dtype=mx.int32))
            totals[("mlx", lam)] = "cluster_locality_loss" in m._last_loss_components
    assert totals[("torch", 0.5)] > totals[("torch", 0.0)]
    assert totals[("mlx", 0.5)] and not totals[("mlx", 0.0)]


# ── lazy engine ───────────────────────────────────────────────────

@pytest.mark.parametrize("speculative", [False, True])
def test_planned_residency_is_exact_and_uses_the_planner(tmp_path, speculative):
    """Greedy output is unchanged; eviction runs through the planner and is removed afterwards.

    Load counts are not asserted here: a 4-expert toy with a 2-page budget is
    too small for next-use prediction to beat LRU reliably; the effect is
    measured on the 50M checkpoint (``tools/bench_residency.py``).
    """
    mx = pytest.importorskip("mlx.core")
    from test_mlx_speculative import _engine, prompt
    with mx.stream(mx.cpu):
        _model, engine = _engine(tmp_path, recommended_resident_experts=2, recommended_ram_experts=2)
        try:
            engine.setup(warm_expert_ids=[0, 1])
            ids = mx.array([prompt(7)], dtype=mx.int32)
            plain = list(engine.generate(ids, max_new_tokens=24, temperature=0.0))
            kw = dict(speculative=True, draft_depth=1, gamma=4) if speculative else {}
            loads, policy = {}, {}
            for res in ("window", "planned"):
                engine.store.replace_live_experts_with_placeholders()
                with engine.store._lock:
                    engine.store._warm.clear()
                    engine.store._warm_lru.clear()
                engine.store.warm_up([0, 1])
                out = list(engine.generate(ids, max_new_tokens=24, temperature=0.0, residency=res, **kw))
                assert out == plain
                st = engine.last_stats
                loads[res] = st["sync_ssd_loads"]
                policy[res] = st["eviction_policy"]
                if res == "planned":
                    planner = st["residency"]["planner"]
                    assert planner["accesses"] > 0 and planner["forwards"] > 0
                    if speculative:
                        assert planner["queue_verifies"] > 0
            assert engine.store.stats()["eviction_policy"] == "lru"      # ranker removed afterwards
        finally:
            engine.stop()
    assert policy == {"window": "lru", "planned": "planned"}
    assert loads["planned"] > 0 and loads["window"] > 0


def test_unknown_residency_mode_is_rejected(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from test_mlx_speculative import _engine, prompt
    with mx.stream(mx.cpu):
        _model, engine = _engine(tmp_path)
        try:
            with pytest.raises(ValueError):
                list(engine.generate(mx.array([prompt(5)], dtype=mx.int32), max_new_tokens=2, residency="queue"))
        finally:
            engine.stop()
