"""Segment-mode cluster objective: documents found from the tokens, no labels."""
from __future__ import annotations


import os
import sys

import numpy as np
import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.expert_clusters import cluster_locality_mode, segment_ids, segment_index  # noqa: E402


def test_segment_ids_split_at_separators_and_chunk_long_documents():
    ids = np.array([[5, 6, 0, 7, 8, 9, 0, 4],
                    [5, 6, 7, 8, 9, 10, 11, 12]])
    np.testing.assert_array_equal(segment_ids(ids), [[0, 0, 0, 1, 1, 1, 1, 2],
                                                     [0, 0, 0, 0, 0, 0, 0, 0]])
    # A separator closes its document; chunk=3 cuts documents into 3-token parts.
    np.testing.assert_array_equal(segment_ids(ids, 3), [[0, 0, 0, 1, 1, 1, 2, 3],
                                                        [0, 0, 0, 1, 1, 1, 2, 2]])
    inv, n = segment_index(segment_ids(ids, 3))
    assert n == 7 and inv.shape == (16,) and inv.dtype == np.int32
    assert len(set(inv[:8]) & set(inv[8:])) == 0                  # rows never share a segment


def test_mode_validation():
    class C:
        cluster_locality_mode = "segment"
    assert cluster_locality_mode(C) == "segment"
    assert cluster_locality_mode(object()) == "window"            # field absent: original form
    C.cluster_locality_mode = "label"
    with pytest.raises(ValueError):
        cluster_locality_mode(C)


def _docs_batch(doc_len=5, n_docs=6, C=3, S=None):
    """Packed rows of documents, each routed entirely into its own cluster ``d % C``."""
    S = S or doc_len * n_docs
    ids = np.full((1, S), 7, np.int64)
    ids[0, doc_len - 1::doc_len] = 0
    seg = segment_ids(ids)
    cluster = (seg % C)[0]
    return ids, cluster


def _router_from_clusters(cluster, E=6, C=3, V=2, sharp=12.0):
    """[1, S, V, E] probabilities concentrated on cluster ``cluster[s]`` (2 experts per cluster)."""
    S = len(cluster)
    logits = np.zeros((1, S, V, E), np.float32)
    for s, c in enumerate(cluster):
        logits[0, s, :, 2 * c:2 * c + 2] = sharp
    p = np.exp(logits - logits.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def _membership(E=6, C=3):
    m = np.zeros((E, C), np.float32)
    for e in range(E):
        m[e, e // (E // C)] = 1
    return m


def test_document_routing_scores_zero_and_window_misalignment_is_not_penalised():
    """Each document in one cluster, documents spread evenly: segment loss ≈ 0.
    The 8-token window form charges the same routing because its windows
    straddle documents (doc length 5)."""
    from chronos.model.temporal_loss import cluster_locality_loss, segment_cluster_loss
    ids, cluster = _docs_batch()
    r = torch.tensor(_router_from_clusters(cluster))
    mem = torch.tensor(_membership())
    seg = segment_cluster_loss(r, mem, segment_ids(ids)).item()
    win = cluster_locality_loss(r[:, :24], mem, 8).item()
    assert seg < 0.01
    assert win > 0.3
    uniform = torch.full_like(r, 1.0 / r.shape[-1])
    assert segment_cluster_loss(uniform, mem, segment_ids(ids)).item() == pytest.approx(1.0, abs=1e-5)


def test_padding_is_neither_a_document_nor_weight():
    from chronos.model.temporal_loss import segment_cluster_loss
    ids, cluster = _docs_batch(n_docs=4)
    r = _router_from_clusters(cluster)
    pad = 9
    ids_p = np.concatenate([ids, np.zeros((1, pad), np.int64)], axis=1)
    noise = np.random.default_rng(0).dirichlet(np.ones(6), size=(1, pad, 2)).astype(np.float32)
    r_p = np.concatenate([r, noise], axis=1)
    mask = torch.tensor(np.arange(ids_p.shape[1])[None, :] < ids.shape[1])
    mem = torch.tensor(_membership())
    a = segment_cluster_loss(torch.tensor(r), mem, segment_ids(ids)).item()
    b = segment_cluster_loss(torch.tensor(r_p), mem, segment_ids(ids_p), mask).item()
    assert b == pytest.approx(a, abs=1e-6)


def test_mlx_matches_torch_and_carries_gradient():
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.training.trainer import _segment_cluster_loss
    from chronos.model.temporal_loss import segment_cluster_loss
    rng = np.random.default_rng(4)
    ids = rng.integers(3, 40, (3, 24))
    ids[ids % 7 == 0] = 0                                          # random document boundaries
    logits = rng.normal(size=(3, 24, 4, 6)).astype(np.float32)
    mask = np.ones((3, 24), bool)
    mask[1, 17:] = False
    seg = segment_ids(ids, 8)
    mem = _membership()
    lt = torch.tensor(logits, requires_grad=True)
    t = segment_cluster_loss(torch.softmax(lt, -1), torch.tensor(mem), seg, torch.tensor(mask))
    t.backward()
    with mx.stream(mx.cpu):
        f = lambda x: _segment_cluster_loss(mx.softmax(x, axis=-1), mx.array(mem), seg, mx.array(mask))  # noqa: E731
        val, grad = mx.value_and_grad(f)(mx.array(logits))
        mx.eval(val, grad)
    assert float(val) == pytest.approx(t.item(), abs=1e-5)
    np.testing.assert_allclose(np.array(grad), lt.grad.numpy(), atol=1e-5)
    assert np.abs(np.array(grad)).sum() > 0
    assert np.abs(np.array(grad)[1, 17:]).max() == 0.0             # padding receives no gradient


def _tiny_cfg(mode, lam=0.5):
    from chronos.model.config import ChronosConfig
    return ChronosConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         vocab_size=48, num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
                         moe_intermediate_size=32, max_position_embeddings=64, recurrent_depth=2,
                         prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                         lambda_cluster_locality=lam, cluster_locality_window=8, cluster_locality_mode=mode)


def test_segment_mode_reaches_both_trainers_and_differs_from_window():
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.trainer import _ce_stage_loss
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.trainer.loss_mixin import chronos_loss_term
    ids = torch.randint(3, 48, (2, 16))
    ids[:, 5] = 0
    ids[1, 11] = 0
    vals = {}
    for mode in ("window", "segment"):
        torch.manual_seed(3)
        c = _tiny_cfg(mode)
        assert c.cluster_locality_mode == mode
        pt = ChronosForCausalLM(c)
        out = pt(ids)
        base = out.logits.sum() * 0.0
        vals[("torch", mode)] = chronos_loss_term(pt, base, None, c, input_ids=ids, labels=ids).item()
        no_ids = chronos_loss_term(pt, base, None, c).item()       # without ids: window form
        vals[("torch_noids", mode)] = no_ids
        with mx.stream(mx.cpu):
            m = ChronosMLXModel.from_chronos_pytorch(pt, c)
            m.train()
            x = mx.array(ids.numpy(), dtype=mx.int32)
            _ce_stage_loss(m, x, x)
            vals[("mlx", mode)] = float(m._last_loss_components["cluster_locality_loss"]) / 0.5
    assert vals[("torch", "segment")] != pytest.approx(vals[("torch", "window")], abs=1e-6)
    assert vals[("torch_noids", "segment")] == pytest.approx(vals[("torch", "window")], abs=1e-6)
    for mode in ("window", "segment"):
        assert 0.0 <= vals[("mlx", mode)] <= 1.0


def test_segment_ids_ignore_everything_but_the_separator():
    """Task/prompt tokens are ordinary tokens: changing them never moves a boundary."""
    rng = np.random.default_rng(2)
    ids = rng.integers(1, 64, (4, 40))
    ids[:, [9, 23]] = 0
    other = ids.copy()
    nz = other != 0
    other[nz] = rng.integers(1, 64, int(nz.sum()))
    np.testing.assert_array_equal(segment_ids(ids, 16), segment_ids(other, 16))


def test_schema_and_catalogs_know_the_field():
    import json
    from chronos.config_schema import MODEL_FIELDS
    from chronos.webui.schema import FIELDS
    assert "cluster_locality_mode" in MODEL_FIELDS
    field = FIELDS["cluster_locality_mode"]
    assert tuple(field.choices) == ("window", "segment") and field.default == "window"
    root = os.path.join(_repo_root, "chronos", "webui", "i18n")
    for lang in ("en", "zh-Hans", "zh-Hant", "ja"):
        cat = json.load(open(os.path.join(root, f"{lang}.json"), encoding="utf-8"))
        for k in ("field.cluster_locality_mode.label", "field.cluster_locality_mode.help", "choice.window",
                  "choice.segment", "choice.cluster_locality_mode.window", "choice.cluster_locality_mode.segment"):
            assert cat.get(k), (lang, k)

