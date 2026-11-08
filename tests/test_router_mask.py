"""Router regularizers and routing telemetry ignore right-padding."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip("mlx.core")


def _sft_like(lengths, prompt, width, pad=0):
    """Aligned (ids, labels) as ``SFTSource`` yields them: labels[t] = token t+1."""
    rng = np.random.default_rng(0)
    raw_ids = np.full((len(lengths), width + 1), pad, np.int32)
    raw_lab = np.full((len(lengths), width + 1), -100, np.int32)
    for r, n in enumerate(lengths):
        seq = rng.integers(3, 50, n)
        raw_ids[r, :n] = seq
        raw_lab[r, prompt:n] = seq[prompt:]
    return raw_ids[:, :-1], raw_lab[:, 1:]


def _router(B, S, V, E, seed=0):
    logits = np.random.default_rng(seed).normal(size=(B, S, V, E)).astype(np.float32)
    return mx.softmax(mx.array(logits), axis=-1)


def test_token_mask_marks_every_real_input_and_no_padding():
    from chronos.mlx.training.longrun import token_mask_for
    from chronos.mlx.training.trainer import real_token_mask
    lengths = [7, 12, 3]
    ids, labels = _sft_like(lengths, prompt=2, width=16)
    mask = token_mask_for(labels)
    assert mask is not None and mask.shape == ids.shape
    np.testing.assert_array_equal(mask.sum(1), lengths)            # last real token included
    np.testing.assert_array_equal(mask, ids != 0)                  # records hold no pad id
    np.testing.assert_array_equal(np.array(real_token_mask(mx.array(labels), aligned=True)), mask)
    # HF convention (labels[t] = token t, the loss shifts): same real positions.
    raw = np.where(np.arange(ids.shape[1])[None, :] >= 2, np.where(ids != 0, ids, -100), -100)
    np.testing.assert_array_equal(np.array(real_token_mask(mx.array(raw))), ids != 0)


def test_token_mask_is_none_for_packed_batches():
    from chronos.mlx.training.longrun import token_mask_for
    packed = np.random.default_rng(1).integers(3, 50, (4, 32))
    assert token_mask_for(packed) is None
    ids, labels = _sft_like([17, 17], prompt=3, width=16)          # rows fill the width
    assert token_mask_for(labels) is None


def _losses(router, lookahead, teacher, membership, mask, window):
    from chronos.mlx.training import trainer as T
    mean = router.mean(axis=2)
    out = {"balance": T._load_balance_loss(router, 2, mask),
           "cluster": T._cluster_locality_loss(router, membership, window, mask),
           "temporal": T._temporal_locality_loss(mean, mask),
           "locality": T._router_locality_loss(mean, 2, mask),
           "lookahead": T._lookahead_supervision_loss(lookahead, teacher, 2, mask),
           "lookahead_topk": T._lookahead_topk_hit_loss(lookahead, teacher, 2, 2, mask),
           "lookahead_union": T._lookahead_union_loss(lookahead, teacher, 2, 2, mask)}
    return {k: float(v.item()) for k, v in out.items()}


def _membership(E, C):
    m = np.zeros((E, C), np.float32)
    m[np.arange(E), np.arange(E) % C] = 1
    return mx.array(m)


@pytest.mark.parametrize("grouped", [False, True])
def test_masked_router_losses_match_unmasked_on_real_tokens(grouped):
    """All-true mask == no mask; a padded row == the same row truncated."""
    B, S, V, E, C, w, n = 1, 32, 3, 6, 3, 8, 16
    router = _router(B, S, V, E)
    member = _membership(E, C)
    G = 2 if grouped else 1
    la = mx.softmax(mx.array(np.random.default_rng(2).normal(size=(B, S, 3, G, E)).astype(np.float32)), -1)
    teacher = router[:, :, :G, :] if grouped else router.mean(axis=2)
    if not grouped:
        la = la[:, :, :, 0, :]
    ones = mx.ones((B, S), dtype=mx.bool_)
    full = _losses(router, la, teacher, member, None, w)
    assert full == pytest.approx(_losses(router, la, teacher, member, ones, w), rel=1e-5, abs=1e-6)

    mask = mx.array(np.arange(S)[None, :] < n)
    padded = _losses(router, la, teacher, member, mask, w)
    trunc = _losses(router[:, :n], la[:, :n], teacher[:, :n], member, None, w)
    assert padded == pytest.approx(trunc, rel=1e-5, abs=1e-6)


def test_padding_content_does_not_change_masked_losses():
    B, S, V, E, C = 3, 24, 2, 6, 3
    lengths = np.array([24, 10, 5])
    mask = mx.array(np.arange(S)[None, :] < lengths[:, None])
    a, b = _router(B, S, V, E, seed=3), _router(B, S, V, E, seed=4)
    keep = mask[:, :, None, None]
    mixed = mx.where(keep, a, b)                                   # same real tokens, other padding
    member = _membership(E, C)
    ta, tm = a.mean(axis=2), mixed.mean(axis=2)
    la = mx.softmax(mx.array(np.random.default_rng(5).normal(size=(B, S, 3, E)).astype(np.float32)), -1)
    lm = mx.where(mask[:, :, None, None], la, mx.softmax(mx.ones_like(la), -1))
    got_a = _losses(a, la, ta, member, mask, 8)
    got_m = _losses(mixed, lm, tm, member, mask, 8)
    assert got_a == pytest.approx(got_m, rel=1e-5, abs=1e-6)
    # Without the mask, padding moves the balance loss (the bug this guards).
    from chronos.mlx.training.trainer import _load_balance_loss
    assert float(_load_balance_loss(a, 2).item()) != pytest.approx(float(_load_balance_loss(mixed, 2).item()))


def test_routing_counts_sum_micro_batches_and_skip_padding():
    from chronos.mlx.training.longrun import RoutingCounts, routing_metrics
    r1, r2 = _router(2, 8, 3, 5, seed=6), _router(2, 8, 3, 5, seed=7)
    m1 = np.arange(8)[None, :] < np.array([[8], [3]])
    m2 = np.ones((2, 8), bool)
    acc = RoutingCounts()
    acc.add(r1, 2, m1)
    acc.add(r2, 2, m2)
    real = mx.array(np.concatenate([np.array(r1).reshape(-1, 3, 5)[m1.reshape(-1)], np.array(r2).reshape(-1, 3, 5)]))
    want = routing_metrics(real.reshape(1, -1, 3, 5), 2)
    got = acc.summary(2)
    assert got["route_tokens"] == 11 + 16 == want["route_tokens"]
    for k in ("route_entropy", "load_cv", "load_cv_max", "max_expert_load", "min_expert_load", "dead_experts"):
        assert got[k] == pytest.approx(want[k], rel=1e-5, abs=1e-6), k


def _tiny_sft_shards(root, lengths, prompt=4):
    rng = np.random.default_rng(0)
    for split in ("train", "valid"):
        d = os.path.join(root, "sft", split)
        os.makedirs(d)
        seqs = [rng.integers(3, 120, n).astype(np.uint16) for n in lengths]
        np.concatenate(seqs).tofile(os.path.join(d, "a.bin"))
        np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64).tofile(os.path.join(d, "a.idx"))
        np.full(len(lengths), prompt, np.int32).tofile(os.path.join(d, "a.plen"))


def test_sft_step_telemetry_covers_every_micro_batch_real_tokens_only(tmp_path):
    import json
    import time
    from chronos.mlx.training.longrun import LongRun, PhaseSpec, RunSpec, SFTSource, token_mask_for
    shards = str(tmp_path / "shards")
    _tiny_sft_shards(shards, [9, 11, 13, 20, 22, 26] * 8)
    model = dict(architecture_version=3, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, vocab_size=128, num_experts=4, num_experts_per_tok=2,
                 num_shared_experts=1, moe_intermediate_size=64, max_position_embeddings=128,
                 recurrent_depth=2, prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0)
    phase = PhaseSpec(name="sft", budget_fraction=1.0, peak_lr=1e-3, tokens_per_step=256, micro_tokens=128)
    spec = RunSpec(run_dir=str(tmp_path / "run"), shards_dir=shards, tokenizer_dir=shards, model_config=model,
                   phases=[phase], budget_hours=1.0, validate_every_s=1e9, checkpoint_every_s=1e9,
                   dtype="float32")
    with mx.stream(mx.cpu):
        run = LongRun(spec, log=lambda *_: None)
        run.state["phase_total"] = 4
        run.run_phase(0, time.monotonic() + 3600, max_steps=2)
    rows = [json.loads(x) for x in open(os.path.join(spec.run_dir, "metrics.jsonl"))]
    steps = [r for r in rows if r.get("phase") == "sft" and "step" in r and "route_tokens" in r]
    assert [r["step"] for r in steps] == [1, 2]
    replay = SFTSource(shards, spec.seed)
    for r in steps:                                    # accum = 256 // 128 = 2 micro-batches per step
        real = processed = 0
        for _ in range(2):
            ids, labels = replay.batch(128)
            mask = token_mask_for(labels)
            real += int(mask.sum()) if mask is not None else ids.size
            processed += ids.size
        assert r["route_tokens"] == real < processed
