"""MLX recurrent-depth self-speculative decoding: exactness, rollback, sampling."""
from __future__ import annotations

import os
import sys

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.mlx.model import ChronosMLXModel  # noqa: E402
from chronos.mlx.speculative import (  # noqa: E402
    _probs, cache_length, rejection_step, speculative_generate, truncate_cache,
)
from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402


def mlx_model(mode="shared_first", window=64, **kw):
    torch.manual_seed(21)
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        vocab_size=96, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                        moe_intermediate_size=64, max_position_embeddings=128, recurrent_depth=4,
                        prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                        recurrent_attention_mode=mode, sliding_window_size=window, **kw)
    pt = ChronosForCausalLM(cfg).eval()
    return ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()


def prompt(n=7, seed=3):
    return [int(t) for t in np.random.default_rng(seed).integers(3, 96, size=n)]


def greedy_reference(model, ids, n):
    logits, _, cache = model(mx.array([ids], dtype=mx.int32))
    seq = list(ids)
    nxt = int(mx.argmax(logits[0, -1]).item())
    for _ in range(n):
        seq.append(nxt)
        logits, _, cache = model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
        nxt = int(mx.argmax(logits[0, -1]).item())
    return seq


@pytest.mark.parametrize("window", [64, 4])
@pytest.mark.parametrize("gamma,draft_depth", [(1, 1), (4, 1), (6, 2), (3, 4)])
def test_mlx_greedy_speculative_equals_full_depth_greedy(window, gamma, draft_depth):
    with mx.stream(mx.cpu):
        model = mlx_model(window=window)
        ids = prompt()
        ref = greedy_reference(model, ids, 24)
        out, stats = speculative_generate(model, ids, max_new_tokens=24, gamma=gamma,
                                          draft_depth=draft_depth, gamma_max=8)
    assert out == ref
    assert stats.generated == 24
    if draft_depth < 4:
        assert stats.verify_calls < 24          # verification is batched
        assert 0.0 <= stats.acceptance_rate <= 1.0
    else:
        assert stats.draft_calls == 0           # full draft depth = plain decoding


@pytest.mark.parametrize("overrides", [dict(kv_cache_dtype="int8"), dict(mla_decode="absorbed", mla_cache_rope=True)])
def test_mlx_speculative_exact_with_cache_options(overrides):
    with mx.stream(mx.cpu):
        model = mlx_model(window=4, **overrides)
        ids = prompt(9, seed=5)
        ref = greedy_reference(model, ids, 16)
        out, _ = speculative_generate(model, ids, max_new_tokens=16, gamma=3, draft_depth=1)
    assert out == ref


def test_mlx_per_iteration_falls_back_to_plain_decoding():
    with mx.stream(mx.cpu):
        model = mlx_model(mode="per_iteration")
        ids = prompt(5)
        ref = greedy_reference(model, ids, 10)
        out, stats = speculative_generate(model, ids, max_new_tokens=10, gamma=4, draft_depth=1)
    assert out == ref and stats.draft_calls == 0


@pytest.mark.parametrize("window", [64, 3])
def test_mlx_cache_rollback_equals_fresh_prefill(window):
    from chronos.mlx.speculative import set_cache_slack
    with mx.stream(mx.cpu):
        model = mlx_model(window=window)
        set_cache_slack(model, 6)
        ids = prompt(8)
        extra = [11, 12, 13, 14, 15]
        _, _, cache = model(mx.array([ids], dtype=mx.int32))
        _, _, longer = model(mx.array([extra], dtype=mx.int32), cache=cache)
        rolled = truncate_cache(longer, len(ids) + 2)       # keep two of the five extra tokens
        assert cache_length(rolled) == len(ids) + 2
        probe = mx.array([[40]], dtype=mx.int32)
        got, _, _ = model(probe, cache=rolled)
        _, _, fresh = model(mx.array([ids + extra[:2]], dtype=mx.int32))
        want, _, _ = model(probe, cache=fresh)
        mx.eval(got, want)
    np.testing.assert_allclose(np.array(got), np.array(want), atol=2e-5, rtol=2e-4)


def test_mlx_sampled_speculative_is_reproducible_and_stops_at_eos():
    with mx.stream(mx.cpu):
        model = mlx_model()
        ids = prompt(6)
        kw = dict(max_new_tokens=20, gamma=3, draft_depth=1, do_sample=True, temperature=0.9, top_k=20, top_p=0.95)
        a, sa = speculative_generate(model, ids, seed=7, **kw)
        b, _ = speculative_generate(model, ids, seed=7, **kw)
        assert a == b and sa.generated == 20 and all(0 <= t < 96 for t in a)
        eos = a[len(ids) + 4]
        c, sc = speculative_generate(model, ids, seed=7, eos_token_id=eos, **kw)
    assert c[-1] == eos and eos not in c[len(ids):-1] and sc.generated == len(c) - len(ids)


def test_mlx_probs_top_p_keeps_the_nucleus_in_vocab_order():
    with mx.stream(mx.cpu):
        logits = mx.array([0.0, 3.0, 1.0, 2.9, -1.0])
        p = np.array(_probs(logits, 1.0, 0, 0.6))
    assert set(np.nonzero(p)[0]) == {1, 3}             # the two largest, in their own slots
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-6)


def test_mlx_rejection_step_preserves_the_target_distribution():
    p_np, q_np = np.array([0.5, 0.2, 0.2, 0.1]), np.array([0.1, 0.6, 0.2, 0.1])
    trials, counts = 3000, np.zeros(4)
    with mx.stream(mx.cpu):
        p, q = mx.array(p_np), mx.array(q_np)
        key = mx.random.key(0)
        for _ in range(trials):
            key, k1, k2 = mx.random.split(key, 3)
            draft = int(mx.random.categorical(mx.log(q), key=k1).item())
            _, tok = rejection_step(p, q, draft, k2)
            counts[tok] += 1
    np.testing.assert_allclose(counts / trials, p_np, atol=0.035)   # ≈4σ at n=3000


def _engine(tmp_path, **kw):
    from chronos.mlx.inference import ChronosMLXInferenceEngine
    model = mlx_model(window=4, **kw)
    return model, ChronosMLXInferenceEngine(model, model.config, ssd_dir=str(tmp_path / "experts"))


def test_mlx_engine_speculative_matches_plain_greedy(tmp_path):
    with mx.stream(mx.cpu):
        model, engine = _engine(tmp_path)
        engine.store.storage_format = "full_dram"
        try:
            ids = mx.array([prompt(7)], dtype=mx.int32)
            plain = list(engine.generate(ids, max_new_tokens=20, temperature=0.0))
            spec = list(engine.generate(ids, max_new_tokens=20, temperature=0.0, speculative=True,
                                        draft_depth=1, gamma=4))
            stats = engine.last_stats
            eos = spec[5]
            stopped = list(engine.generate(ids, max_new_tokens=20, temperature=0.0, speculative=True,
                                           eos_token_id=eos))
            stopped_plain = list(engine.generate(ids, max_new_tokens=20, temperature=0.0, eos_token_id=eos))
        finally:
            engine.stop()
    assert spec == plain and len(spec) == 20
    assert stats["tokens_generated"] == 20 and stats["speculative"]["verify_calls"] < 20
    assert stopped == stopped_plain == spec[:spec.index(eos) + 1]


def test_mlx_engine_speculative_runs_with_lazy_experts(tmp_path):
    with mx.stream(mx.cpu):
        model, engine = _engine(tmp_path, recommended_resident_experts=2, recommended_ram_experts=2)
        try:
            engine.setup(warm_expert_ids=[0, 1])
            assert engine._all_resident_fast_path_enabled() is False
            toks = list(engine.generate(mx.array([prompt(6)], dtype=mx.int32), max_new_tokens=12,
                                        temperature=0.0, speculative=True, draft_depth=1, gamma=3))
            stats = engine.last_stats
        finally:
            engine.stop()
    assert len(toks) == 12 and all(0 <= t < 96 for t in toks)
    assert stats["all_resident_fast_path"] is False
    assert stats["speculative"]["draft_calls"] > 0 and stats["speculative"]["generated"] == 12


@pytest.mark.parametrize("min_depth", [2, 3])
def test_draft_depth_below_min_recurrent_depth_keeps_the_last_state(min_depth):
    """A draft that stops before ``min_recurrent_depth`` must not zero the ACT mixture."""
    torch.manual_seed(21)
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        vocab_size=96, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                        moe_intermediate_size=64, max_position_embeddings=128, recurrent_depth=4,
                        min_recurrent_depth=min_depth, prelude_layers=1, coda_layers=1, dropout=0.0,
                        fallback_mask_prob=0.0, recurrent_attention_mode="shared_first", sliding_window_size=64)
    pt = ChronosForCausalLM(cfg).eval()
    ids = np.random.default_rng(3).integers(3, 96, size=(1, 7))
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()
        for depth in range(1, min_depth):
            with torch.no_grad():
                ref = pt(torch.tensor(ids), recurrent_depth=depth).logits.numpy()
                full = pt(torch.tensor(ids)).logits.numpy()
            got, _, _ = model(mx.array(ids, dtype=mx.int32), recurrent_depth=depth)
            mx.eval(got)
            np.testing.assert_allclose(np.array(got), ref, atol=3e-5, rtol=3e-4)
            # a zeroed mixture feeds identical states to the coda: every position gets the same logits
            assert np.array(got).std(axis=1).max() > 1e-3
            assert np.abs(ref - full).max() > 1e-4          # the draft really ran fewer iterations
