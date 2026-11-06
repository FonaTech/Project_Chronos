"""Recurrent-depth self-speculative decoding: exactness and cache rollback."""
from __future__ import annotations

import os
import sys

import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM, cache_sequence_length
from chronos.runtime.speculative import speculative_generate, truncate_cache


def model_for(mode="shared_first", window=64, **kw):
    torch.manual_seed(21)
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        vocab_size=96, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                        moe_intermediate_size=64, max_position_embeddings=128, recurrent_depth=4,
                        prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                        recurrent_attention_mode=mode, sliding_window_size=window, **kw)
    return ChronosForCausalLM(cfg).eval()


def greedy_reference(model, ids, n):
    out = model(ids, use_cache=True)
    cache, seq = out.past_key_values, ids[0].tolist()
    nxt = int(out.logits[0, -1].argmax())
    for _ in range(n):
        seq.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
    return seq


@pytest.mark.parametrize("window", [64, 4])
@pytest.mark.parametrize("gamma,draft_depth", [(1, 1), (4, 1), (6, 2), (3, 4)])
def test_greedy_speculative_equals_full_depth_greedy(window, gamma, draft_depth):
    model = model_for(window=window)
    ids = torch.randint(3, 96, (1, 7))
    with torch.no_grad():
        ref = greedy_reference(model, ids, 24)
        out, stats = speculative_generate(model, ids, max_new_tokens=24, gamma=gamma,
                                          draft_depth=draft_depth, gamma_max=8)
    assert out[0].tolist() == ref
    assert stats.generated == 24
    if draft_depth < 4:
        assert stats.verify_calls < 24          # verification is batched
        assert 0.0 <= stats.acceptance_rate <= 1.0


def test_full_draft_depth_degrades_to_plain_decoding():
    model = model_for()
    ids = torch.randint(3, 96, (1, 5))
    with torch.no_grad():
        out, stats = speculative_generate(model, ids, max_new_tokens=10, gamma=4, draft_depth=4)
        ref = greedy_reference(model, ids, 10)
    assert stats.draft_calls == 0 and out[0].tolist() == ref


def test_eos_stops_generation_and_is_kept():
    model = model_for()
    ids = torch.randint(3, 96, (1, 6))
    with torch.no_grad():
        ref = greedy_reference(model, ids, 20)
    eos = ref[len(ids[0]) + 5]
    with torch.no_grad():
        out, _ = speculative_generate(model, ids, max_new_tokens=20, gamma=4, eos_token_id=eos)
    seq = out[0].tolist()
    first = seq.index(eos, len(ids[0]))
    assert seq == ref[:first + 1]


def test_cache_rollback_matches_fresh_prefill():
    model = model_for(window=4)
    from chronos.runtime.speculative import set_cache_slack
    set_cache_slack(model, 8)
    ids = torch.randint(3, 96, (1, 12))
    with torch.no_grad():
        long = model(ids, use_cache=True).past_key_values
        rolled = truncate_cache(long, 9)
        assert cache_sequence_length(rolled) == 9
        a = model(ids[:, 9:10], past_key_values=rolled, use_cache=True).logits
        b = model(ids[:, :10]).logits[:, -1:]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_rejection_step_output_is_distributed_as_target():
    from chronos.runtime.speculative import rejection_step
    gen = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.2, 0.2, 0.1, 0.0])
    q = torch.tensor([0.1, 0.1, 0.3, 0.2, 0.3])   # very different draft
    counts = torch.zeros(5)
    trials = 20000
    for _ in range(trials):
        draft = int(torch.multinomial(q, 1, generator=gen))
        _, tok = rejection_step(p, q, draft, gen)
        counts[tok] += 1
    tv = 0.5 * (counts / trials - p).abs().sum().item()
    assert tv < 0.015, tv
    assert counts[4] == 0                        # zero target mass is never emitted


def test_sampled_speculative_decoding_runs_and_respects_budget():
    model = model_for()
    ids = torch.randint(3, 96, (1, 6))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        out, stats = speculative_generate(model, ids, max_new_tokens=20, gamma=4, draft_depth=1,
                                          do_sample=True, temperature=0.8, top_k=20, generator=gen)
    assert out.shape[1] == ids.shape[1] + 20 and stats.generated == 20
    assert stats.verify_calls <= 20


def test_per_iteration_cache_rejects_depth_override():
    model = model_for(mode="per_iteration")
    with pytest.raises(ValueError):
        model(torch.randint(3, 96, (1, 4)), use_cache=True, recurrent_depth=1)
