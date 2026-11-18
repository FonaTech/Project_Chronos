"""0.3.4 decoding: repetition_penalty through speculative decoding and serving."""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from chronos.model.model_chronos import apply_repetition_penalty  # noqa: E402
from chronos.runtime.speculative import _apply_rep_penalty, speculative_generate  # noqa: E402
from test_speculative import model_for  # noqa: E402


def penalized_greedy_reference(model, ids, n, penalty):
    """Plain full-depth greedy decode with the CTRL penalty, token by token."""
    out = model(ids, use_cache=True)
    cache, seq = out.past_key_values, ids[0].tolist()
    for _ in range(n):
        row = apply_repetition_penalty(out.logits[:, -1], torch.tensor([seq]), penalty)
        nxt = int(row[0].argmax())
        seq.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return seq


def test_apply_rep_penalty_matches_kernel_helper():
    torch.manual_seed(0)
    logits = torch.randn(96)
    seen = torch.tensor([3, 7, 7, 40, 95])
    ours = _apply_rep_penalty(logits, seen, 1.3)
    ref = apply_repetition_penalty(logits[None], seen[None], 1.3)[0]
    assert torch.allclose(ours, ref)
    assert torch.equal(_apply_rep_penalty(logits, seen, 1.0), logits)


@pytest.mark.parametrize("gamma,draft_depth", [(1, 1), (4, 1), (3, 2)])
def test_greedy_speculative_with_penalty_equals_penalized_greedy(gamma, draft_depth):
    model = model_for("shared_first")
    ids = torch.tensor([[5, 9, 2, 17]])
    ref = penalized_greedy_reference(model, ids, 24, 1.3)
    out, _ = speculative_generate(model, ids, max_new_tokens=24, draft_depth=draft_depth,
                        gamma=gamma, repetition_penalty=1.3)
    assert out[0].tolist() == ref


def test_penalty_changes_greedy_output():
    model = model_for("shared_first")
    ids = torch.tensor([[5, 9, 2, 17]])
    plain, _ = speculative_generate(model, ids, max_new_tokens=24, draft_depth=1)
    pen, _ = speculative_generate(model, ids, max_new_tokens=24, draft_depth=1, repetition_penalty=2.0)
    assert plain[0].tolist() != pen[0].tolist()


def test_sampled_speculative_with_penalty_runs_and_respects_budget():
    model = model_for("shared_first")
    ids = torch.tensor([[5, 9, 2, 17]])
    gen = torch.Generator().manual_seed(3)
    out, stats = speculative_generate(model, ids, max_new_tokens=16, draft_depth=1, do_sample=True,
                        temperature=0.9, repetition_penalty=1.2, generator=gen)
    assert out.shape[1] == ids.shape[1] + 16
    assert stats.generated == 16


def test_nonpositive_penalty_rejected():
    model = model_for("shared_first")
    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        speculative_generate(model, torch.tensor([[1, 2]]), max_new_tokens=2, repetition_penalty=0.0)


def test_max_inference_depth_is_an_inference_override():
    from chronos.serving.runtime import INFERENCE_OVERRIDE_KEYS, apply_inference_overrides
    assert "max_inference_depth" in INFERENCE_OVERRIDE_KEYS
    model = model_for("shared_first")
    applied = apply_inference_overrides(model.config, {"max_inference_depth": 3})
    assert applied == {"max_inference_depth": 3}
    assert model.config.max_inference_depth == 3
