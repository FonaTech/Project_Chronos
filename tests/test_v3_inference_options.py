"""v3 inference / training options: int8 KV cache, absorbed MLA decode,
shared routing scope, draft-exit KL and hard expert budgets (Torch + MLX)."""
from __future__ import annotations

import os
import sys

import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import chronos.model.hybrid_attention as ha
from chronos.eval.benchmark import kv_cache_memory_bytes
from chronos.model.config import ChronosConfig
from chronos.model.kv_quant import QuantizedKV, kv_read
from chronos.model.model_chronos import ChronosForCausalLM, cache_sequence_length, draft_exit_kl
from chronos.model.moe_chronos import ChronosMOEFeedForward
from chronos.runtime.speculative import set_cache_slack, speculative_generate, truncate_cache


def cfg_for(**overrides):
    values = dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=128, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=64, max_position_embeddings=64, recurrent_depth=3,
        prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
        kv_latent_dim=32, rope_dim=8,
    )
    values.update(overrides)
    return ChronosConfig(**values)


def model_for(seed=7, **overrides):
    torch.manual_seed(seed)
    return ChronosForCausalLM(cfg_for(**overrides)).eval()


def incremental_logits(model, ids, split):
    with torch.no_grad():
        out = model(ids[:, :split], use_cache=True)
        cache, steps = out.past_key_values, [out.logits]
        for t in range(split, ids.shape[1]):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits)
    return torch.cat(steps, dim=1), cache


IDS = torch.randint(3, 128, (2, 12), generator=torch.Generator().manual_seed(3))


# ---- config ---------------------------------------------------------------------

def test_new_fields_default_off_and_validate():
    cfg = cfg_for()
    assert (cfg.kv_cache_dtype, cfg.mla_decode, cfg.mla_cache_rope, cfg.router_scope) == \
        ("auto", "expanded", False, "per_iteration")
    assert cfg.draft_exit_kl_coef == 0.0 and cfg.max_resident_experts is None and cfg.ram_budget_mb is None
    assert cfg_for(kv_cache_dtype="bf16").kv_cache_dtype == "auto"
    for bad in (dict(kv_cache_dtype="int4"), dict(mla_decode="fast"), dict(router_scope="global")):
        with pytest.raises(ValueError):
            cfg_for(**bad)
    from chronos.config_schema import canonical_config_dict
    saved = canonical_config_dict(cfg_for(kv_cache_dtype="int8", router_scope="shared"))
    assert saved["kv_cache_dtype"] == "int8" and saved["router_scope"] == "shared"


# ---- int8 KV storage --------------------------------------------------------------

def test_quantized_kv_roundtrip_slicing_and_bytes():
    x = torch.randn(2, 9, 2, 64)
    q = QuantizedKV.quantize(x, 32)
    err = (q.dequantize(torch.float32) - x).abs()
    step = x.reshape(2, 9, 2, 2, 32).abs().amax(-1, keepdim=True) / 127.0
    assert bool((err.reshape(2, 9, 2, 2, 32) <= step * 0.5 + 1e-7).all())   # half a quantization step
    grown = q.append(torch.randn(2, 3, 2, 64))
    assert grown.shape == (2, 12, 2, 64)
    head = grown[:, :9]
    torch.testing.assert_close(head.dequantize(), q.dequantize(), atol=0, rtol=0)
    assert q.nbytes() == x.numel() * 1 + (x.numel() // 32) * 4
    assert q.nbytes() < x.numel() * 2                       # smaller than an fp16 cache
    with pytest.raises(TypeError):
        q[0]


def test_int8_cache_layers_store_int8_and_stay_close():
    ref = model_for()
    model = model_for(kv_cache_dtype="int8")
    full = ref(IDS).logits.detach()
    inc, cache = incremental_logits(model, IDS, 9)
    assert all(isinstance(t, QuantizedKV) for entry in cache for t in entry[:2] if not isinstance(t, int))
    scale = full.abs().max()
    assert float((inc - full).abs().max() / scale) < 5e-2
    _, fp_cache = incremental_logits(ref, IDS, 9)
    assert kv_cache_memory_bytes(cache, model) < 0.6 * kv_cache_memory_bytes(fp_cache, ref)
    assert cache_sequence_length(cache) == IDS.shape[1]


def test_int8_cache_rollback_matches_fresh_prefill():
    """Per-token quantization: a rolled-back int8 cache decodes exactly like a fresh one."""
    model = model_for(kv_cache_dtype="int8", sliding_window_size=4)
    set_cache_slack(model, 8)
    with torch.no_grad():
        long = model(IDS[:1], use_cache=True).past_key_values
        rolled = truncate_cache(long, 9)
        fresh = model(IDS[:1, :9], use_cache=True).past_key_values
        assert cache_sequence_length(rolled) == 9
        a = model(IDS[:1, 9:10], past_key_values=rolled, use_cache=True).logits
        b = model(IDS[:1, 9:10], past_key_values=fresh, use_cache=True).logits
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_int8_speculative_greedy_matches_int8_greedy():
    model = model_for(kv_cache_dtype="int8", recurrent_depth=4)
    ids = IDS[:1, :6]
    with torch.no_grad():
        out = model(ids, use_cache=True)
        cache, seq = out.past_key_values, ids[0].tolist()
        nxt = int(out.logits[0, -1].argmax())
        for _ in range(12):
            seq.append(nxt)
            out = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
            cache, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
        spec, _stats = speculative_generate(model, ids, max_new_tokens=12, gamma=3, draft_depth=1)
    assert spec[0].tolist() == seq


# ---- absorbed MLA decode -----------------------------------------------------------

@pytest.mark.parametrize("cache_rope", [False, True])
def test_absorbed_decode_matches_expanded(cache_rope):
    ref = model_for(mla_decode="expanded")
    model = model_for(mla_decode="absorbed", mla_cache_rope=cache_rope)
    full = ref(IDS).logits.detach()
    torch.testing.assert_close(model(IDS).logits.detach(), full, atol=1e-5, rtol=1e-4)
    inc, cache = incremental_logits(model, IDS, 9)
    torch.testing.assert_close(inc, full, atol=1e-5, rtol=1e-4)
    assert {len(e) for e in cache[::2]} == {2 if cache_rope else 1}   # MLA layers are even
    assert cache_sequence_length(cache) == IDS.shape[1]


def test_absorbed_explicit_matmul_branch_matches(monkeypatch):
    """The MPS fallback (SDPA rejects V width != Q/K width there), run on CPU."""
    ref = model_for(mla_decode="expanded")
    full = ref(IDS).logits.detach()
    monkeypatch.setattr(ha, "_sdpa_mixed_value_width", lambda device: False)
    model = model_for(mla_decode="absorbed")
    inc, _ = incremental_logits(model, IDS, 9)
    torch.testing.assert_close(inc, full, atol=1e-5, rtol=1e-4)


def test_auto_decode_absorbs_only_short_cached_chunks():
    assert next(m for m in model_for().modules() if isinstance(m, ha.MLAAttention)).mla_decode == "expanded"
    attn = next(m for m in model_for(mla_decode="auto").modules() if isinstance(m, ha.MLAAttention))
    assert attn._use_absorbed(1, 10) and attn._use_absorbed(8, 20)
    assert not attn._use_absorbed(9, 20) and not attn._use_absorbed(5, 5)
    attn.train()
    assert not attn._use_absorbed(1, 10)


def test_rope_cache_continues_a_cache_built_without_it():
    plain = model_for()
    roped = model_for(mla_cache_rope=True)
    full = plain(IDS).logits.detach()
    with torch.no_grad():
        cache = plain(IDS[:, :9], use_cache=True).past_key_values      # (c_kv,) entries
        out = roped(IDS[:, 9:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(out.logits, full[:, 9:], atol=1e-5, rtol=1e-4)
    assert {len(e) for e in out.past_key_values[::2]} == {2}


# ---- router_scope -------------------------------------------------------------------

def test_shared_router_scope_reuses_first_iteration_experts():
    model = model_for(router_scope="shared")
    k = model.config.num_experts_per_tok
    recurrent = model.model.layers[1].mlp
    torch.manual_seed(0)
    captured = []
    original = recurrent.forward
    def spy(x, available_expert_mask=None):
        out = original(x, available_expert_mask)
        captured.append(recurrent._shared_topk_idx.clone())
        return out
    recurrent.forward = spy
    with torch.no_grad():
        model(IDS)
    assert len(captured) == model.config.recurrent_depth
    assert all(torch.equal(c, captured[0]) for c in captured)
    # Iteration scores differ, yet the union of experts per token is <= top-k.
    probs = recurrent.iteration_router_probs
    assert len(probs) == model.config.recurrent_depth and not torch.equal(probs[0], probs[-1])
    assert captured[0].shape[-1] == k


def test_shared_router_scope_output_differs_and_gate_trains():
    base, shared = model_for(), model_for(router_scope="shared")
    assert torch.equal(base(IDS).logits, model_for()(IDS).logits)     # default unchanged
    shared.train()
    out = shared(IDS, labels=IDS)
    out.loss.backward()
    gate = shared.model.layers[1].mlp.gate.weight.grad
    assert gate is not None and float(gate.abs().sum()) > 0


def test_mlx_shared_router_scope_matches_torch():
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.model import ChronosMLXModel
    pt = model_for(router_scope="shared")
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, pt.config).eval()
        logits, _, _ = model(mx.array(IDS.numpy()))
        mx.eval(logits)
    with torch.no_grad():
        ref = pt(IDS).logits
    np.testing.assert_allclose(np.array(logits), ref.numpy(), atol=3e-5, rtol=3e-4)


# ---- draft_exit_kl ------------------------------------------------------------------

def test_draft_exit_kl_is_off_by_default_and_trains_the_shallow_exit():
    model = model_for().train()
    model(IDS, labels=IDS)
    assert model.last_draft_exit_kl is None
    from chronos.trainer.loss_mixin import chronos_loss_term
    model = model_for(draft_exit_kl_coef=0.5, draft_exit_depth=1).train()
    out = model(IDS, labels=IDS)
    kl = model.last_draft_exit_kl
    assert kl is not None and float(kl.detach()) > 0
    total = chronos_loss_term(model, out.loss, out.lookahead_probs, model.config, aux_loss=out.aux_loss)
    model.zero_grad()
    (0.5 * kl).backward(retain_graph=True)
    assert model.lm_head.weight.grad is not None
    assert float(total.detach()) > float(out.loss.detach())
    # Routing statistics describe the full-depth pass (run last).
    assert len(model.model.layers[1].mlp.iteration_router_probs) == model.config.recurrent_depth
    model.eval()
    model(IDS, labels=IDS)
    assert model.last_draft_exit_kl is None                       # inference never pays for it


def test_draft_exit_kl_teacher_is_detached_and_masked():
    full = torch.randn(2, 5, 11, requires_grad=True)
    draft = torch.randn(2, 5, 11, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, -100, 3], [-100, -100, -100, -100, -100]])
    value = draft_exit_kl(draft, full, labels)
    value.backward()
    assert full.grad is None and draft.grad is not None
    # Logit t is supervised when label t+1 is kept: row 0 -> positions 0 and 3.
    assert torch.count_nonzero(draft.grad.abs().sum(-1)) == 2
    assert float(draft_exit_kl(full.detach(), full.detach(), None)) == pytest.approx(0.0, abs=1e-6)


def test_mlx_draft_exit_kl_matches_torch():
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.training.trainer import draft_exit_kl as mlx_kl, _shifted_keep
    full, draft = torch.randn(2, 5, 11), torch.randn(2, 5, 11)
    labels = torch.tensor([[1, 2, -100, -100, 3], [4, 5, 6, -100, 7]])
    ref = draft_exit_kl(draft, full, labels)
    with mx.stream(mx.cpu):
        got = mlx_kl(mx.array(draft.numpy()), mx.array(full.numpy()), _shifted_keep(mx.array(labels.numpy())))
        mx.eval(got)
    np.testing.assert_allclose(float(got), float(ref), rtol=1e-5, atol=1e-6)


def test_mlx_ce_stage_loss_includes_draft_exit_term():
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.trainer import _ce_stage_loss
    pt = model_for(draft_exit_kl_coef=0.25)
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, pt.config).train()
        ids = mx.array(IDS.numpy())
        _ce_stage_loss(model, ids, ids)
        terms = model._last_loss_components
        assert "draft_exit_kl" in terms and float(terms["draft_exit_kl"]) > 0
        off = ChronosMLXModel.from_chronos_pytorch(model_for(), model_for().config).train()
        _ce_stage_loss(off, ids, ids)
        assert "draft_exit_kl" not in off._last_loss_components


# ---- hard expert budget + per-virtual-layer routing --------------------------------

def test_lru_protected_keys_survive_until_everything_is_protected():
    from chronos.io.expert_store import LRUCache
    lru = LRUCache(capacity=3)
    for key in (0, 1, 2):
        lru.put(key)
    lru.protected = frozenset({0, 1})
    assert lru.put(3) == 2 and lru.contains(0) and lru.contains(1)
    lru.protected = frozenset({0, 1, 3})
    assert lru.put(4) == 0 and len(lru) == 3                 # the hard cap still holds


def test_expert_store_hard_budget_caps_both_tiers(tmp_path):
    from chronos.io.expert_store import ExpertStore, expert_budget_pages
    cfg = cfg_for(storage_format="pt", num_experts=8, max_resident_experts=2, ram_budget_mb=0.25)
    model = ChronosForCausalLM(cfg).eval()
    store = ExpertStore(model, cfg, ssd_dir=str(tmp_path))
    page = store._expert_bytes * store.num_layers
    assert expert_budget_pages(cfg, page) == (2, max(1, int(0.25 * 2 ** 20 // page)))
    assert store.vram_capacity <= 2
    assert store.ram_capacity <= store.ram_budget_pages and store._safe_ram_capacity() <= store.ram_budget_pages
    stats = store.stats()
    assert stats["max_resident_pages"] == 2 and stats["rss_mb"] > 0 and stats["peak_rss_mb"] >= stats["rss_mb"]
    unbounded = ExpertStore(ChronosForCausalLM(cfg_for(storage_format="pt", num_experts=8)).eval(),
                            cfg_for(storage_format="pt", num_experts=8), ssd_dir=str(tmp_path / "u"))
    assert unbounded.max_resident_pages is None and unbounded.ram_budget_pages is None


def test_expert_store_never_exceeds_ram_budget_and_protects_predictions(tmp_path):
    from chronos.io.expert_store import ExpertStore
    cfg = cfg_for(storage_format="pt", num_experts=8, max_resident_experts=1)
    model = ChronosForCausalLM(cfg).eval()
    store = ExpertStore(model, cfg, ssd_dir=str(tmp_path))
    store.ram_budget_pages = 2                                # tight hard cap (pages)
    store.ram_capacity = 2
    store.offload_all_to_ssd()
    store.set_protected_experts([0])
    for eid in range(8):
        store.prefetch_to_ram([eid])
        assert len(store.ram_lru) <= 2
    assert store.ram_lru.contains(0)                          # predicted page kept
    store.set_protected_experts(())


def test_engine_routing_stats_cover_every_loop_iteration(tmp_path):
    from chronos.runtime.inference_engine import ChronosInferenceEngine
    cfg = cfg_for(storage_format="pt", num_experts=8, num_experts_per_tok=1)
    model = ChronosForCausalLM(cfg).eval()
    engine = ChronosInferenceEngine(model, cfg, ssd_dir=str(tmp_path))
    with torch.no_grad():
        model(IDS[:1, :4])
    moe = model.model.layers[1].mlp
    probs = engine._virtual_router_probs(moe, last_token=True)
    assert probs.shape[1] == cfg.recurrent_depth              # one row per virtual layer
    union = {int(p.argmax()) for p in probs[0]}
    assert union <= set(engine._get_current_expert_ids())
    stats = engine._route_cache_stats(None, include_all_tokens=False)
    recurrent = stats["per_layer"][1]
    assert recurrent["hits"] == cfg.recurrent_depth           # every dispatch counted
    assert set(recurrent["activated_experts"]) == union


def test_mlx_store_budget_and_stats(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.expert_store import MLXExpertStore
    from chronos.mlx.model import ChronosMLXModel
    cfg = cfg_for(num_experts=8, recommended_resident_experts=6, max_resident_experts=3, ram_budget_mb=0.5)
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(ChronosForCausalLM(cfg).eval(), cfg)
        store = MLXExpertStore(model, cfg, ssd_dir=str(tmp_path))
        pages = max(1, int(0.5 * 2 ** 20 // store.page_bytes))
        assert store._capacity == min(3, pages) and store._warm_capacity <= max(store._capacity, pages)
        stats = store.stats()
        assert stats["resident_experts"] == stats["hot_experts"] and "peak_rss_mb" in stats


# ---- MLX parity for the new cache formats ------------------------------------------

@pytest.mark.parametrize("overrides", [
    dict(mla_decode="absorbed"), dict(mla_cache_rope=True), dict(kv_cache_dtype="int8"),
    dict(kv_cache_dtype="int8", mla_cache_rope=True, mla_decode="absorbed", sliding_window_size=3),
])
def test_mlx_matches_torch_with_new_cache_options(overrides):
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.kv_quant import QuantizedKVMLX
    from chronos.mlx.model import ChronosMLXModel
    pt = model_for(**overrides)
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, pt.config).eval()
        cache_pt = cache_mx = None
        for chunk in (IDS[:1, :5], IDS[:1, 5:6], IDS[:1, 6:7], IDS[:1, 7:8]):
            with torch.no_grad():
                out = pt(chunk, past_key_values=cache_pt, use_cache=True)
            logits, _, cache_mx = model(mx.array(chunk.numpy()), cache=cache_mx)
            mx.eval(logits)
            np.testing.assert_allclose(np.array(logits), out.logits.numpy(), atol=3e-5, rtol=3e-4)
            cache_pt = out.past_key_values
        assert [len(e) for e in cache_mx] == [len(e) for e in cache_pt]
        if overrides.get("kv_cache_dtype") == "int8":
            assert isinstance(cache_mx[0][0], QuantizedKVMLX)
