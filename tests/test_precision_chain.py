"""Single-compute-dtype precision chain (Torch CPU/CUDA/MPS and MLX)."""
from __future__ import annotations

import os
import sys

import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.precision import normalize_dtype_name, resolve_torch


def tiny(**overrides):
    values = dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=128, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=64, max_position_embeddings=64, recurrent_depth=2,
        prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
        kv_latent_dim=32, rope_dim=8, sliding_window_size=6,
    )
    values.update(overrides)
    return ChronosConfig(**values)


def _rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ---- policy -------------------------------------------------------------------

def test_policy_resolution_per_backend():
    assert normalize_dtype_name("bf16") == "bfloat16"
    assert normalize_dtype_name("torch.float16") == "float16"
    with pytest.raises(ValueError):
        normalize_dtype_name("int8")
    assert resolve_torch("auto", "cpu").compute == "float32"
    assert resolve_torch("auto", "mps").compute == "float32"     # measured faster than BF16 on MPS
    assert resolve_torch("bf16", "mps").compute == "bfloat16"
    assert resolve_torch("bf16", "cpu").compute == "bfloat16"
    p = resolve_torch("auto", "cpu")
    assert p.kv_cache == p.expert_storage == p.compute and p.master == "float32"
    assert "lm_head_loss" in p.as_dict()["boundary_casts"]


def test_mlx_policy_cpu_is_float32():
    pytest.importorskip("mlx.core")
    from chronos.precision import resolve_mlx
    assert resolve_mlx("bf16", device="cpu").compute == "float32"
    assert resolve_mlx("auto", device="gpu").compute == "bfloat16"
    assert resolve_mlx("float16", device="gpu").compute == "float16"


# ---- Torch kernels ------------------------------------------------------------

def _explicit_attention(q, k, v, allowed):
    """Float64 oracle: repeated GQA K/V, masked softmax."""
    rep = q.shape[1] // k.shape[1]
    k, v = k.double().repeat_interleave(rep, 1), v.double().repeat_interleave(rep, 1)
    scores = (q.double() @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    return torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1) @ v


@pytest.mark.parametrize("S,T,window,pad", [(6, 6, None, False), (6, 6, None, True), (1, 9, None, False),
                                            (3, 9, None, True), (6, 6, 3, False), (2, 8, 3, True)])
def test_torch_sdpa_masks_match_explicit_oracle(S, T, window, pad):
    from chronos.model.precision_ops import attention, build_attention_mask, causal_allowed
    torch.manual_seed(0)
    B, nH, nKV, d = 2, 4, 2, 8
    q, k, v = torch.randn(B, nH, S, d), torch.randn(B, nKV, T, d), torch.randn(B, nKV, T, d)
    am = None
    if pad:
        am = torch.ones(B, T, dtype=torch.long)
        am[0, :2] = 0                                     # left padding on row 0
    mask, is_causal = build_attention_mask(S, T, q.device, past=T - S, window=window, attention_mask=am)
    out = attention(q, k, v, mask=mask, is_causal=is_causal, dropout_p=0.0)
    allowed = causal_allowed(S, T, q.device, past=T - S, window=window)[None, None].expand(B, 1, S, T)
    if am is not None:
        diag = torch.arange(T)[None, :] == torch.arange(S)[:, None] + (T - S)
        allowed = (allowed & am.bool()[:, None, None, :]) | diag
    ref = _explicit_attention(q, k, v, allowed)
    assert torch.isfinite(out).all()
    assert _rel(out, ref) < 1e-6


def test_torch_rope_per_head_is_a_rotation_and_legacy_is_kept():
    from chronos.model.precision_ops import apply_rope, rope_tables
    torch.manual_seed(0)
    cos, sin = rope_tables(8, 32, 1e6, None, "cpu")
    x = torch.randn(1, 32, 2, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    # Relative-position property: <R_m q, R_n k> depends only on m - n.
    q, k = torch.randn(8), torch.randn(8)
    qs = apply_rope(q.expand(1, 32, 1, 8).contiguous(), cos, sin)[0, :, 0]
    ks = apply_rope(k.expand(1, 32, 1, 8).contiguous(), cos, sin)[0, :, 0]
    assert abs(float(qs[7] @ ks[3]) - float(qs[20] @ ks[16])) < 1e-4
    legacy = tiny()
    assert legacy.rope_layout == "legacy_sliced"
    mla = ChronosForCausalLM(legacy).model.layers[0].self_attn
    assert mla._rope_table_dim == legacy.head_dim
    per_head = ChronosForCausalLM(tiny(rope_layout="per_head")).model.layers[0].self_attn
    assert per_head._rope_table_dim == per_head.rope_dim
    with pytest.raises(ValueError):
        tiny(rope_layout="bogus")


@pytest.mark.parametrize("layout", ["legacy_sliced", "per_head"])
def test_torch_bf16_chain_has_no_fp32_activations_and_tables_stay_fp32(layout):
    torch.manual_seed(0)
    model = ChronosForCausalLM(tiny(rope_layout=layout)).eval().to(torch.bfloat16)
    ids = torch.randint(0, 128, (2, 7))
    seen = set()
    hooks = [m.register_forward_hook(lambda _m, _i, o: seen.add(o.dtype) if torch.is_tensor(o) else None)
             for name, m in model.named_modules() if name.endswith(("_proj", "_norm", "layernorm"))]
    with torch.no_grad():
        out = model(ids, use_cache=True)
    for h in hooks:
        h.remove()
    assert seen == {torch.bfloat16}
    assert out.logits.dtype == torch.bfloat16              # inference: compute dtype
    c_kv = out.past_key_values[0][0]
    assert c_kv.dtype == torch.bfloat16                     # KV cache in compute dtype
    attn = model.model.layers[0].self_attn
    assert attn._rope_cos.dtype == torch.float32            # never rounded by .to(bf16)
    assert model.model.freqs_cos.dtype == torch.float32
    with torch.no_grad():
        lossy = model(ids, labels=ids)
    assert lossy.logits.dtype == torch.float32 and torch.isfinite(lossy.loss)


def test_torch_bf16_prefill_decode_matches_full_forward():
    torch.manual_seed(1)
    model = ChronosForCausalLM(tiny(rope_layout="per_head")).eval().to(torch.bfloat16)
    ids = torch.randint(0, 128, (1, 9))
    with torch.no_grad():
        full = model(ids).logits.float()
        out = model(ids[:, :6], use_cache=True)
        steps = [out.logits.float()]
        cache = out.past_key_values
        for t in range(6, 9):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            steps.append(out.logits.float())
            cache = out.past_key_values
    assert _rel(torch.cat(steps, 1), full) < 2e-2


def test_torch_router_softmax_fp32_and_moe_output_in_compute_dtype():
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    model = ChronosForCausalLM(tiny()).eval().to(torch.bfloat16)
    moe = next(m for m in model.modules() if isinstance(m, ChronosMOEFeedForward))
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        y = moe(x)
    assert y.dtype == torch.bfloat16
    assert moe.last_router_probs.dtype == torch.float32


def test_lm_head_logits_born_fp32_from_bf16_operands():
    from chronos.model.precision_ops import lm_head_logits
    torch.manual_seed(0)
    h, w = torch.randn(3, 5, 64).bfloat16(), torch.randn(128, 64).bfloat16()
    ref = h.double() @ w.double().T
    assert lm_head_logits(h, w).dtype == torch.bfloat16
    out = lm_head_logits(h, w, torch.float32)
    assert out.dtype == torch.float32 and _rel(out, ref) < 1e-6


# ---- placement / expert pages -------------------------------------------------

def test_place_torch_model_moves_placeholders_and_expert_pages_to_trunk_dtype(tmp_path):
    from chronos.io.expert_store import ExpertStore
    from chronos.model.moe_chronos import ChronosMOEFeedForward, LazyExpertPlaceholder
    from chronos.precision import place_torch_model, torch_model_dtype
    cfg = tiny()
    model = ChronosForCausalLM(cfg).eval()
    moe = next(m for m in model.modules() if isinstance(m, ChronosMOEFeedForward))
    moe.experts[0] = LazyExpertPlaceholder(64, 64, torch.float16)
    model, policy = place_torch_model(model, "cpu", "bf16")
    assert policy.compute == "bfloat16" and model._chronos_precision is policy
    assert torch_model_dtype(model) == torch.bfloat16
    assert moe.experts[0].dtype == torch.bfloat16
    store = ExpertStore(model, cfg, ssd_dir=str(tmp_path))
    assert store._expert_dtype_cached == torch.bfloat16
    fp32, _ = place_torch_model(ChronosForCausalLM(cfg), "cpu", "auto")
    assert torch_model_dtype(fp32) == torch.float32


# ---- training: FP32 masters, no autocast ----------------------------------------

def _micro_batches():
    g = torch.Generator().manual_seed(5)
    return [torch.randint(0, 128, (2, 8), generator=g) for _ in range(2)]


def test_fp32_master_weights_accumulate_exactly_and_sync_back():
    import copy
    from chronos.trainer.precision_training import attach_precision, optimizer_step
    torch.manual_seed(3)
    base = ChronosForCausalLM(tiny()).train()
    twin = copy.deepcopy(base).to(torch.bfloat16)
    model = base
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    policy, ctx, scaler = attach_precision(model, optimizer, "cpu", "bf16")
    assert policy.compute == "bfloat16" and not scaler.is_enabled()
    masters = optimizer.chronos_masters
    assert masters.pairs and all(m.dtype == torch.float32 for _, m in masters.pairs)
    assert all(p.dtype == torch.float32 for g in optimizer.param_groups for p in g["params"])

    expected = {}
    for ids in _micro_batches():
        torch.manual_seed(7)
        with ctx:
            model(ids, labels=ids).loss.backward()
        torch.manual_seed(7)
        twin.zero_grad(set_to_none=True)
        twin(ids, labels=ids).loss.backward()
        for name, p in twin.named_parameters():
            if p.grad is not None:
                expected[name] = expected.get(name, 0) + p.grad.float()
    names = dict(model.named_parameters())
    checked = 0
    for name, ref in expected.items():
        p = names[name]
        assert p.grad is None                               # folded and freed
        torch.testing.assert_close(masters.master_of[id(p)].grad, ref, atol=0, rtol=0)
        checked += 1
    assert checked > 10

    before = {id(p): m.detach().clone() for p, m in masters.pairs}
    optimizer_step(scaler, optimizer, model, grad_clip=1.0)
    moved = 0
    for p, m in masters.pairs:
        assert torch.equal(p.detach(), m.detach().to(p.dtype))   # one rounding per update
        moved += int(not torch.equal(m, before[id(p)]))
        assert m.grad is None and p.grad is None
    assert moved > 10
    state = next(iter(optimizer.state.values()))
    assert state["exp_avg"].dtype == torch.float32


def test_fp16_compute_enables_loss_scaling_only_for_fp16():
    from chronos.precision import resolve_torch
    from chronos.trainer.precision_training import grad_scaler_for
    assert grad_scaler_for(resolve_torch("fp16", "cpu"), "cpu").is_enabled()
    assert not grad_scaler_for(resolve_torch("bf16", "cpu"), "cpu").is_enabled()
    assert not grad_scaler_for(resolve_torch("auto", "cpu"), "cpu").is_enabled()


def test_autocast_is_disabled_for_native_dtype_models():
    import contextlib
    from chronos.trainer.device_utils import autocast_context
    model = ChronosForCausalLM(tiny())
    model._chronos_precision = resolve_torch("bf16", "cpu")
    assert isinstance(autocast_context("cpu", "bf16", model=model), contextlib.nullcontext)


# ---- MLX <-> torch bridges and checkpoints ---------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_mlx_torch_bridge_is_bit_exact(dtype):
    pytest.importorskip("mlx.core")
    from chronos.precision import mx_to_torch, torch_to_mx
    torch.manual_seed(0)
    t = torch.cat([torch.randn(257) * 30, torch.tensor([0.0, -0.0, 1e-30, 65504.0, float("inf")])]).to(dtype)
    back = mx_to_torch(torch_to_mx(t))
    assert back.dtype == dtype
    bits = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[dtype]
    assert torch.equal(back.view(bits), t.view(bits))


def test_save_mlx_checkpoint_keeps_bf16_bits(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.trainer import _save_mlx_checkpoint
    torch.manual_seed(4)
    cfg = tiny(rope_layout="per_head")
    pt = ChronosForCausalLM(cfg).eval().to(torch.bfloat16)
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg, dtype="bfloat16")
        path = str(tmp_path / "m.pth")
        _save_mlx_checkpoint(model, cfg, path, stage="pretrain")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    ref = pt.state_dict()
    assert set(saved) == set(ref)
    for k, v in saved.items():
        assert v.dtype == ref[k].dtype, k
        if v.is_floating_point():
            assert torch.equal(v.view(torch.int16), ref[k].view(torch.int16)), k


def test_mlx_matches_torch_with_per_head_rope():
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.model import ChronosMLXModel
    torch.manual_seed(11)
    cfg = tiny(rope_layout="per_head", sliding_window_size=3)
    pt = ChronosForCausalLM(cfg).eval()
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()
        cache_pt = cache_mx = None
        for chunk in (torch.tensor([[1, 4, 2, 5, 7]]), torch.tensor([[6]]), torch.tensor([[9]])):
            with torch.no_grad():
                out = pt(chunk, past_key_values=cache_pt, use_cache=True)
            logits, _, cache_mx = model(mx.array(chunk.numpy()), cache=cache_mx)
            mx.eval(logits)
            np.testing.assert_allclose(np.array(logits), out.logits.numpy(), atol=2e-5, rtol=2e-4)
            cache_pt = out.past_key_values


# ---- MLX expert pages / dispatch -------------------------------------------------

@pytest.mark.parametrize("clusters", [[[0, 1], [2, 3]], None])
def test_mlx_expert_store_keeps_bf16_pages_bit_exact(tmp_path, clusters):
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.expert_store import MLXExpertStore
    from chronos.mlx.model import ChronosMLXModel
    torch.manual_seed(2)
    cfg = tiny()
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(ChronosForCausalLM(cfg).eval(), cfg, dtype="bfloat16")
        ref = {(li, e): model.layers[li].mlp.experts[e].gate_proj.weight for li in range(cfg.num_hidden_layers)
               for e in range(cfg.num_experts)}
        store = MLXExpertStore(model, cfg, ssd_dir=str(tmp_path / "c"))
        assert store.page_dtype == mx.bfloat16
        store.offload_all_to_ssd(clusters=clusters)
        loaded = store._load_from_ssd(1)
        assert loaded
        for li, params in loaded.items():
            w = params["gate_proj.weight"]
            assert w.dtype == mx.bfloat16
            assert bool(mx.array_equal(w, ref[(li, 1)]))
        assert store.stats().get("page_dtype_conversions", 0) == 0


def test_mlx_expert_store_converts_legacy_fp16_cache_once(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.expert_store import MLXExpertStore
    from chronos.mlx.model import ChronosMLXModel
    from ui.tabs.inference_tab import _populate_mlx_cache_from_state_dict
    cfg = tiny()
    pt = ChronosForCausalLM(cfg).eval()
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg, dtype="bfloat16")
        store = MLXExpertStore(model, cfg, ssd_dir=str(tmp_path / "lazy"))
        _populate_mlx_cache_from_state_dict(store, pt.state_dict())      # written in page dtype
        assert store.attach_cluster_manifest(str(tmp_path / "lazy"))
        w = store._load_from_ssd(0)[0]["up_proj.weight"]
        assert w.dtype == mx.bfloat16 and store.stats().get("page_dtype_conversions", 0) == 0
        store.page_dtype = mx.float32                                     # trunk/page mismatch
        store._warm.clear(); store._warm_lru.clear(); store._loaded_clusters.clear()
        assert store._load_from_ssd(0)[0]["up_proj.weight"].dtype == mx.float32
        assert store.stats()["page_dtype_conversions"] > 0


def test_mlx_expert_stacks_are_packed_once_per_forward_and_refreshed_after_update():
    mx = pytest.importorskip("mlx.core")
    import chronos.mlx.moe as mlx_moe
    from chronos.mlx.model import ChronosMLXModel
    cfg = tiny(recurrent_depth=3)
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(ChronosForCausalLM(cfg).eval(), cfg)
        calls = []
        real = mlx_moe.stack_expert_weights
        mlx_moe.stack_expert_weights = lambda experts: calls.append(1) or real(experts)
        try:
            ids = mx.array([[1, 4, 2, 5, 7, 3]])
            out1 = model(ids)[0]
            mx.eval(out1)
            moes = sum(1 for layer in model.layers if hasattr(layer.mlp, "_expert_stacks"))
            assert len(calls) == moes                      # one stack per physical MoE layer
            assert all(layer.mlp._pack_scope is None for layer in model.layers if hasattr(layer.mlp, "_pack_scope"))
            e = model.layers[1].mlp.experts[0]
            e.down_proj.weight = e.down_proj.weight * 0      # optimizer-style replacement
            calls.clear()
            out2 = model(ids)[0]
            mx.eval(out2)
            assert len(calls) == moes                      # re-packed, never stale
        finally:
            mlx_moe.stack_expert_weights = real


def test_mlx_sparse_dispatch_self_check_gates_backends():
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.moe_kernels import sparse_dispatch_ok
    with mx.stream(mx.cpu):
        assert sparse_dispatch_ok(mx.float32)
        assert not sparse_dispatch_ok(mx.bfloat16)        # CPU GatherMM is FP32-only -> dense fallback


# ---- Torch expert pages / checkpoints ---------------------------------------------

def test_cluster_storage_keeps_trunk_dtype_and_converts_only_on_request(tmp_path):
    from chronos.io.storage import ClusterStorage
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    cfg = tiny()
    model = ChronosForCausalLM(cfg).to(torch.bfloat16)
    moes = [layer.mlp for layer in model.model.layers if isinstance(layer.mlp, ChronosMOEFeedForward)]
    ClusterStorage.write_clusters(moes, [[0, 1], [2, 3]], str(tmp_path / "a"), len(moes), cfg.num_experts)
    page = ClusterStorage(str(tmp_path / "a")).load_cluster(0)[0][0]
    w = moes[0].experts[0].state_dict()["gate_proj.weight"]
    assert page["gate_proj.weight"].dtype == torch.bfloat16
    assert torch.equal(page["gate_proj.weight"].view(torch.int16), w.view(torch.int16))
    ClusterStorage.write_clusters(moes, [[0, 1], [2, 3]], str(tmp_path / "b"), len(moes), cfg.num_experts,
                                  dtype=torch.float16)
    assert ClusterStorage(str(tmp_path / "b")).load_cluster(0)[0][0]["gate_proj.weight"].dtype == torch.float16


def test_checkpoint_half_compacts_fp32_but_never_rounds_bf16_to_fp16(tmp_path):
    from chronos.model.checkpoint import save_state_dict_with_config
    cfg = tiny()
    fp32 = ChronosForCausalLM(cfg)
    save_state_dict_with_config(fp32, str(tmp_path / "a.pth"), cfg)
    assert {v.dtype for v in torch.load(str(tmp_path / "a.pth"), weights_only=True).values()
            if v.is_floating_point()} == {torch.float16}
    bf16 = ChronosForCausalLM(cfg).to(torch.bfloat16)
    save_state_dict_with_config(bf16, str(tmp_path / "b.pth"), cfg)
    saved = torch.load(str(tmp_path / "b.pth"), weights_only=True)
    ref = bf16.state_dict()
    assert all(torch.equal(saved[k], ref[k]) and saved[k].dtype == ref[k].dtype for k in ref)


def test_torch_masked_moe_dispatch_matches_per_token_oracle_values_and_grads():
    """Sorted single-sync dispatch of the fallback/offload path == explicit oracle."""
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    torch.manual_seed(8)
    model = ChronosForCausalLM(tiny(num_experts=6)).train()
    moe = next(m for m in model.modules() if isinstance(m, ChronosMOEFeedForward))
    x = torch.randn(2, 7, 64, requires_grad=True)
    avail = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)
    y = moe(x, available_expert_mask=avail)
    x_flat = x.view(-1, 64)
    scores = torch.softmax(moe.gate(x_flat).float(), -1)
    w, idx = torch.topk(scores, 2, -1, sorted=False)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) if moe.norm_topk_prob else w
    shared = moe._shared_expert_output(x_flat)
    rows = []
    for t in range(x_flat.shape[0]):
        acc = shared[t]
        for j in range(2):
            e = int(idx[t, j])
            out = moe.experts[e](x_flat[t:t + 1])[0] if avail[e] else shared[t]
            acc = acc + w[t, j] * out
        rows.append(acc)
    ref = torch.stack(rows).view_as(y)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-6)
    g = torch.randn_like(y)
    gx, gw = torch.autograd.grad((y * g).sum(), (x, moe.experts[0].up_proj.weight), retain_graph=True)
    rx, rw = torch.autograd.grad((ref * g).sum(), (x, moe.experts[0].up_proj.weight))
    torch.testing.assert_close(gx, rx, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(gw, rw, rtol=1e-4, atol=1e-6)
    assert moe.last_fallback_experts or not (~avail[idx]).any()
