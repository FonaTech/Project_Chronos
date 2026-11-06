"""Sparse expert dispatch must match the masked-dense oracle in value and gradient."""
from __future__ import annotations

import os
import sys

import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.model.config import ChronosConfig
from chronos.model.moe_chronos import ChronosMOEFeedForward


def moe_cfg(**kw):
    values = dict(hidden_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  vocab_size=64, num_experts=6, num_experts_per_tok=2, num_shared_experts=1,
                  moe_intermediate_size=40, dropout=0.0, fallback_mask_prob=0.0)
    values.update(kw)
    return ChronosConfig(**values)


def torch_dense_oracle(moe, x):
    """Every expert on every token, weighted by the routed top-k mask."""
    B, S, H = x.shape
    flat = x.reshape(-1, H)
    scores = torch.softmax(moe.gate(flat), dim=-1)
    weight, idx = torch.topk(scores, moe.num_experts_per_tok, dim=-1, sorted=False)
    if moe.norm_topk_prob:
        weight = weight / (weight.sum(-1, keepdim=True) + 1e-20)
    y = torch.zeros_like(flat)
    for e, expert in enumerate(moe.experts):
        y = y + expert(flat) * torch.where(idx == e, weight, 0).sum(-1, keepdim=True)
    y = y + sum(s(flat) for s in moe.shared_experts) / len(moe.shared_experts)
    return y.view(B, S, H)


@pytest.mark.parametrize("top_k", [1, 2])
def test_torch_sorted_dispatch_matches_dense_value_and_grad(top_k):
    torch.manual_seed(0)
    moe = ChronosMOEFeedForward(moe_cfg(num_experts_per_tok=top_k)).train()
    x = torch.randn(3, 7, 48, requires_grad=True)
    sparse = moe(x)
    grads_sparse = torch.autograd.grad(sparse.square().sum(), [x, *moe.parameters()], allow_unused=True)
    x2 = x.detach().clone().requires_grad_(True)
    dense = torch_dense_oracle(moe, x2)
    grads_dense = torch.autograd.grad(dense.square().sum(), [x2, *moe.parameters()], allow_unused=True)
    torch.testing.assert_close(sparse, dense, atol=1e-5, rtol=1e-5)
    for a, b in zip(grads_sparse, grads_dense):
        if b is None:
            continue
        torch.testing.assert_close(a if a is not None else torch.zeros_like(b), b, atol=1e-5, rtol=1e-4)


mx = pytest.importorskip("mlx.core")


def _mlx_case(dtype, seed=0, E=6, H=64, I=48, N=40, K=2):
    mx.random.seed(seed)
    with mx.stream(mx.cpu):
        params = [mx.random.normal(s) * 0.1 for s in ((E, I, H), (E, I, H), (E, H, I), (N, H))]
        idx = mx.random.randint(0, E, (N, K))
        weight = mx.softmax(mx.random.normal((N, K)), axis=-1)
        mx.eval(params, idx, weight)
    return params, idx, weight


def _value_and_grads(fn, params, idx, weight):
    """params = [w_gate, w_up, w_down, x]; grads w.r.t. all four."""
    def loss(*p):
        return fn(p[3], idx, weight, *p[:3]).astype(mx.float32).square().sum()
    value = fn(params[3], idx, weight, *params[:3])
    grads = mx.grad(loss, argnums=(0, 1, 2, 3))(*params)
    mx.eval(value, grads)
    return value, grads


def _rel(a, b):
    return (mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max() /
            (mx.abs(b.astype(mx.float32)).max() + 1e-9)).item()


@pytest.mark.parametrize("dtype_name,stream_name,tol", [
    ("float32", "cpu", 1e-5),
    # Apple M5 GPU fp32 matmul is TF32-class: dense and sparse both measure
    # ~0.2-0.6% relative error vs CPU fp32 (2026-11-05), so budget 1e-2.
    ("float32", "gpu", 1e-2),
    ("bfloat16", "gpu", 2.5e-2),
])
def test_mlx_grouped_gather_mm_matches_dense_and_fp32_reference(dtype_name, stream_name, tol):
    from chronos.mlx.moe_kernels import dense_reference_swiglu, gather_mm_supported, grouped_swiglu
    if not gather_mm_supported():
        pytest.skip("mlx < 0.32.4: gather_mm unreliable on this GPU; dense path is used")
    dtype = getattr(mx, dtype_name)
    params32, idx, weight32 = _mlx_case(mx.float32)
    with mx.stream(mx.cpu):
        ref_value, ref_grads = _value_and_grads(dense_reference_swiglu, params32, idx, weight32)
    stream = mx.cpu if stream_name == "cpu" else mx.gpu
    with mx.stream(stream):
        params = [p.astype(dtype) for p in params32]
        weight = weight32.astype(dtype)
        sparse_value, sparse_grads = _value_and_grads(grouped_swiglu, params, idx, weight)
        dense_value, dense_grads = _value_and_grads(dense_reference_swiglu, params, idx, weight)
    # Both paths meet the absolute budget against the fp32 CPU reference ...
    assert _rel(sparse_value, ref_value) <= tol
    assert _rel(dense_value, ref_value) <= tol
    for s, d, r in zip(sparse_grads, dense_grads, ref_grads):
        sparse_err, dense_err = _rel(s, r), _rel(d, r)
        assert sparse_err <= tol, (sparse_err, dense_err)
        assert dense_err <= tol, (sparse_err, dense_err)
        # ... and sparse is never materially worse than the dense oracle.
        assert sparse_err <= 2.0 * dense_err + 1e-6, (sparse_err, dense_err)


@pytest.mark.parametrize("dtype_name,tol", [("float32", 1e-2), ("bfloat16", 3e-2)])
def test_mlx_moe_module_sparse_matches_dense_value_and_grad(dtype_name, tol):
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from chronos.mlx.moe import ChronosMLXMOE
    from chronos.mlx.moe_kernels import gather_mm_supported
    if not gather_mm_supported():
        pytest.skip("mlx < 0.32.4")
    dtype = getattr(mx, dtype_name)
    mx.random.seed(3)
    moe = ChronosMLXMOE(moe_cfg())
    moe.update(tree_map(lambda a: a.astype(dtype), moe.parameters()))
    x = (mx.random.normal((2, 9, 48)) * 0.5).astype(dtype)

    def run(sparse):
        moe.sparse_dispatch = sparse
        value, grads = nn.value_and_grad(moe, lambda inp: moe(inp).astype(mx.float32).square().sum())(x)
        mx.eval(value, grads)
        return value, dict(tree_flatten(grads))

    dense_value, dense_grads = run(False)
    sparse_value, sparse_grads = run(True)
    assert abs(sparse_value.item() - dense_value.item()) <= tol * abs(dense_value.item())
    assert sparse_grads.keys() == dense_grads.keys()
    for key, dense_grad in dense_grads.items():
        assert _rel(sparse_grads[key], dense_grad) <= tol, key



def test_mlx_grouped_fallback_matches_dense_oracle():
    from chronos.mlx.moe_kernels import dense_reference_swiglu, gather_mm_supported, grouped_swiglu
    if not gather_mm_supported():
        pytest.skip("mlx < 0.32.4")
    params, idx, weight = _mlx_case(mx.float32)
    with mx.stream(mx.cpu):
        available = mx.array([True, False, True, True, False, True])
        shared = mx.random.normal((params[3].shape[0], params[3].shape[1]))
        def run(fn):
            def loss(*p):
                return fn(p[3], idx, weight, *p[:3], fallback=(available, shared)).square().sum()
            return fn(params[3], idx, weight, *params[:3], fallback=(available, shared)), \
                mx.grad(loss, argnums=(0, 1, 2, 3))(*params)
        sv, sg = run(grouped_swiglu)
        dv, dg = run(dense_reference_swiglu)
        mx.eval(sv, sg, dv, dg)
    assert _rel(sv, dv) < 1e-5
    for a, b in zip(sg, dg):
        assert _rel(a, b) < 1e-4
    # Unavailable experts receive no gradient (they were not executed).
    assert mx.abs(sg[0][1]).max().item() == 0.0 and mx.abs(sg[0][4]).max().item() == 0.0
