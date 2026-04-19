"""
tests/test_smoke.py — Project Chronos smoke tests for CI
"""
import sys
import os

# When running pytest from repo root, ensure chronos package is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import chronos.deps  # auto-bootstrap minimind on sys.path

import torch
from torch.utils.data import DataLoader, TensorDataset



def make_model(window=16):
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    cfg = ChronosConfig(
        hidden_size=128, num_hidden_layers=4, num_experts=4,
        use_moe=True, use_hybrid_attention=True,
        kv_latent_dim=16, rope_dim=8, sliding_window_size=window,
        vram_budget_gb=0.5,
    )
    return ChronosForCausalLM(cfg), cfg


def test_forward():
    model, cfg = make_model()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out, lp = model(x, labels=x)
    assert out.loss.item() > 0
    assert lp.shape == (2, 16, cfg.lookahead_steps + 1, cfg.num_experts)


def test_kv_cache_bounded():
    model, cfg = make_model(window=8)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out, _ = model(x, use_cache=True)
        past_kv = out.past_key_values
        for _ in range(30):
            xn = torch.randint(0, cfg.vocab_size, (1, 1))
            out2, _ = model(xn, past_key_values=past_kv, use_cache=True)
            past_kv = out2.past_key_values
    from chronos.model.hybrid_attention import SlidingWindowAttention
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.self_attn, SlidingWindowAttention):
            assert past_kv[i][0].shape[1] <= 8, f"Layer {i} KV not bounded"


def test_temporal_loss():
    from chronos.model.temporal_loss import temporal_locality_loss, total_loss
    probs = torch.rand(2, 16, 4)
    tl = temporal_locality_loss(probs)
    assert tl.item() >= 0
    ce = torch.tensor(2.0)
    bal = torch.tensor(0.01)
    loss = total_loss(ce, bal, probs, 5e-4, 1e-3)
    assert loss.item() > ce.item()


def test_lru_cache():
    from chronos.io.expert_store import LRUCache
    lru = LRUCache(capacity=3)
    lru.put(0); lru.put(1); lru.put(2)
    evicted = lru.put(3)
    assert evicted == 0
    assert lru.contains(3)
    assert not lru.contains(0)


def test_expert_store_init():
    from chronos.io.expert_store import ExpertStore
    model, cfg = make_model()
    store = ExpertStore(model, cfg, ssd_dir='/tmp/chronos_ci_test')
    s = store.stats()
    assert s['vram_capacity'] >= 1
    assert s['ram_capacity_dynamic'] >= 1
    assert 'available_ram_gb' in s


def test_async_prefetcher():
    import time
    from chronos.io.expert_store import ExpertStore
    from chronos.io.async_prefetcher import AsyncPrefetcher
    model, cfg = make_model()
    store = ExpertStore(model, cfg, ssd_dir='/tmp/chronos_ci_test')
    pf = AsyncPrefetcher(store)
    pf.start()
    pf.submit([0, 1])
    time.sleep(0.05)
    pf.stop()
    assert pf.stats['total_requests'] == 2


def test_hybrid_attention_layers():
    from chronos.model.hybrid_attention import MLAAttention, SlidingWindowAttention
    model, cfg = make_model()
    for i, layer in enumerate(model.model.layers):
        if i % 2 == 0:
            assert isinstance(layer.self_attn, MLAAttention), f"Layer {i} should be MLA"
        else:
            assert isinstance(layer.self_attn, SlidingWindowAttention), f"Layer {i} should be SW"


def test_benchmark_functions():
    from chronos.eval.benchmark import compute_perplexity, measure_throughput
    model, cfg = make_model()
    ids = torch.randint(0, cfg.vocab_size, (4, 16))
    ds = TensorDataset(ids, ids.clone())
    loader = DataLoader(ds, batch_size=2)
    ppl = compute_perplexity(model, loader, 'cpu', max_batches=2)
    assert ppl > 1.0
    tps = measure_throughput(model, ids[:1, :8], 'cpu', max_new_tokens=5)
    assert tps > 0


if __name__ == '__main__':
    tests = [
        test_forward,
        test_kv_cache_bounded,
        test_temporal_loss,
        test_lru_cache,
        test_expert_store_init,
        test_async_prefetcher,
        test_hybrid_attention_layers,
        test_benchmark_functions,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {t.__name__}: {e}')
    print(f'\n{passed}/{len(tests)} tests passed.')
    if passed < len(tests):
        sys.exit(1)
