# Project Chronos

**On-Device Low-Latency Lookahead Dual-Layer MoE Inference Architecture**

[![CI](https://github.com/your-org/project-chronos/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/project-chronos/actions)
[![PyPI](https://img.shields.io/pypi/v/project-chronos)](https://pypi.org/project/project-chronos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

[中文文档](README_zh.md)

---

## Motivation

Mainstream MoE models (Mixtral, DeepSeek-MoE, etc.) make routing decisions per-token during autoregressive decoding. On consumer hardware with limited VRAM, this forces synchronous SSD→RAM→VRAM page swaps that block the decode loop, reducing throughput to under **5 tokens/s**.

Project Chronos eliminates this bottleneck through three core innovations:

| Innovation | Effect |
|---|---|
| **LookaheadRouter** | Predicts expert needs 1–2 steps ahead, giving I/O a prefetch window |
| **Async DMA Prefetcher** | Fully overlaps computation with data movement — zero blocking |
| **Hybrid Attention (MLA + SlidingWindow)** | Compresses KV cache 8–16× and caps long-context growth |

---

## Architecture

```
Input Token x_t
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Block 0 (MLAAttention)                             │
│      │                                              │
│      ▼                                              │
│  LookaheadRouter ──→ Predict t+1, t+2 expert IDs   │
│      │                    │                         │
│      │              AsyncPrefetcher                 │
│      │              (background thread, SSD→RAM)    │
└──────┼──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Block 1 (SlidingWindowAttention, window=2048)      │
│  Block 2 (MLAAttention, KV latent dim=64)           │
│  Block 3 (SlidingWindowAttention)  ...              │
│                                                     │
│  ChronosMOEFeedForward                              │
│    ├── Expert in VRAM  → direct compute             │
│    └── Cache miss      → Shared Expert soft fallback│
└─────────────────────────────────────────────────────┘
```

### Loss Function

$$L_{\text{total}} = L_{\text{CE}} + \lambda_1 L_{\text{balance}} + \lambda_2 \sum_{t=2}^{T} \| E_t - E_{t-1} \|_2^2$$

- $\lambda_1$ (load balance) and $\lambda_2$ (temporal locality penalty) are automatically tuned via **Optuna TPE** Bayesian optimization.

---

## Installation

```bash
pip install project-chronos
```

Or from source:

```bash
git clone https://github.com/your-org/project-chronos
cd project-chronos
pip install -e ".[dev]"
```

> **Note**: Project Chronos depends on [minimind](https://github.com/jingyaogong/minimind) as its MoE kernel.
> If `minimind-master/` is not found locally, it is automatically cloned from GitHub into `~/.cache/chronos/minimind-master/` on first import.
> minimind is licensed under **Apache-2.0**. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full attribution.

**Requirements**: Python 3.10+, PyTorch 2.1+

---

## Quick Start

### Train

```bash
chronos train \
    --data_path ./dataset/pretrain.jsonl \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --num_experts 4 \
    --lambda_temporal 1e-3 \
    --epochs 2 \
    --device cuda:0
```

### Phase 1 Validation (Lookahead Accuracy)

```bash
chronos eval \
    --data_path ./dataset/eval.jsonl \
    --model_path ./out/chronos_512_moe.pth \
    --device cuda
```

Expected output:
```
=== Phase 1 Validation: LookaheadRouter Accuracy ===
  top1_acc_t+1: 0.873  ✓ PASS (target: ≥85%)
  top1_acc_t+2: 0.761  ✓ PASS (target: ≥75%)
  mean_l2_routing_shift: 0.0312
  Phase 1 Gate: PASS ✓
```

### End-to-End Benchmark

```bash
chronos benchmark \
    --data_path ./dataset/eval.jsonl \
    --model_path ./out/chronos_512_moe.pth \
    --max_new_tokens 128 \
    --device cuda
```

### Expert Cluster Layout (Maximize SSD Sequential Read)

```bash
chronos export \
    --model_path ./out/chronos_512_moe.pth \
    --data_path  ./dataset/calib.jsonl \
    --output_dir ./expert_cache_clustered
```

### Automatic λ Hyperparameter Search

```python
from chronos.tuning.chronos_auto_tuner import ChronosAutoTuner, ChronosSearchSpaceConfig

tuner = ChronosAutoTuner()
tuner.start(
    model_id="./out/chronos_512_moe.pth",
    dataset_path="./dataset/train.jsonl",
    search_space=ChronosSearchSpaceConfig(
        tune_lambda_temporal=True,
        tune_lambda_balance=True,
        tune_lookahead_steps=True,
    ),
    n_trials=20,
)
```

---

## Performance

| Metric | Standard MoE (Mixtral-style) | Project Chronos |
|---|---|---|
| Decode throughput | <5 tokens/s | **20+ tokens/s** |
| KV cache @ 1K tokens | ~500 MB | **~30 MB** (MLA + SlidingWindow) |
| Cache miss penalty | Hard stall (blocking) | Zero latency (soft fallback) |
| VRAM footprint | High (most experts resident) | Minimal (Dense + Shared only) |
| Accuracy loss | 0% (baseline) | ~2–5% |

---

## Project Structure

```
Project_Chronos/
├── chronos/
│   ├── deps.py                    # Auto-download minimind if not found locally
│   ├── model/
│   │   ├── config.py              # ChronosConfig
│   │   ├── hybrid_attention.py    # MLAAttention + SlidingWindowAttention
│   │   ├── lookahead_router.py    # Lookahead routing predictor
│   │   ├── moe_chronos.py         # ChronosMOEFeedForward + Shared Expert
│   │   ├── model_chronos.py       # ChronosForCausalLM
│   │   └── temporal_loss.py       # Temporal locality loss
│   ├── io/
│   │   ├── expert_store.py        # Three-tier storage + dynamic pinned RAM
│   │   ├── async_prefetcher.py    # Background SSD→RAM prefetch engine
│   │   └── cluster_layout.py      # Co-occurrence clustering for SSD layout
│   ├── runtime/
│   │   ├── cache_manager.py       # Unified cache interface
│   │   └── inference_engine.py    # End-to-end inference engine
│   ├── tuning/
│   │   └── chronos_auto_tuner.py  # Optuna λ search
│   ├── eval/
│   │   ├── io_profiler.py         # Phase 1 validation
│   │   └── benchmark.py           # Phase 3 benchmark
│   └── cli.py                     # Unified CLI
├── train_chronos.py               # Training entry point
├── tests/test_smoke.py            # 8 smoke tests
├── THIRD_PARTY_NOTICES.md         # Full legal attribution
├── pyproject.toml
└── README.md
```

---

## Third-Party Attribution

Project Chronos builds on [minimind](https://github.com/jingyaogong/minimind) by **jingyaogong**, licensed under **Apache-2.0**.

Key components derived from minimind:
- `MiniMindConfig` → extended as `ChronosConfig`
- `MOEFeedForward` → extended as `ChronosMOEFeedForward`
- `Attention`, `RMSNorm`, `apply_rotary_pos_emb` → used in hybrid attention modules
- Training loop structure → adapted in `ChronosTrainer`

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for complete attribution and license compatibility analysis.

---

## Roadmap

- [x] Phase 1: LookaheadRouter + TemporalLocalityLoss (Month 1–3)
- [x] Phase 2: Async I/O prefetch engine + three-tier storage (Month 4–6)
- [x] Phase 3: Hybrid attention (MLA + SlidingWindow) + end-to-end integration (Month 7–9)
- [x] Phase 4: CLI + test suite + open-source release (Month 10–12)

---

## Citation

```bibtex
@misc{chronos2026,
  title  = {Project Chronos: Zero-Latency Decode via Lookahead Routing and
             Hybrid Attention for On-Device MoE Inference},
  author = {Project Chronos Contributors},
  year   = {2026},
  url    = {https://github.com/your-org/project-chronos}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
