# Changelog

## [0.1.0] — 2026-04-20

### Added
- `ChronosConfig`: extends `MiniMindConfig` with lookahead, shared experts, λ1/λ2, hybrid attention params
- `LookaheadRouter`: lightweight dense predictor inserted after block 0, outputs [B, S, K+1, E] routing probs
- `TemporalLocalityLoss`: L_total = L_CE + λ1·L_balance + λ2·Σ‖E_t − E_{t-1}‖²
- `ChronosMOEFeedForward`: shared experts always in VRAM + soft gating fallback on cache miss
- `MLAAttention`: Multi-head Latent Attention — KV cache stores latent vectors (8-16x smaller)
- `SlidingWindowAttention`: KV cache capped at `window_size` tokens, O(1) memory per decode step
- Hybrid attention: even layers → MLA, odd layers → SlidingWindow
- `ExpertStore`: three-tier VRAM/RAM/SSD storage with dynamic pinned memory safety limit
- `AsyncPrefetcher`: background thread prefetches SSD→RAM driven by LookaheadRouter predictions
- `CacheManager`: unified cache interface for inference engine
- `ChronosInferenceEngine`: end-to-end decode loop with async prefetch + soft gating
- `cluster_layout.py`: co-occurrence matrix + Louvain/greedy clustering for sequential SSD layout
- `ChronosAutoTuner`: extends Optuna TPE search with λ1/λ2/lookahead_steps dimensions
- `chronos eval`: Phase 1 validation — t+1/t+2 lookahead accuracy + LRU cache hit rate
- `chronos benchmark`: Phase 3 — PPL, TPS, KV cache memory, layer-wise analysis
- `chronos export`: expert cluster layout generation
- 8-test smoke suite (`tests/test_smoke.py`)
- GitHub Actions CI (Python 3.10/3.11, lint + tests)
- PyPI packaging (`pyproject.toml`, `MANIFEST.in`)
