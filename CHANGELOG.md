# Changelog

## [0.3.0] — 2026-11-06

v3 kernel: recurrent-depth MoE. Quality results for the 50M en→zh demo are not
included yet; they will be added after the 72-hour run and its ablations.

### Added
- Web UI Inference tab: self-speculative switch, draft depth and γ. All four runner paths use it (Torch / MLX, full DRAM / lazy experts). The stats table shows acceptance, tokens per verify, and verify / draft forwards. Models without a shared KV stream (`per_iteration`, depth 1) decode plainly and say so.
- `ChronosInferenceEngine.generate_speculative`: self-speculative decoding with lazy experts. Every draft and verify forward runs under the current availability masks, and the lookahead prediction is prefetched before it. `chronos.runtime.speculative.speculative_supported(cfg)` tells callers whether a model can draft.
- `tools/bench_speculative.py`: plain vs speculative decoding on the real runtimes. Reports wall time, tokens/s, acceptance, identical-output count and SSD loads, with the environment and checkpoint hash.
- Recurrent-depth execution: prelude → shared recurrent groups looped `recurrent_depth` times with a contractive update → coda.
- `recurrent_attention_mode`: `shared_first` (default; attention every iteration, K/V from iteration 0), `per_iteration` (ablation), `first_only` (v2 semantics).
- Tensor-only ACT early exit, exact and batch-invariant; per-token decode exit without KV gaps.
- Recurrent-depth self-speculative decoding (`chronos/runtime/speculative.py`): greedy-exact, rejection sampling, cache rollback.
- MLX self-speculative decode loop (`chronos/mlx/speculative.py`), Torch parity: same invariant, rejection sampling with `mx.random` keys, cache truncation for MLA, SWA and int8 entries. `ChronosMLXInferenceEngine.generate(speculative=True, draft_depth=, gamma=)` runs it with the lazy-expert masks; experts each draft routes to are unioned with the lookahead prediction and prefetched before the verify forward. `SlidingWindowAttentionMLX.cache_slack` keeps `window + γ_max + 1` entries for rollback.
- `chronos bundle build|pack|check|probe|translate|eval|prepare-data|finetune|serve`: a self-contained model bundle (weights + sidecar, tokenizer, FLORES+ dev/devtest, vendored minimind, wheel, source recipe, recommended fine-tune tiers, launcher, setup script, EN/ZH README, NOTICE, SHA-256 file table). `bundle translate` takes `--backend torch|mlx` and `--speculative`.
- `CHRONOS_MINIMIND_DIR` points Chronos at a local minimind copy (bundles set it to `third_party/minimind`), so no git clone or network access is needed.
- Stage scripts take `--tokenizer_path`; `load_stage_tokenizer` loads the checkpoint's own tokenizer and refuses one that does not fit `vocab_size`.
- `chronos longrun` spec fields `init_checkpoint` (start from existing weights, checked against `model_config`) and `export_prefix`.
- Sorted grouped MoE dispatch: `mx.gather_mm` on MLX (requires mlx >= 0.32.4), `argsort` + `index_add_` on Torch; per-expert reference paths kept for parity tests.
- Single-dtype precision chain for CPU / CUDA / MPS / MLX with FP32 only in RMSNorm statistics, SDPA softmax, RoPE tables, router softmax, LM-head logits and optimizer masters; bit-exact MLX↔Torch bf16 bridge.
- `chronos serve`: Ollama/OpenAI protocol server on the native runtime (lazy experts, speculative decoding); optional `/monitor/` mount.
- `chronos tokenizer`: self-built byte-fallback BPE with language-balanced merges and HF-compatible `tokenizer.json`.
- `chronos corpus download|build|shards`: pinned downloads, cleaning, global dedup, eval-set leak removal, uint16 shards.
- `chronos longrun` (alias `run`): time-budgeted resumable MLX training with NaN rollback, validation and translation probes, and run telemetry (`status.json`, `runtime.json`, `parameters.jsonl`).
- `chronos monitor`: read-only training monitor page + JSON API (loopback by default; key login, session cookie, rate limiting off loopback); Training Monitor tab in the Web UI.
- `chronos.model.factory.fit_config`: parameter-budget configuration search.
- Optional extras `serve` and `translate` with pinned versions.
- Inference and memory options, all off by default: `kv_cache_dtype="int8"` (MLA `c_kv` and SWA k/v, 32-element groups), `mla_decode="absorbed"|"auto"` with optional `mla_cache_rope`, `router_scope="shared"`, the `draft_exit_kl_coef` / `draft_exit_depth` training term, and hard expert budgets `max_resident_experts` / `ram_budget_mb` with prediction-protected LRU and per-virtual-layer prefetch. Torch and MLX both implement them.
- Adaptive training planner (`chronos.model.planner`, Config tab → "Auto plan"). Inputs: target size, MoE on/off, experts, and either active experts or active parameters, plus a memory-saving level and datasets. Outputs: a v3 architecture, scaling-law LR/batch, a memory fit, and independent pretrain / SFT / DPO / ORPO / GRPO / distill values. Dataset profiling supplies vocabulary, sequence length, token count and tokenizer path.
- Per-stage overrides (`config["stage_overrides"]`); the Train tab merges the selected stage's values.
- Five v3 presets generated by the planner (`ui/presets_v3.json`, `python -m ui.presets`) replace the old presets; old names are aliases.
- Tokenizer and Serve tabs in the Web UI; the Train tab honours `tokenizer_path`.

### Changed
- Routing defaults to top-2 plus a shared expert (top-1 with `norm_topk_prob=True` gave the gate no CE gradient).
- `--dtype auto` resolves to fp32 on MPS (native bf16 measured slower there).
- `lookahead_source` now comes from the prelude output.
- Repetition penalty divides positive logits and multiplies negative ones.
- `gradio` is constrained to `>=5.0,<6`; `tokenizers` is a declared dependency.
- `setup.py` is a shim; metadata lives only in `pyproject.toml`.
- `mla_decode` defaults to `expanded`. With `auto`, a bf16 prefill took the expanded path and the following decode took the absorbed path. Prefill+decode then differed from the full forward pass by 0.06 in max logit (measured). `absorbed` and `auto` are opt-in.
- Minimum torch is 2.6: transformers 5 disables its PyTorch backend below that version, which broke imports.
- CI installs torch 2.10.0 (CPU) and pins the rest in `.github/ci-constraints.txt`, tested on Python 3.10 and 3.11. The dead token-based publish job was removed; releases publish through `pypi.yml` (trusted publishing).
- The Web UI no longer assumes a task.
  - The monitor (page and Gradio tab) discovers evaluation metrics from whatever numeric keys a run logs on validation / final events. Keys are grouped by metric name (`probe_en_zh_bleu`, `qa_accuracy`, `pass1` …) and charted per family, with an improvement direction for known metrics.
  - Counts and durations are listed as context and not charted.
  - Samples show input / reference / output for any schema (`src`/`prompt`/`question`, `ref`/`answer`, `hyp`/`response`/`prediction`). Other fields appear as tags.
  - Offline inspection reports are flattened to every numeric leaf.
- Data tab: scans a file, directory or glob of jsonl / json / parquet / csv / tsv / txt. Reports record shapes (chat, preference, instruction, prompt–response, parallel pair, plain text), field presence and types, scripts, duplicates and per-record character lengths over all text fields.

### Fixed
- The Inference tab loaded the minimind tokenizer for every checkpoint. It now uses the checkpoint's own tokenizer: the config `tokenizer_path`, the bundle's `tokenizer/` folder, or the sidecar path. Minimind is the last resort.
- MLX inference ran in training mode when the model was converted from a freshly built torch model (the Inference tab did this). MoE then masked experts at random (`fallback_mask_prob`), so repeated greedy runs disagreed. `ChronosMLXInferenceEngine` now puts its model in eval mode.
- The Torch lazy engine failed with "Inference tensors cannot be saved for backward" when a plain request followed a speculative one, because experts paged in under `torch.inference_mode` were reused. The decode paths now use `torch.no_grad`.
- The v2 every-iteration attention switch rewrote a single KV slot per iteration (prefill+decode vs full forward gap 0.94).
- CLI registered `serve`, `tokenizer` and `corpus` twice.
- Monitor CPU time on macOS: `rusage` times are Mach ticks, now scaled by `mach_timebase_info`.
- `apply_inference_overrides` validated overrides against a config that used class-default KV heads (4). Every serve/inference override on models with other head counts (e.g. the 50M demo, 6/2) raised.
- The UI estimator ignored `recurrent_groups` and `recurrent_attention_mode`; its counts now match `count_parameters`.
- Planner/dataset profiling counted GRPO prompt files (empty assistant turn) as SFT.
- Stage scripts loaded the minimind tokenizer (6,400 entries) for checkpoints trained with another vocabulary.
- A forward with `recurrent_depth` below `min_recurrent_depth` (a speculative draft) gave the last state zero ACT weight on MLX, so the coda saw zeros (max logit gap to Torch 16.6 on the 50M checkpoint, measured; draft acceptance 2 % instead of 45 %). Torch special-cased depth 1 but had the same fault for 2 ≤ depth < `min_recurrent_depth`. The minimum now clamps to the executed depth; training depths are unaffected.
- MLX ≥ 0.32 binds a lazy `mx.load` to the loading thread's stream; experts prefetched by the background thread raised "There is no Stream(cpu, N) in current thread" when promoted on the decode thread. Pages are now evaluated in the thread that loads them.

### Known gaps
- Self-speculative decoding is exact but **not faster** for the 50M demo, on CPU, GPU or lazy experts. All figures are measured on the pretrain step-13,996 checkpoint, greedy, batch 1, FLORES+ dev en→zh, draft depth 1, γ 4. The 72-hour run was paused (SIGSTOP) for the GPU runs. Reports are in `reports/speculative/`.

  | Runtime | Plain tok/s | Speculative tok/s | Ratio | Acceptance | Identical |
  |---|---|---|---|---|---|
  | MLX GPU bf16, all experts in DRAM | 112.0 | 107.4 | 0.96× | 42 % | 22/24 |
  | MLX GPU bf16, lazy experts (12 of 216 pages) | 43.7 | 32.0 | 0.73× | 42 % | 22/24 |
  | Torch MPS fp32, all experts | 32.1 | 25.9 | 0.81× | 40 % | 12/12 |
  | Torch MPS fp32, lazy experts, sync misses | 17.0 | 14.7 | 0.86× | 40 % | 12/12 |

  Draft depth 2 raised acceptance to 55–59 % but did not change the result (0.72–0.92×).

  At this size one full-depth step costs about as much as a batched verify pass, so drafting at depth 1 of 4 does not save enough.

  With lazy experts, drafts and verify passes route to different experts. Speculative decoding then read 1.4–1.6× more expert pages from SSD (12,092 vs 8,372 for MLX).

  bf16 on the GPU flips near-tie tokens (the 2/24 and 3/24 mismatches). In GPU fp32 one sentence differed, at a top-1/top-2 logit margin of 0.0006. CPU fp32 matches exactly. The switch therefore stays off by default.
- At a 1280 px window, Gradio moves the Auto-Tune and IO Monitor tabs into its "…" overflow menu.
- Planner peak-memory figures are estimates; for the 50M recipe the estimate (16.4 GB) is about twice the measured MLX peak (8.72 GB).
- Gradio 5.35 does not render the LinePlot legend; colour keys are written into plot titles instead.

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
