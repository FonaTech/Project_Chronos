# Changelog

## [Unreleased]

## [0.3.2a2] — 2026-11-09 (beta)

Adds narrower recomputation, two lookahead options and web UI fields for every long-context switch, and fixes a crash found by training on the real corpus. Every new switch is off by default. `reports/long_context_20261109/real_corpus_ab.json` holds the measurements below.

### Added
- `recompute_scope="mlp"`: with `activation_checkpointing`, recompute only the MoE/FFN of each block (`"block"` recomputes the whole block, as before). At 32k on the real corpus it saves 1.4 GB (5%) and loses 11% speed, against 6.9 GB (25%) and 19% for whole blocks, so it suits runs that need only a little more memory.
- `mla_upproj_recompute`: keep only the MLA latent and rebuild K and V per attention block. Exact, but it saved no memory (24.47 -> 24.29 GiB at 64k) and costs 6.7% speed, so it stays off.
- `lookahead_input_detach` (the lookahead router reads a gradient-free hidden state) and `temporal_grad_span` (cut the temporal-locality gradient chain every n positions). Both are exact in the forward pass. On the en-zh data they were neutral (validation CE within 1e-3), so they stay off.
- Web UI: every long-context field (`attention_block_size`, `ce_chunk_size`, `moe_chunk_tokens`, `recompute_scope`, `recompute_policy`, `mla_upproj_recompute`, `document_masking`, `reset_router_at_documents`, `reset_lookahead_at_documents`, `lookahead_input_detach`, `temporal_grad_span`) with labels and help in all four languages.

### Fixed
- `reset_router_at_documents` and `document_masking` raised "Attempting to eval an array without a primitive" inside the compiled LongRun step, because the document ids were built with NumPy from a traced array. `chronos.expert_clusters.segment_ids_mx` builds them in MLX, and a regression test runs the option through `mx.compile`.
- `reset_lookahead_at_documents="false"` (the string the web UI sends) was read as true. Strings are now parsed; the empty string follows the router switch.

### Measured on the real en-zh corpus (production 50M config, random init, M5 Max, bf16)
`long` = `attention_block_size=1024`, `ce_chunk_size=1024`, `moe_chunk_tokens=8192`. Validation CE differs by at most 3e-3 between arms.

| Context | Default: tokens/s, peak | long: tokens/s, peak |
|---|---|---|
| 8k | 8.9k, 40.6 GiB | 19.5k, 7.5 GiB |
| 16k | 1.9k, 141.5 GiB (over this machine's 128 GB) | 15.2k, 13.8 GiB |
| 32k | not run | 8.8k, 27.0 GiB |
| 64k | not run | 3.7k, 54.6 GiB |

Larger attention blocks speed up one MLA layer (0.62 s -> 0.53 s at 64k with blocks of 8192) but slow the whole model, because the sliding-window layers get slower (9.9 s -> 16.6 s per step); 1024 stays the recommendation.

## [0.3.2a1] — 2026-11-09 (beta)

This release lets the MLX trainer run long contexts on one Mac. Every new switch is off by default, so existing configs, checkpoints and recipes behave as before.

### Added
- `attention_block_size`: blockwise causal attention (`chronos/mlx/blockwise_attention.py`). Each block of queries reads only the keys it can see, under `mx.checkpoint`, and a block that ends at its last visible key uses the kernel's `"causal"` mode with no mask array. Value difference 0, gradient difference ≤ 8.5e-7 against the full-mask path.
- `ce_chunk_size`: chunked cross-entropy from the final hidden state, so no `[B, S, V]` logits exist (`chronos/mlx/chunked_ce.py`). Loss difference 0, gradient relative difference ≤ 5e-6.
- `moe_chunk_tokens`: MoE sparse dispatch in token chunks, each recomputed in the backward pass, including the cache-miss simulation.
- `recompute_policy` (`all`, `mla`, `swa`, `alternate`) chooses which blocks `activation_checkpointing` recomputes, on both the MLX and the Torch side.
- `rope_train_scaling` (`yarn`, `linear`, `ntk`): training-time RoPE scaling; MLX and Torch tables agree to 2e-5.
- `document_masking` and `reset_router_at_documents`: attention and the temporal and locality losses stay inside each packed document. `reset_lookahead_at_documents` gives the lookahead terms their own switch (default: follow the router switch).
- `PhaseSpec.seq_len_curriculum` (`[[progress, seq_len], ...]`) steps the pretrain window length up during a phase, and `PhaseSpec.pack_sft` joins short SFT records into full rows separated by the document separator (`SFTRecords.packed_batch`).

### Fixed
- The lookahead terms with several routing groups paired document ids of the wrong batch size when `reset_router_at_documents` was on.

### Measured
Full MLX training step (hidden 512, 8 layers, 16 experts top-2, bf16, batch 1, M5 Max). "long" = `attention_block_size=1024`, `ce_chunk_size=1024`, `moe_chunk_tokens=8192`; "+recompute" adds `activation_checkpointing`.

| Context | Default: tokens/s, extra peak | long: tokens/s, extra peak | long + recompute: tokens/s, extra peak |
|---|---|---|---|
| 2k | 20.5k, 2.96 GiB | 23.0k, 2.16 GiB | 19.4k, 1.50 GiB |
| 8k | 9.9k, 21.5 GiB | 23.3k, 5.55 GiB | 19.3k, 2.87 GiB |
| 16k | 4.3k, 76.2 GiB | 18.1k, 6.76 GiB | 15.6k, 4.76 GiB |
| 32k | not run (exceeds RAM) | 11.7k, 12.7 GiB | 9.7k, 8.3 GiB |
| 64k | not run (exceeds RAM) | 6.6k, 24.5 GiB | 5.9k, 15.4 GiB |

A single MLA layer at 64k needs 148.6 GiB extra without blocks and 2.5 GiB with them.

## [0.3.2a0] — 2026-11-08 (beta)

This release replaces the minimind kernel with a native one, compiles the MLX training step, adds MLX activation recomputation, and keeps tool-call tags that a custom tokenizer registers as special tokens. README "Native kernel (0.3.2)" has the migration reasons and the measured comparison.

### Changed
- `chronos/model/kernel.py` replaces the minimind kernel (config base, RoPE/YaRN, gated FFN, dense attention, trainer utilities).
  - `import chronos` no longer clones anything or changes `sys.path`, and works offline.
  - Config field names and defaults, RoPE tables and FFN parameter names are unchanged. Logits are bit-identical on `out/chronos_256_moe.pth` and the 50M en→zh checkpoint (full, padded and prefill+decode paths).
  - All layers share one FP32 RoPE table per `(dim, base, scaling, device)`: 3.50 → 0.50 MiB at the 50M shape, 216 → 24 MiB at the 768H base shape.
  - Resume state is written to `chronos_<H>_moe_resume.pth`; the release checkpoint keeps its dtype.
- `chronos.deps` is a shim. `get_tokenizer_path()` returns the bundled legacy 6400-token tokenizer (`CHRONOS_DEFAULT_TOKENIZER` overrides it); `ensure_minimind()` is a no-op with a deprecation warning.
- `tools/compare_minimind_chronos*.py` use minimind only as an external baseline, from `--minimind-dir` or `CHRONOS_MINIMIND_DIR`. Model bundles no longer ship `third_party/minimind`.
- `benchmark_compare.py` compares Chronos with a dense GQA baseline built from the same kernel (result key `baseline`, formerly `minimind`).
- The preset "MiniMind-MoE (default)" is now "Chronos-v2 legacy (768H, 6400 vocab)", and `MINIMIND_MOE_DEFAULTS` is now `BASE_MOE_DEFAULTS`. The old names remain as aliases.
- MLX training micro steps are compiled by default (`chronos longrun`, and `run_mlx_stage` pretrain and SFT). `CHRONOS_MLX_COMPILE=auto|on|off` or a recipe's `extra.mlx_compile` switches it.

### Added
- `chronos.mlx.training.compiled`: one `mx.compile` graph per signature for loss and gradients.
  - The recurrent depth is sampled once per step outside the trace. Depth, the ACT switch, shapes and plain config values select the graph.
  - Weights and the RNG state are compile inputs and outputs, so dropout and the cache-miss simulation consume the RNG stream as in eager mode.
  - Router probabilities, loss components and recurrent diagnostics are returned from the graph and written back to the modules.
  - Multimodal batches, lazy experts and signatures that fail to trace run eagerly; the reason is logged once.
  - Measured on real en→zh shards with the 50M config (M5 Max, bf16, shared GPU): 1.08× (SFT) and 1.18× (pretrain) per step, loss curves within 2e-3 of eager.
- `chronos.mlx.checkpointing`: MLX activation recomputation for `activation_checkpointing=true`.
  - Each block call is one recomputed region. Dropout masks, the cache-miss draws and routing indices are recorded in the forward pass and replayed during recomputation.
  - Region outputs are packed per dtype. mlx 0.32.4 keeps the gradient of only one copy when a checkpointed function returns the same graph node twice, and a full-length cache slice is such a copy.
  - Measured peak memory: 5.62 → 3.50 GB (SFT) and 6.07 → 3.62 GB (pretrain), at 17% and 21% more time per step than the compiled step alone. It stays off by default at 50M; the planner's memory-constrained tiers enable it.
- `chronos.tokenizer.visible.decode_visible`: hides control tokens and keeps `<tool_call>`, `</tool_call>`, `<tools>` and `</tools>` as text, for HF tokenizers and `ChronosTokenizer`. Serving, the workbench and GRPO rewards use it.
- Tests: `tests/test_native_kernel.py`, `tests/test_tool_tags_special_tokens.py`, `tests/test_mlx_checkpointing.py`, and per-micro-step exactness of the compiled and recomputed steps inside the LongRun loop (`tests/test_longrun.py`).

### Fixed
- Tool calls were lost when a custom HF tokenizer registered `<tool_call>` as a special token: decoding with `skip_special_tokens=True` removed the tags, so the parser saw bare JSON.
- Dense mode (`use_hybrid_attention=False`) crashed on Torch in every recurrent mode, and on MLX with an int32 overflow (a 2**63 window). It now runs dense causal GQA; prefill+decode matches the full forward within 1e-6.
- The Torch pretrain save rewrote the BF16 release checkpoint as FP16.
- Trainer log lines did not reach the Web UI.
- RoPE tables: with many torch threads, a cached table's prefix differed by one ulp from a shorter table (measured with 18 threads on torch 2.10). The tables are now built single-threaded.
- `tools/compare_minimind_chronos*.py` used `chronos.deps` without importing it.

## [0.3.1a3] — 2026-11-08 (beta)

This release fixes the router masks and the lazy-expert decode. It also adds a document-level cluster objective, MLX multimodal training in all six stages, tool-call SFT records and tool-call parsing in `chronos serve`, a kernel-level MLX cache limit and progress-based ETAs.

### Added
- `cluster_locality_mode="segment"`: the cluster objective treats each document as one unit.
  - Documents are found from the tokens: `<|endoftext|>` separators, and long documents are cut every `cluster_locality_window` tokens. No labels or domain tags are used.
  - The loss is `1 − I(document; cluster) / log C`, pooled over real tokens and all virtual layers.
  - The default stays `"window"`, the 0.3.1a2 objective.
  - Torch and MLX give the same value and gradient. The option is on the Web UI config page (advanced) in four languages.
- MLX multimodal training. `chronos.mlx.multimodal` ports the image patch encoder, the audio conv encoder and the gated cross-attention fusion.
  - Pre-encoded `*_embeds` inputs and the per-modality and per-token masks work as in Torch.
  - Logits match Torch within 2e-4 in FP32 tests.
  - MLX checkpoints save and load the fusion weights, and `freeze_modality_encoders` works on MLX.
  - `run_mlx_stage` takes media in all six stages: pretrain, SFT, DPO, ORPO, GRPO and distill.
  - DPO and ORPO: the chosen and rejected answers condition on the prompt's media.
  - GRPO: every rollout conditions on the prompt's media. The rollout encodes the media once and reuses it at every decode step, with identical logits. The policy update re-encodes the media, so the encoders get gradients.
  - Distill: the student gets the media. A text-only teacher is accepted, as on PyTorch.
- Multimodal preference and GRPO data.
  - `MultimodalStreamingDPODataset` and `preference_collate` read `chosen`/`rejected` records that carry `image`, `audio` or `*_embeds` fields, the same fields SFT uses.
  - `load_grpo_prompt_records` keeps each prompt's record so the media can be resolved per step.
  - `record_media` is the shared resolver.
  - The Web UI selects these automatically when the dataset contains media.
  - The PyTorch DPO, ORPO and GRPO trainers are text-only. With media, `auto` picks MLX, and an explicit PyTorch backend raises an error instead of dropping the media.
- Tool calls in the chat template.
  - `tools` render as one `<tools>` system block.
  - The assistant's `tool_calls` render as `<tool_call>{json}</tool_call>` lines, and `tool` turns carry the results.
  - Conversations without tools render exactly as before.
- `chronos.data.chat_records` provides `render_chat`, `encode_chat` and `write_sft_records`.
  - Supervised tokens: the assistant text, its tool calls and `<|im_end|>`.
  - Context tokens: system and user turns, tool definitions and tool results.
  - Records are written as `.bin/.idx/.plen/.lmask`.
- `SFTRecords` reads `.lmask` (one uint8 per token). When present it replaces the `.plen` prompt rule. Older readers still use `.plen`.
- `chronos serve` supports tool calls end to end.
  - It passes `tools` and the assistant's `tool_calls` from `/api/chat` and `/v1/chat/completions` into the template.
  - When the request has `tools`, it parses `<tool_call>` blocks in the output into structured calls. `/api/chat` returns Ollama `message.tool_calls`. `/v1/chat/completions` returns OpenAI `tool_calls` with string arguments and `finish_reason: "tool_calls"`.
  - Streaming parses the calls as they are generated, and no client sees part of a tag.
  - An invalid block stays in the content. Requests without `tools` are unchanged.
  - `chronos.serving.tool_calls` holds the parser.
- `chronos.mlx.memory`: every MLX model sets the MLX buffer-cache limit before it allocates weights.
  - The limit is `min(RAM/8, 16 GiB)`.
  - A recipe (`extra.mlx_cache_limit_gb`) or `CHRONOS_MLX_CACHE_LIMIT_GB` can lower it. Larger values are clamped to the limit.
  - `LongRun` logs the limit and records it in `runtime.json`.
  - Only reclaimable cache is bounded. Weights, activations and gradients are not.
- Monitor and Web UI ETAs follow measured progress.
  - A phase ends at the first of three points: the wall deadline, the step total, or `target_passes`.
  - The pass rate comes from row timestamps over the last 30 minutes, so evaluation and checkpoint time count.
  - Each phase shows its estimated end and which of the three points sets it.
  - Later phases also get the current phase's unused time.
  - A stopped or failed run shows its remaining work and its state, but no end time.

### Fixed
- Router auxiliary losses and routing telemetry counted right-padding in SFT batches. Balance, entropy, lookahead and cluster terms now use only real tokens. Packed pretraining is unchanged.
- Training telemetry now sums every micro-batch of a step. Before, it sampled only the last one.
- `SFTRecords` with `.lmask`: under NumPy 2, the -100 context labels of uint16 shards wrapped to 65436. Labels are now built in int32.
- MLX lazy-expert decode used the per-expert loop, which sums in a different order from the full-memory grouped kernel. In BF16 on the GPU this flipped near-tied greedy tokens: agreement was 31.25 % at every budget. The lazy path now runs the same grouped kernel over the live experts. Lazy and full greedy now agree at every budget (measured: 0.278, 0.5, 0.778 and 1.0 of experts, planned and window residency).

## [0.3.1a2] — 2026-11-07 (beta)

Kernel/extension split: the `chronos` kernel trains general models and holds no translation-specific code. The en↔zh demo moved to `demos/translation` (`chronos-translate`), which imports the kernel and is not part of the `Project_Chronos` wheel. The extension is not published on PyPI; install it from a checkout with `pip install -e demos/translation` (see `demos/translation/README.md`).

### Moved (to `demos/translation/chronos_translate/`)
- `chronos.data.corpus.*` → `chronos_translate.corpus.*` (WMT19/OPUS-100/FLORES+ download, cleaning, dedup, en–zh shards, doc-mix sets).
- `chronos.bundle.*` → `chronos_translate.bundle.*`; `chronos bundle …` → `chronos-translate bundle …`. A bundle's `bin/chronos bundle …` still works: the launcher sends `bundle` to `chronos-translate`. `bin/setup.sh` installs both wheels from `wheels/`; `bundle build --wheel` can be repeated.
- `chronos.mlx.training.doc_run` → `chronos_translate.doc_run` (`chronos-translate doc-run`).
- `inspect_checkpoint.translation_probe` → `chronos_translate.probe` (`chronos-translate inspect`).
- `chronos.model.factory.ENZH_DEMO_BASE` → `chronos_translate.model_presets.ENZH_50M_BASE`.
- `chronos/mlx/training/recipes/enzh_50m_72h.json` → `demos/translation/recipes/`, with paths relative to the file.

### Added
- Monitor (`chronos monitor`): a "Lazy experts / 懒加载" tab shows every metric the lazy-expert probe records after each validation. It has four tables (efficiency with the full-memory reference row; paging and loads; hit rates; fidelity), per-variant history charts (hit rate, fidelity, speed, loads, peak RSS) and the full probe log with skip and failure reasons. New endpoint: `GET /api/lazy_probes?run=…&limit=…`.
- Web UI monitor page: the probe card shows the same four grouped tables and four history charts in all four UI languages. Previously it showed a subset of the columns and no load, timing or reference metrics.
- The run snapshot's `lazy_probe` entries carry every measured probe field plus `reference_measured` (the full-memory reference per backend).
- Probes run from Web UI training jobs. For a CLI run (`chronos longrun`, `chronos-translate run`), start `chronos lazy-probe --watch <run_dir>` next to it, or the tab stays empty.

### Changed
- `LongRun` has task hooks: `task_eval(phase, n, final=)` (returns `{}` in the kernel), `eval_samples_for`, `generate`, `write_samples`, `pinned_modules`. The kernel `RunSpec` keeps `eval_samples` / `final_eval_samples` (default 0) and `export_prefix` (default `chronos`); `bleu_samples` and `probe_samples` moved to `TranslationRunSpec`. Run translation recipes with `chronos-translate run --spec …`; the kernel runner rejects unknown keys and names the task runner.
- `chronos longrun` and `load_spec` resolve relative recipe paths against the recipe's folder (they resolved against the working directory before). `--max-steps` and `--run-dir` give a short smoke run.
- Task-eval exceptions during SFT validation and the final evaluation are logged as `task_eval_error` instead of stopping the run (pretrain probes were already logged). The smoke path still raises.
- Tokenizer special tokens: the kernel default is `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|pad|>` and reserved slots up to 64. Task tokens come from `special_tokens_with(extra)` or `chronos tokenizer --extra-special …`. `special_tokens_with(["<|en|>", "<|zh|>", "<|translate|>"])` reproduces the 0.3.1a1 order, so existing tokenizers and checkpoints are unchanged.
- The shard readers (`PackedTokenStream`, `SFTRecords`, `dtype_for`, `manifest_dtype`) live in `chronos.data.shards`. `median_record_tokens` and `cluster_locality_config` moved to `chronos.expert_clusters`.
- Preset "Chronos-v3 50M (en→zh recipe)" is now "Chronos-v3 50M (16k vocab)" (the old name is an alias) and uses the default 20 tokens/param schedule instead of the production-shard measurement. Same shape: 49,691,717 total / 12,090,437 active parameters (measured, meta device).

### Fixed
- Web UI monitor page: a card with nothing to show no longer prints a stray "null".

### Removed
- `chronos corpus`, `chronos bundle` and the `translate` extra (`sacrebleu`, `sacremoses` are dependencies of `chronos-translate`).
- Translation metric labels (`bleu_zh`, `flores_devtest_n`) from the Web UI i18n files; the monitor still shows any numeric metric a task writes.

## [0.3.1a1] — 2026-11-07 (beta)

Web UI half of 0.3.1: the rewritten Web UI, file picking and upload, LAN access, and the lazy-expert probe.

### Changed
- `chronos-ui` and `python chronos_app.py` start the new Web UI (`chronos/webui/`: FastAPI and a hand-written single-page app, no build step, CSP `script-src 'self'`). The Gradio console stays available as `--legacy`.
- The UI is complete in zh-Hans, zh-Hant, en and ja. It follows the browser language and falls back to English. Buttons, chips, dropdown options and picker shortcuts are translated; device and format ids (cuda, float16, gguf) are shown as they are.
- Hover help appears after the pointer rests for 2 s; the "?" mark shows it after 0.4 s and Tab focus shows it at once.

### Added
- Path fields have two ways to pick a path.
  - "Browse…" lists folders on the server's machine.
  - "Upload…" uses the system file manager. On the server's own machine (loopback) the server opens the native dialog (macOS `osascript`, Windows PowerShell, Linux `zenity`/`kdialog`) and the chosen path is used in place. From another machine the files are streamed to `<workdir>/uploads/<datasets|checkpoints|configs|folders|files>/<batch>/` and that path is used.
  - Uploads refuse traversal, hidden names and executable or script types. They stop while less than max(8 GiB, 5 % of the disk) would stay free. `--max-upload-mb` caps a single file (default 20480).
- Dataset fields take several files or folders, one per line. Training stages and export calibration merge them into one JSONL file (`chronos/workbench/files.py:merge_datasets`, with a `.sources.json` manifest). An unchanged selection reuses the earlier merge.
- `--lan` opens the UI to the local network; it is off by default. Browsers on other machines sign in with `CHRONOS_UI_KEY`, or with a key printed at start when none is set. The server's own machine needs no sign-in. The key crosses plain HTTP in clear text unless `--ssl-certfile`/`--ssl-keyfile` or a TLS proxy is used. `--no-native-dialog` turns the native dialog off (for example over SSH).
- `chronos lazy-probe`: lazy-expert efficiency, hit rate and fidelity of a checkpoint, or of each validation of a run with `--watch`; the Train page can run it at every validation.

## [0.3.1a0] — 2026-11-07 (beta)

Core half of 0.3.1: cluster-locality training on by default, inference that
adapts its expert paging to how the checkpoint was trained, and lazy experts in
the bundle. The rewritten Web UI follows in 0.3.1a1.

### Changed
- Cluster locality is on by default for new training: `train_chronos*.py --lambda_cluster_locality` defaults to 0.3, the planner and presets set λ = 0.3 for MoE models with more than 2k experts, and the window follows the data (`recommended_window`: power of two nearest the median SFT record, clamped to [16, seq_len]; 32 without record lengths). The `ChronosConfig` class default stays 0.0, so a stored recipe without the field keeps its original objective on resume, and checkpoints written before the field existed load as λ = 0. The λ = 0.3 default comes from the synthetic sweep below (from-scratch training); its effect on real data and on fine-tuning has not been measured.
- Lazy-expert inference picks its page layout from the checkpoint (`paging="auto"`): weights trained with λ > 0 are paged one training cluster per SSD read, other weights one expert per read. `paging="cluster"|"expert"` forces a layout. The layout is part of the expert-cache directory name (`…_mlx_c2`, `…_c2`), so a cache written for one layout is never reused for the other. Per-expert cache names are unchanged.

### Added
- `chronos/lazy_experts.py`: one lazy-expert implementation for the Web UI, bundle and server — `LazyOptions` (paging, residency, decay, decode/verify waits, RAM ratio; validated), `page_layout`, `checkpoint_cache_dir`, the expert budget, and `build_mlx_lazy_engine` (trunk only in memory, cache built or reused).
- Torch offload writes cluster layouts as `cluster_<c>.ctsr` files with a v2 manifest (`ClusterStorage.write_clusters_from_states`); `setup_from_state_dict/_reader(clusters=…)`.
- MLX `generate(decode_wait_s=0.012, verify_wait_s=0.02, repetition_penalty=1.0)`. The penalty uses the CTRL rule and is rejected together with `speculative=True`, which needs penalty 1.0 to stay exact.
- `chronos bundle translate|eval --lazy-experts` with `--paging`, `--residency`, `--decay`, `--ram-ratio`, `--decode-wait`, `--verify-wait`. The cache lives outside the bundle (`$CHRONOS_EXPERT_CACHE`, default `~/.cache/chronos/expert_cache`); `eval` also gains `--backend` and `--dtype`.
- `chronos bundle finetune --[no-]cluster-locality --lambda-cluster --cluster-window --clusters` (on by default; dense models and models with fewer than two clusters stay off). The spec records where each value came from (`extra.cluster_locality_source`); the window comes from the median SFT record length of `--data`.
- `chronos serve --paging auto|cluster|expert`; `/api/ps` metadata reports the resolved layout.
- Web UI Inference tab: paging, planner decay and the two prefetch waits, with four-language labels.

- Next-use residency planning for lazy MLX experts (`chronos/residency_planner.py`, `generate(residency="planned")`). The planner records the page-access trace of every forward, keyed by (layer, loop iteration). The previous forward of the same kind, and in speculative decoding the queue of draft traces since the last verify, then predicts when each page is needed next. The store evicts the page needed latest (`MLXExpertStore.set_eviction_ranker`). The running dispatch's pending experts are never evicted. Pages needed early are prefetched only into spare slots. Greedy output is unchanged. The Web UI Inference tab has a switch for it, on by default (MLX lazy only); the stats table shows the eviction policy.
- Cluster-locality training constraint `lambda_cluster_locality`, with `cluster_locality_clusters` (0 = ⌈E/2k⌉) and `cluster_locality_window`. The loss is 1 − I(window; cluster)/log C over every MoE dispatch, recurrent iterations included. It keeps a stream inside few expert clusters and spreads different streams over different clusters. The partition (`chronos/expert_clusters.py`) is the one lazy inference pages by. Torch and MLX trainers and all stage scripts take it.
- `tools/bench_residency.py`: LRU window vs planned residency, plain and speculative, on a real checkpoint.
- `tools/bench_cluster_locality.py`: a tiny-model training A/B of the constraint, reporting held-out CE, cluster entropies, topic purity and lazy-decode cluster reads.

### Measured (reports/residency_20261107/)
- 50M s13996 checkpoint, 16 FLORES+ dev prompts, greedy, CPU, 12 hot / 12 RAM pages.
  - Plain decoding: planned residency needs 4438 synchronous SSD loads instead of 6876 (−35%). Throughput is 22.47 vs 23.17 tok/s.
  - Speculative decoding: 6258 vs 9297 loads (−33%), and 18.96 vs 16.03 tok/s.
  - All outputs are identical to plain greedy (16/16).
  - The checkpoint was trained without the constraint and routes near-uniformly over all 36 experts within a stream.
- Tiny model on synthetic 4-topic streams, 300 steps, λ=0.3 vs 0.
  - Window cluster entropy falls from 0.49 to 0.005; topic→cluster purity rises from 0.81 to 0.999.
  - Marginal cluster entropy rises from 0.71 to 0.99, and expert-usage entropy is unchanged (0.87).
  - Held-out CE is 1.712 vs 1.703.
  - Cluster-paged lazy decoding needs 0.023 vs 1.48 cluster reads per token.

### Fixed
- MLX lazy store: an expert evicted by the prefetch thread between an SSD read and promotion raised "could not be loaded on demand". Reads now return the page they loaded.

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
