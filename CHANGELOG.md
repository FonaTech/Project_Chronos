# Changelog

## [0.3.3] — 2026-11-14 (stable)

Stable release. No kernel code changes beyond 0.3.3a1; this release bumps the version string and ships the final documentation set.

### Added
- Integrated kernel documentation. `docs/KERNEL_ARCHITECTURE.md` is the new top-level walk-through of the forward stack, block interior, cluster-locality objective, expert-residency mechanism, recurrence and self-speculative decoding. Four code-generated architecture figures under `docs/assets/arch/` carry the schematics, with a reproducible generator (`docs/assets/arch/make_arch.py`) and symbol table (`architecture_config.json`).
- Per-component chapters: `MOE_ROUTING.md`, `CLUSTER_LOCALITY.md`, `RESIDENCY.md`, `LAZY_EXPERTS.md`, `LOOKAHEAD.md`, `LOOP_DEPTH.md`, `LONG_CONTEXT.md`, `SPEC_DECODE.md`, with source-code line refs, mermaid mechanism diagrams and measured numbers traced to recorded JSONs under `experiments/figures/source_data/`.
- Data figures. 18 redesigned matplotlib panels replace the earlier exports under `docs/assets/`; each panel has a `_outlined.svg` with text converted to paths. Generator and snapshotted source data live under `experiments/figures/` with a `verify_values.py` numeric audit (87 checks, 880 numeric entries; exact except for a 1e-12 tolerance on the sigmoid inverse).
- Evidence ledger (`docs/REPRODUCIBILITY.md`) and symbol table (`docs/SYMBOLS.md`); a Chinese reading guide (`docs/READING_GUIDE_ZH.md`).

### Changed
- `docs/README.md` rewritten with a reading-route diagram, a kernel-interior gallery, and an at-a-glance measured-evidence row.
- `docs/OVERVIEW.md` and `docs/THEORY.md` redirect legacy `RECURRENCE.md`/`ROADMAP.md` references into the new kernel and loop-depth chapters.
- `docs/KERNEL_ARCHITECTURE.md` corrects the block-order vs layer-order claim: the sign depends on the per-group step-size schedule (sigmoid `dt` → block-order wins by 0.03 nats; clamp `dt = 1` → layer-order wins by 0.29 nats on the 15 M depth-4 arms).

### Fixed
- Documentation cross-reference graph. Every `.md` link resolves, every asset path exists, every source-code line reference points at a real file in the tree.

### Not changed
- Kernel code, trainer, planner, auto-tune, Web UI, serving adapter. Existing configs and checkpoints behave as in 0.3.3a1.

## [0.3.3a1] — 2026-11-12 (beta)

This release closes three gaps in 0.3.3a0. Every loop knob now has a planner recommendation and an auto-tune switch. The auto-tune probe now scores what it was meant to score. The en-zh model folder has left the repository root. Kernel defaults are unchanged, so existing configs and checkpoints behave as in 0.3.3a0.

### Added
- Planner: a looped plan (`recurrent_depth > 1`) now also sets six more loop knobs. Each comes with a `sources` entry that states the measurement behind it (15M parameters, one seed per arm). The Web UI shows these values as planner recommendations, and `ui/presets_v3.json` gains the four keys whose value differs from the base defaults.
  - `recurrent_later_topk=0`: routing later iterations to 1 expert raised validation CE from 4.543 to 4.600 at the same speed (71.6k against 71.5k tokens/s).
  - `recurrent_later_groups=0`: the full block has the lowest loss per step. Running 2 of 4 layers in later iterations gives about 1.3x more steps per second at depth 4. It costs up to 0.012 more training loss per step with dt fixed at 1.0 and 0.013-0.055 with the sigmoid step size. 1 is never recommended, because one layer looping alone is no longer a block loop.
  - `lambda_iteration_cluster=0`: with cluster locality on, 99.5-100% of the clusters that later iterations use are already used at iteration 0.
  - `lookahead_teacher_scope="first"`: the lookahead head already picks the right cluster for 0.97 of tokens at every iteration.
  - `recurrent_dt_min=0.05` and `recurrent_dt_max=1.0`: the sigmoid arms used the ranges 0.05-1 (depth 4) and 0.1-1 (depth 8), and their learned dt stayed between 0.14 and 0.30, inside the range.
- Auto-tune can now search every loop knob. The list adds `recurrent_dt_param`, `recurrent_dt_min`, `recurrent_dt_max`, `recurrent_later_topk` (0 or 1), `recurrent_later_groups` (0 or 2), `lambda_iteration_cluster` (0, 0.1, 0.3 or 1.0) and `lookahead_teacher_scope`. They join `recurrent_order` and `recurrent_dt_init`.
  - Searching any of them makes the probe a looped model with 4 block layers, one prelude layer and one coda layer. The 0.3.3a0 probe had 2 block layers, so `recurrent_later_groups=2` would have meant the full block.
  - `recurrent_later_groups` is drawn only in block order, and 1 is not offered.
  - If a drawn `recurrent_dt_init` falls outside the drawn `[dt_min, dt_max]`, the trial starts at the nearest bound.
  - Each trial logs its median step time and stores it as the Optuna user attribute `step_time_s`. `recurrent_later_topk` and `recurrent_later_groups` save time, not loss, so a cross-entropy score alone favours "off".
- Help text for these knobs and for the auto-tune search list now gives the measured reason and the recommended value, in English, 简体中文, 繁體中文 and 日本語.

### Fixed
- Auto-tune score. The probe trained on and scored `CE + lambda_balance * L_balance + lambda_temporal * L_temporal`, so a smaller weight lowered a trial's score by itself. The probe now trains on the trainers' objective (`chronos_loss_term`, every weighted term of the config) and scores cross-entropy only.
- Auto-tune `lambda_lookahead` had no effect. The probe never passed it to the model config or to the loss, and it never trained the lookahead head, so `lookahead_steps` changed the model without supervising it. With the 0.3.3a0 probe, `lambda_lookahead` 1e-3 and 1.0 gave identical per-step scores; with this release the scores differ after the first update. The lookahead, iteration-cluster and recurrent terms of the config now act in the probe as they do in training.
- Web UI schema payload. `Field.as_dict` dropped every value equal to `False`, and in Python `0 == False`. So `min=0` of 48 fields, and `recommended=0` of `max_steps` and `cluster_locality_clusters`, never reached the page. Those number inputs had no lower bound, and their tooltips showed the minimum as "−∞". The payload now drops only unset values.

### Changed
- The en-zh model folder moved from `models/chronos-enzh-50m-sft` to `demos/translation/models/chronos-enzh-50m-sft`, and its driver test moved from `tests/` to `demos/translation/tests/`. CI runs that test in its own step after the kernel suite, and the repository root no longer holds a task-specific model. The driver finds its files relative to itself, so commands run inside the folder are unchanged, and the weights stay on the release `enzh-50m-sft-v1`. `MANIFEST.in` no longer prunes `models/`.
- The main READMEs now describe the speculative-decoding, native-kernel parity and MLX compile measurements as taken on the 50M demo checkpoint. They link to a new section of `demos/translation/README.md` and `README_zh.md` that states the exact conditions (FLORES+ dev en→zh prompts, en→zh shards); the numbers are unchanged. Each main README names the translation demo in one pointer paragraph only. `tests/test_ui_generic.py::test_task_models_and_tests_live_under_demos` checks this rule and the folder layout.
- The preset label "Chronos-v3 50M (16k vocab)" stays. It names the preset's explicit 16,384-token vocabulary and no task.

### Verified
- End to end through the Web UI endpoint, with an in-process client and no open port: a 4-trial study of 10 probe steps each searched all nine loop knobs plus `lambda_lookahead`. It trained on 19,090 real dataset tokens with the bundled 32k tokenizer, completed every trial, logged each trial's cross-entropy and median step time, and returned a best-config patch with every searched knob.

### Tests
- New `tests/test_loop_knobs_wiring.py` (10 items). It checks that every loop knob is searchable and reaches the probe model, and that `recurrent_later_groups` never offers 1 and is drawn only in block order. It checks that `lambda_lookahead` 1e-3 and 1.0 give the same score at step 0 and different scores after the first update. It also covers the best-config patch, the Web UI search list with help in four languages, planner recommendations with a source at depths 2, 4 and 8, a depth-1 plan with no loop knob, and a schema payload that keeps 0.
- New `tests/test_ui_generic.py::test_task_models_and_tests_live_under_demos`: `models/` and `tests/` hold no task-specific names, and each main README names the en-zh demo in one line, which points to `demos/translation`.
- `tests/test_autotune_loop.py`: `recurrent_later_groups` is now expected in the search list.
- `demos/translation/tests/test_model_enzh_50m_sft.py` (15 items, moved from `tests/`) passes from its new folder; CI runs it as the step "Demo driver tests".
- Full suite on macOS (Apple silicon, Python 3.11, MLX on CPU): 728 passed, 4 skipped. `ruff check .` passes on the whole tree.
- A local wheel and sdist build both carry the `general_32k` tokenizer and no file from `demos/`, no model folder and no weights.

## [0.3.3a0] — 2026-11-12 (beta)

### Added
- `recurrent_order` (`"layer"` default, `"block"`): the order in which the recurrent layers repeat. `layer` runs each recurrent layer `recurrent_depth` times in a row (A A B B); `block` runs the whole group of recurrent layers and then repeats it (A B A B), so iteration 0 is an ordinary depth-1 forward pass and later iterations refine it. In `block` order there is one state trajectory and one halting head per block iteration (the `halting_probs` and `depth_used` group axis has length 1); every physical layer keeps its own `dt`, loop and group embeddings and its own KV slot. The default path is bit-identical to 0.3.2a3 (Torch and MLX logits compared on four attention modes, max difference 0). Available in the Torch and MLX models, the web UI (four languages) and `MODEL_FIELDS`. A `block` checkpoint is not interchangeable with a `layer` checkpoint.
- `recurrent_later_topk` (int, default 0 = off): loop iterations after the first route each token to this many experts (capped at `num_experts_per_tok`); the shared expert still runs. With `router_scope="shared"` the kept experts are the best-scoring ones of the first dispatch, so no new expert page is touched. The count is an explicit argument of the MoE, block and recompute calls in Torch and MLX, so activation recomputation replays it. The default path is bit-identical (max difference 0 on both backends, four attention modes).
- `recurrent_later_groups` (int, default 0 = all): in `block` order, loop iterations after the first run only the first N layers of the block. Cost in this kernel scales with the number of layer calls, so this removes calls and the expert pages those layers would load. Iteration 0 still runs every layer and writes every KV slot. It needs `recurrent_attention_mode` `shared_first` or `first_only`, because a skipped layer has no per-iteration KV slot, and the config raises a `ValueError` otherwise. The default path is bit-identical.
- `lambda_iteration_cluster` (float, default 0) and `lookahead_teacher_scope` (`"first"` default, `"iterations"`): the first adds a total-variation penalty between a token's per-iteration cluster-mass distribution and its mean over iterations, so the loop layers keep a token in one expert cluster across depth without locking experts. The second makes the lookahead router's teacher the router probabilities averaged over the layer's iterations. Both exist in Torch and MLX; the MLX loss is checked against the Torch loss and against `mx.compile`.
- `tools/bench_residency.py --paging expert|cluster|auto` (default `expert`, the historical layout) and `tools/bench_speculative.py` `mlx_runner(..., paging=...)`: the lazy cache can be paged by training cluster, as production serving does for checkpoints trained with `lambda_cluster_locality > 0`. Each row now also records `cluster_experts_loaded`, `cluster_aware`, `num_clusters`, `warmup_cluster_loads` and `prefill_cluster_loads`, and the summary `decode_cluster_loads`.
- `recurrent_dt_param` (`"clamp"` default, `"sigmoid"`): how the learned loop step size is stored. `clamp` keeps `dt = clamp(w, dt_min, dt_max)`, whose gradient is zero once `w` leaves the interval. `sigmoid` uses `dt = dt_min + (dt_max - dt_min) * sigmoid(w)`, whose gradient is positive everywhere, so a step size that has become small can still grow. The stored logit is initialised so that step 0 gives exactly `recurrent_dt_init` (clamp and sigmoid logits agree to 1.1e-6 at init; Torch and MLX agree to 1.3e-6). The default path is unchanged.
- `chronos.model.recurrent.recommended_dt_init(depth, horizon=2.0, cap=0.5)` returns `min(cap, horizon / depth)`: 0.5 at depth 4, 0.333 at depth 6, 0.25 at depth 8. One loop iteration is an explicit Euler step `x <- x + dt * f(x)`, so `depth` iterations integrate over the horizon `T = depth * dt`; the rule holds `T` fixed and lets depth refine the integration. It is a starting value for `recurrent_dt_init`, not a fitted optimum (see Physical mechanism and Measured). An earlier draft used `0.5 * (4 / depth) ** 3`; the depth-8 run below refuted it and it was removed before release.
- `recurrent_dt_param`, `recurrent_dt_init`, `recurrent_dt_min` and `recurrent_dt_max` appear in the Web UI recurrent section with labels and help in four languages. The planner recommends block order, `sigmoid` and the horizon-based `recurrent_dt_init` whenever `recurrent_depth > 1` (config, `sources` report entries, and the schema's recommended value).
- Auto-tune: `tune_recurrent_order` and `tune_recurrent_dt_init` join the search space (Web UI "Search over" list, `ChronosSearchSpaceConfig`, best-config patch). When either is searched the probe model runs `probe_recurrent_depth` iterations (the Web UI passes the configured `recurrent_depth`, clamped to 2-8) with one prelude and one coda layer. `recurrent_later_groups` is deliberately not searched: it trades quality for cost, so a loss-only search would always choose the full block. The probe loss comes from a few steps and cannot resolve differences as small as dt 0.2-0.5 at depth 4; treat the two loop knobs as a coarse screen.
- Auto-tune probe data: `chronos/tuning/probe_data.py` encodes a bounded sample of the user's dataset (at most 400k tokens and 20k records) with the configured tokenizer and trains the probe on it. The probe used to train on random token ids, so no hyperparameter could change its loss in a way that carries over to real data. It falls back to random ids, with a log line, when no text can be read or the checkpoint's vocabulary is smaller than the tokenizer's. On Apple silicon the probe now picks a torch device itself when the training backend is `mlx`, which is not a torch device.
- `chronos/assets/tokenizers/general_32k` (32,768 tokens, 1.9 MB) ships in the wheel, with `chronos.deps.get_general_tokenizer_path()` and `GENERAL_TOKENIZER_VOCAB`. It is the vocabulary built for the pretraining mix, with its control tokens relabelled to the kernel's task-neutral layout (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|pad|>`, then reserved ids 4-63); ordinary tokens, merges and ids are unchanged (1,004 of 1,004 test texts encode to the same ids as with the original tokenizer, on the native and on the Hugging Face tokenizer). Because the fingerprint hashes the special tokens, relabelling changed it from `169668b1…` to `6d4b3b52…`; the old value is recorded under `derived_from`, so a checkpoint trained with the original relabel-free tokenizer reports a different fingerprint than this asset. It carries no task tokens. `NOTICE` lists the corpus licences: FineWeb-Edu and Cosmopedia v2 (ODC-By), Wikipedia (CC-BY-SA 3.0 and GFDL), and Fineweb-Edu-Chinese V2.1, whose commercial use needs the owner's permission.
- Planner default vocabulary: a new plan whose request and dataset profile name no vocabulary now uses 32,768 and writes `tokenizer_path` pointing at the bundled tokenizer, with `sources` entries (it was 16,384 with no path). An explicit `vocab_size` keeps the user's choice and adds no path. `get_tokenizer_path()` still returns the legacy 6,400 tokenizer, so checkpoints that never recorded a tokenizer are not changed.
- Web UI: `recurrent_later_topk` and `recurrent_later_groups` appear in the recurrent section, `lambda_iteration_cluster` and `lookahead_teacher_scope` in the objectives section, each with labels and help in English, 简体中文, 繁體中文 and 日本語.

### Fixed
- MLX trainer: a recurrent step size (`recurrent_dt_*`) that sat exactly on `recurrent_dt_min` or `recurrent_dt_max` received no gradient, because `mx.clip` has gradient 0 on its bounds while the Torch trainer's `clamp` gives 1. A run with `recurrent_dt_init` equal to a bound therefore kept its step sizes frozen on MLX only (a 30-minute arm with init 1.0 = max ended with raw step sizes exactly 1.0). `RecurrentControllerMLX.contract` now returns the same value with the Torch gradient on the bounds. Values strictly inside the interval, and the forward pass, are unchanged, so default configs train as before; values strictly outside the interval still get no gradient on either backend.
- MLX trainer: with `recurrent_depth_jitter > 0` and gradient accumulation, each micro-batch drew its own loop depth, so one optimizer step mixed graphs of different depth and the router statistics accumulated across micro-batches failed with a shape error (`broadcast_shapes (14,24) and (10,24)`). The depth is now drawn once per optimizer step and shared by all its micro-batches, in eager and compiled steps. Runs without jitter are unchanged.

### Changed
- `tools/bench_residency.py` `cluster_loads` no longer includes the page reads of the store reset's warm-up (they are reported separately as `warmup_cluster_loads`); earlier reports included them, because the store's counters are cleared before the warm-up. Every mode starts from the same cold warm-up, so absolute counts drop by that constant per (prompt, mode).
- The kernel is now task-neutral in code and text. `chronos/cli.py`, `chronos/mlx/training/longrun.py`, `chronos/mlx/training/compiled.py`, `chronos/tokenizer/extract.py`, `chronos/data/chat_records.py` and the examples in `ui/i18n.py` no longer name a translation task, and the tokenizer asset's provenance metadata no longer names task tokens. `tests/test_ui_generic.py::test_kernel_has_no_translation_task_code` walks `chronos/` and `ui/` and fails on any task-specific term. The en-zh translation work stays in `demos/translation` (`chronos-translate`), which imports the kernel; its status, released-model and bundle sections moved verbatim from `README.md` and `README_zh.md` into `demos/translation/README.md` and `demos/translation/README_zh.md`. The main READMEs keep one pointer sentence and still quote en-zh numbers where they are the measurement context (speculative decoding, the 0.3.2 native-kernel benchmarks, the 50M preset row); those are measurements on one checkpoint, not defaults. `models/chronos-enzh-50m-sft` stays in the repository as the demo's model.
- Planner and Web UI recommendations for a looped model changed (`recurrent_order="block"`, `recurrent_dt_param="sigmoid"`, `recurrent_dt_init = min(0.5, 2 / depth)`), and `ui/presets_v3.json` was regenerated: all five tiers gain those three keys and nothing else changed. The kernel defaults did not change (`recurrent_order="layer"`, `recurrent_dt_param="clamp"`, `recurrent_dt_init=0.25`), so a config that does not set them, and every existing checkpoint, behaves as in 0.3.2a3.

### Physical mechanism
A shared block applied again and again to the residual stream acts as one operator, so each iteration adds a vector that points along nearly the same direction as the one before. The contributions add constructively instead of as independent, orthogonal increments. Measured on the trained checkpoints, the cosine between the outputs of consecutive block calls rises with iteration, for example from 0.92 to 0.985 (depth 8, dt 0.25) and from 0.98 to 0.996 (depth 8, dt 0.5); independent layers would give a cosine near 0.
- One iteration is `h <- h + dt * (block(h + e_loop) - h)`. Near the dominant direction the block update is about `lambda_eff * h`, so each iteration multiplies the residual norm by `g = 1 + dt * lambda_eff`. `lambda_eff` is learned, not set: the optimiser rescales the block's output when `dt` changes. Across step sizes of 0.146 (learned by a sigmoid arm), 0.25 and 0.5 (both fixed) at depth 8, `g` stayed between 1.067 and 1.075 while `lambda_eff = (g - 1) / dt` fell from 0.48 to 0.30 to 0.13.
- The `R - 1` iterations after iteration 0 each apply the gain, so depth `R` raises it to the power `R - 1`: total amplification `g^(R-1)`, total log-gain `(R - 1) * ln g`. In every arm whose step size could stay large that total was 0.36-0.51 (1.4-1.7x) at depth 4 and at depth 8 (depth 4: 0.36-0.44; depth 8: 0.38-0.51). Arms whose step size collapsed onto its floor or stayed tiny had a lower total (0.09-0.24); at equal step they were worse than the best arm of their depth, but not always worse than every arm with a larger total (see the limits below).
- `dt` alone is therefore not the physical quantity; `kappa = dt * lambda_eff` and the total log-gain are. This is why the loss is flat for `dt` between about 0.2 and 0.5 at depth 4, and why the horizon `T = depth * dt` (the length of the Euler integration of `dh/dtau = block(h) - h`) is a better handle than `dt`: more depth at fixed `T` refines the integration without raising the total gain.
- Limits of the claim: the total gain is not a loss predictor (the depth-8 arm with a collapsed step size, total 1.27x, beat the fixed-0.5 arm, 1.57x, at equal step by 0.02-0.05), the measurements are one seed per arm at 15M parameters, and `T = 2` is a safe starting point inside a flat region, not a fitted optimum: the sigmoid arm that started at `T = 2` settled at `T = 1.16`.

### Measured
15M-parameter config, recurrent depth 4, 24 experts with top-2, one Mac, 30 minutes per arm (about 1,850-1,960 steps), one seed per arm unless stated, same data order in every arm. Paired differences are training loss in 10-step windows; a repeat of the same block-order arm differs from the first by +0.002 to +0.008, so differences below about 0.01 are not separable from run-to-run noise.
- Block against layer order at the default step size (init 0.25, clip [0.05, 1]): block order has the lower paired loss by 0.025-0.075 in every block of steps after step 100, at equal speed (about 71.5k tokens/s). Validation CE at 30 minutes: 4.543 (block, 1,960 steps) against 4.586 (layer, 1,855 steps).
- The default step sizes collapse: the learned values end at 0.045-0.09 against a floor of 0.05, and the gradient below the floor is exactly zero on both backends, so a collapsed step size never recovers. Fixing every step size at 1.0 raises the paired loss of block order by 0.050 in steps 10-100, then lowers it by 0.022, 0.025, 0.101, 0.072 and 0.026 in the blocks 100-300, 300-600, 600-1,000, 1,000-1,500 and 1,500-1,960 (largest at steps 600-1,000). Validation CE at 30 minutes is slightly higher, 4.559 at 1,850 steps against 4.543 at 1,960, because the fixed-step arm completed fewer steps.
- With step sizes fixed at 1.0, layer order has the lower loss after about step 600 (block minus layer +0.015 to +0.034) and block order only learns faster during the first 600 steps. The advantage of block order therefore holds at the default, collapsing step size only; the default is not changed in this release.
- Depth 8 does not follow the depth-4 result. With step sizes fixed at 1.0, block order at depth 8 (1,255 steps in 30 minutes, 1.43 s per step, final validation CE 4.821) has a higher paired loss than the default-step-size depth-8 arm in every one of 82 windows (+0.199, +0.075, +0.039 and +0.077 in steps 10-100, 100-300, 300-600 and 600-1,000), and its gradient norm is 1.4-1.8 times higher (mean 1.27 in the first 100 steps, so the 1.0 clip was active there). Against fixed-step depth 4 it has no equal-step gain (validation CE 5.328 against 5.295 at step 800, 4.794 against 4.773 at step 1,250) at 1.5 times the time per step. The default step-size rule is therefore left as it is.
- Truncating the fixed-step depth-8 model to 1 to 8 iterations gives validation CE 5.615, 5.201, 4.995, 4.884, 4.823, 4.794, 4.779 and 4.773, so iterations 5 to 8 add 0.11 nats.
- Cost of the loop: a depth-4 step takes 0.945-0.976 s against 0.569 s at depth 1 (1.7x), for roughly 5-13% fewer steps to reach the same validation CE, so the loop loses on wall time at this size. Cost follows the number of layer calls, not the number of experts or tokens.
- Where the gain sits (validation CE with the loop truncated to 1, 2, 3, 4 iterations): 4.977, 4.637, 4.527, 4.506. The top-1 token equals the full-depth top-1 for 52%, 71%, 86% and 100% of tokens. The fourth iteration is worth about 0.02 nats.
- `recurrent_later_groups=2` (block order, step sizes fixed at 1.0, 2 of the 4 block layers in iterations 2-4): 2,450 steps in 30 minutes against 1,850 for the full block (0.748 against 1.001 s per step, 89.2k against 68.4k tokens/s, peak memory 6.57 against 7.43 GB) and a final validation CE of 4.456 against 4.559. At equal step the paired loss differs from the full block by -0.005, +0.007, -0.019, +0.001, +0.005 and +0.012 across the six blocks of steps, so the saving costs at most about 0.01 per step (the noise floor is 0.005-0.008). One seed. A depth-1 model trained for the same 30 minutes still reaches 4.379 in 3,170 steps, so the loop does not yet pay for itself in wall time at this size. Truncated to 1, 2, 3 and 4 iterations this model gives validation CE 4.624, 4.494, 4.439 and 4.418, against 4.987, 4.659, 4.542 and 4.514 for the full block: its first iteration is much better and the loop adds 0.206 nats in total.
- `recurrent_later_groups=2` at depth 8 (step sizes fixed at 1.0): 1,825 steps in 30 minutes against 1,255 for the full block (1.004 against 1.458 s per step) and a final validation CE of 4.586 against 4.821. At equal step the paired loss is higher by 0.007, 0.029, 0.029 and 0.022 after step 100, and validation CE at step 1,250 is 4.809 against 4.794, so skipping half the layers in seven later iterations costs about 0.02-0.03 per step, more than at depth 4. Depth 4 with the same setting is better at every point (validation CE 4.775 against 4.809 at step 1,250, 4.456 against 4.586 at 30 minutes).
- `recurrent_later_topk=1` raised validation CE from 4.543 to 4.600 and was no faster (71.6k against 71.5k tokens/s): at this width fewer routed experts saves no time.
- Cluster level: with `lambda_cluster_locality=0.3` a token touches 1.003-1.024 of the 6 expert clusters in the first iteration, and 99.5-100% of the clusters touched in the later three iterations were already touched in the first. `lambda_iteration_cluster` therefore has almost nothing to act on; with step sizes fixed at 1.0 its arm had paired loss lower by 0.007, 0.000, 0.005, 0.024, 0.018 and 0.008 across the six blocks of steps (one seed; all but the 600-1,500 blocks are within the noise floor), and the top-2 overlap with the first iteration differed by at most 0.011.
- Lookahead head: its next-token prediction recovers the top-2 experts of loop iterations 1-4 with recall 0.68-0.78 (top-2 slots) and 0.97-0.98 (top-4 slots), and picks the right cluster for 0.97 of tokens at every iteration, the same as repeating the previous token's cluster. An iteration-aware head was not built.
- Step size at depth 4 (block order, fixed step size, one seed per arm): fixing `dt = 0.5` against `dt = 1.0` lowers the paired loss by 0.052, 0.016, 0.032, 0.066, 0.047 and 0.026 across the six blocks of steps (every window lower in every block), and validation CE is 4.505 at 1,975 steps against 4.559 at 1,850. A sigmoid arm started at 0.5 (range 0.05-1) moved to 0.182-0.203 in all four layers with no change in loss: its paired difference from the fixed-0.5 arm is +0.002, -0.003, -0.002, +0.007, +0.001 and -0.002, inside the noise floor. At depth 4 the loss is therefore flat for `dt` between about 0.2 and 0.5, and worse at 1.0 or at a `dt` stuck at 0.05.
- Block against layer order with a healthy step size (both arms sigmoid, init 0.5, range 0.05-1): block order has the lower paired loss in every window block, by 0.051, 0.093, 0.081, 0.053, 0.039 and 0.027 (layer minus block; 0 of 20, 0 of 30, 0 of 40, 0 of 50 and 0 of 27 windows lower for layer order after step 100). Validation CE at 30 minutes is 4.542 for block (1,830 steps) and 4.572 for layer (1,775 steps). This supersedes the reading above that block order is only better at a collapsing step size: with step sizes fixed at 1.0 layer order wins late, with a trainable step size block order wins in every block of steps measured (in steps 10-100, 7 of 9 windows).
- Depth 8 with a fixed integration horizon (block order, one seed per arm, 30 minutes, 1,185-1,265 steps, same data order): `dt = 0.25` (`T = depth * dt = 2`) has a lower paired loss than `dt = 0.5` (`T = 4`) by 0.046, 0.018, 0.031, 0.079 and 0.075 across the five blocks of steps from step 10 to the end of the run (every window lower in every block); validation CE is 4.776 at 1,215 steps against 4.794 at 1,265. Against the default clamp arm, whose step size collapsed onto the 0.05 floor, it is lower by 0.003, 0.013, 0.015 and 0.002 (about the noise floor). A sigmoid arm started at 0.25 (range 0.1-1) moved to 0.141-0.147 (`T` about 1.16), tied with the fixed arm until step 600 (+0.002 in three blocks) and then lower by 0.023 and 0.020 (58 of 61 windows), with validation CE 4.761 at 1,220 steps against 4.776 at 1,215. At equal step the depth-8 `dt = 0.25` arm is lower than the depth-4 `dt = 0.5` arm by 0.041 and 0.049 in steps 100-300 and 300-600 and level or slightly higher afterwards (+0.002 and +0.008, at the noise floor).
- The cubic depth rule `dt0 = 0.5 * (4 / depth)^3` was an earlier draft and is refuted at depth 8: a sigmoid arm started at its prediction 0.0625 and ended at 0.031-0.037, and its paired loss is higher than the fixed `dt = 0.5` arm by 0.066, 0.079, 0.122 and 0.256 after step 100, and higher than the fixed `dt = 0.25` arm by 0.085, 0.110, 0.201 and 0.333. Its total log-gain was 0.088, the lowest of any arm. The rule was replaced by `T / depth`.
- Progressive loop curriculum (`recurrent_depth` rising 1 to 4 over 1,600 steps, fixed `dt = 0.5`, depth jitter 0.01, block order): median time per step is 0.56, 0.725, 0.899 and 1.08 s at depth 1, 2, 3 and 4. At equal step it trails the fixed-depth-4 arm until the depth reaches 3-4: validation CE 5.596 against 5.215 at step 800 and 5.168 against 4.920 at step 1,000. Its early steps are cheaper, and it gives up equal-step quality until the full depth is reached; the loop's advantage is equal-token sample efficiency, not wall clock.
- `recurrent_later_groups=1` leaves three of the four block layers dormant in every later iteration, so the loop reduces to one layer applied repeatedly. With step sizes fixed at 1.0 its paired loss is higher than `recurrent_later_groups=2` by 0.062 and 0.050 in steps 100-300 and 300-600 and by 0.004, 0.004 and 0.012 afterwards, and the two arms end 0.049 apart in validation CE at 30 minutes (4.407 at 2,810 steps against 4.456 at 2,450) only because 1 layer is cheaper. Use `recurrent_later_groups` of at least 2.
- `recurrent_later_groups=2` with the default clamp step size (block order, init 0.25, clip [0.05, 1]) was much worse than with a step size that can move. Against the full-block arm with the same default step size (block order, a repeat run), it trailed by +0.007, +0.017, +0.030, +0.046, +0.200 and +0.097 across the six blocks of steps (about +0.235 in steps 1,200-1,400), at a median gradient norm of 0.54 against 0.51 after step 1,000 and the same learning rate; it had no rollbacks. The learned step sizes ended at 0.056, 0.050, 0.133 and 0.139, so the two layers that still run in later iterations scaled their updates by 0.056 and 0.050. With a sigmoid step size from 0.5 (range 0.05-1) and the same `recurrent_later_groups=2`, the paired loss is lower than the clamp arm by 0.037, 0.067, 0.150, 0.284 and 0.128 after step 100, validation CE at step 1,000 is 4.980 against 5.254, and the two later-iteration layers end at 0.185 and 0.170. The step-size collapse therefore accounts for most of the regression but not all of it. The prediction that the sigmoid arm would stay within about 0.01 of the full block was not met: it is above the full-block sigmoid arm by 0.013, 0.021, 0.022, 0.055, 0.034 and 0.019 (2 of 9, 0 of 20, 2 of 30, 0 of 40, 1 of 50 and 2 of 33 windows lower), more than the 0.01 that skipping layers cost with the step size fixed at 1.0. The two arms also differ in initial step size (0.5 against 0.25), so the effect of the parameterisation and of the initial value are not separated, and the loss gap that opens at steps 1,000-1,500 and then closes is not explained. At 30 minutes the sigmoid arm reaches validation CE 4.440 in 2,365 steps, against 4.491 in 2,460 for the clamp arm and 4.542 in 1,830 for the full-block sigmoid arm; at equal step the full block is ahead (4.928 against 4.980 at step 1,000).
- Step time of this release's kernel: the same depth-4 configuration takes a median 0.997 s per step with the clamp and 1.017 s with the sigmoid (steps 50-400, one 6-minute arm each), against 0.958 s on the development kernel that had block order and `recurrent_later_groups` but not the clip-gradient fix. The 4% and 6% are not attributed to a cause (the differences are the `where`-based clip gradient, the per-step depth draw and the sigmoid branch), and one run each cannot separate them from run-to-run noise.

### Known limits
- Block order with lazy experts: `tests/test_recurrent_order.py::test_mlx_engine_block_order_runs_with_lazy_experts` checks that the MLX engine with a two-expert resident set generates 10 valid tokens speculatively, and the all-resident engine is checked for exact speculative decoding. Nothing compares lazy block-order output with all-resident block-order output token by token, and the residency planner, the cluster-paged store and the async prefetcher have not been benchmarked on a block-order checkpoint (`expert_store.py` and `cluster_layout.py` index pages by physical layer, which is the same page set in both orders; only the order in which layers are used differs). Treat lazy serving of a block-order checkpoint as unmeasured.
- All loop measurements are one seed per arm at 15M parameters (block-order depth 4 and 8, 24 experts, 30 minutes). They are not evidence for larger models, and the planner's block-order and `T = 2` recommendations are extrapolated to every size it plans.
- The Web UI probe for auto-tune sees a few steps of a 4-layer model; it cannot rank step sizes that differ by less than the measured noise.

### Tests
- `tests/test_recurrent_order.py`, `tests/test_later_topk.py`, `tests/test_later_groups.py` and `tests/test_iteration_cluster.py` cover config validation, the `MODEL_FIELDS` round trip, Torch against MLX, cache decode against a full forward, exact self-speculative decoding, and gradient reach into every recurrent layer, router and the controller. `tests/test_mlx_checkpointing.py` adds block-order and graded-top-k cases to the recompute and compile matrices.
- `tests/test_dt_clip.py` compares the MLX and Torch step-size gradient at 0.4999, 0.5, 0.5001, 0.75, 0.9999, 1.0 and 1.0001; on the 0.3.2a3 kernel it fails at 0.5 and 1.0.
- `tests/test_bench_paging.py` checks that `mlx_runner(paging=...)` builds one page per expert for `expert` and the training clusters for `cluster` and `auto`. The prefill/decode split and warm-up exclusion in `tools/bench_residency.py` run inside its `main()` and need a tokenizer, so they are verified by running the tool on the benchmark checkpoints, not by a unit test.
- `tests/test_dt_sigmoid.py` covers config validation of `recurrent_dt_param`, the sigmoid logit initialisation (step 0 gives `recurrent_dt_init` at 0.05, 0.0625, 0.25, 0.5 and 1.0), the nonzero sigmoid gradient where the clamp gradient is zero, Torch against MLX at four logits, and `recommended_dt_init` (`depth * dt = 2`, the 0.5 cap at depths 1-3). `tests/test_depth_jitter.py` covers one depth draw per optimizer step under gradient accumulation, with and without `mx.compile`.
- `tests/test_general_tokenizer.py` (7 tests) checks the bundled tokenizer's files, its 32,768 vocabulary and control ids 0-63, native against Hugging Face encodings and the round trip, that the report names no local path and records `derived_from`, the planner's new default (32,768 and the bundled path, with `sources`), that an explicit vocabulary adds no path, and that the legacy default is still 6,400.
- `tests/test_autotune_loop.py` (8 tests) covers the probe's real tokens (bundled vocabulary, `None` for unreadable or tiny data, deterministic batches that advance with the step), the loop knobs being sampled only when searched, the best-config patch, a probe trial that builds a looped block-order model and draws its batches from real tokens, a probe without loop knobs staying a plain depth-1 model on random ids, and the Web UI search choices and four-language labels. `tests/test_planner_loop.py` (7 tests) covers the planner's block order, sigmoid and horizon init at depths 2, 4, 6 and 8, the plain defaults at depth 1, that the planned config builds a valid model, and that the schema's recommended values equal what the planner emits.
- Full suite on the development tree: 732 passed, 4 skipped (three browser tests need `playwright`, which is not installed here; one needs the v2 reference checkpoint, which is not in the repository). `ruff check .` is clean.

## [0.3.2a3] — 2026-11-10 (beta)

Fixes two kernel bugs found while benchmarking the 50M and 30M multilingual configs on the real corpus. Nothing new is switched on by default.

### Fixed
- `cluster_locality_mode="segment"` could not be compiled. The loss found document ids with NumPy (`segment_ids`, `np.unique`) on a traced array, so `CHRONOS_MLX_COMPILE=on` raised "Attempting to eval an array during function transformations", and the default `auto` mode crashed on the next `mx.eval`. The loss now uses `segment_ids_mx(input_ids, chunk=window)`, which has a static shape (`row * S + id` addresses `B * S` segment slots and unused slots weigh 0), and runs inside the compiled step. It matches the NumPy ids on 300 random batches and the Torch loss within 1.7e-5 on 60.
- A compile trace that failed part way left tracers in the model weights, the RNG key and the router outputs, because `mx.compile` swaps them back only on success. The eager fallback then trained on them and the next `mx.eval` raised "Attempting to eval an array without a primitive". `CompiledStep` now snapshots those on a first trace and restores them on failure; the RNG is re-seeded from the saved key words, so the random stream continues exactly (checked on 2,000 random keys).

- `run_mlx_stage` and LongRun now check every batch's token ids against the model's `vocab_size` and raise a `ValueError` that names the range. MLX does not bounds-check embedding gathers, so an id beyond the table read neighbouring memory and crashed the process at random. `tests/test_smoke.py::test_mlx_training_six_stage_loss_paths` hit this about one run in two on CPU: its GRPO stage tokenizes with the bundled 6,400-token tokenizer while the model had `vocab_size=64`. The test now uses 6400 (25 of 25 runs pass), and `tests/test_vocab_range_guard.py` covers the check.

### Measured
- Segment mode, compiled, real corpus, 10 steps, document masking on, steady tokens/s: 50M 24,856 at ctx 2048 and 20,416 at 4096; 30M 22,454 and 16,416. The eager fallback cost 15-17%.

### Tests
- `tests/test_segment_compile.py` (5 cases) fails on 0.3.2a2 (4 of 5) and passes here: ids equal the NumPy rule with chunking, loss equals the Torch reference eagerly and compiled, a one-segment batch gives 0, a compiled segment-mode step matches the eager gradients, and a failed trace leaves weights, RNG and side outputs usable.

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
