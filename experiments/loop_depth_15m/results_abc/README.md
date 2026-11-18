# ABC experiments — five 15M arms at 30 min each

Five arms share one data recipe (`chronos_data_30m`), one tokenizer, one
1800-s wall budget. Each arm varies one knob against `base`. Every arm's
`.pth` was probed on the same 12 EN/ZH greedy continuations, 60 tokens
each, no sampling, no penalty.

The shared skeleton is `prelude_layers=1`, `recurrent_groups=4`,
`coda_layers=1`, `recurrent_depth=4` (`B_plain` is depth 1). The `dt sum`
in the last column is the sum of the four per-group `dt` values after
the sigmoid, not a cumulative step across iterations (every iteration
runs all four groups).

## Arms

| arm | depth | attn_mode | aux_losses | valid_ce | steps | wall_s | unique | max_run | max_4gram | dt per group (sum) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| `base` | 4 | shared_first | on | 4.735 | 1290 | 1804 | 0.151 | 13.08 | 18.58 | 0.25/0.22/0.25/0.23 (0.96) |
| `A_dtcap` | 4 | shared_first | on, dt_max=0.5 | 4.768 | 1290 | 1801 | 0.124 | 15.67 | 19.67 | 0.14/0.14/0.14/0.16 (0.58) |
| `B_plain` | 1 | — | on | **4.528** | 2195 | 1803 | 0.150 | 6.42 | 15.33 | — |
| `C_periter` | 4 | **per_iteration** | on | 4.727 | 1235 | 1800 | 0.144 | **3.58** | **10.67** | 0.23/0.23/0.24/0.26 (0.97) |
| `obj_lm` | 4 | shared_first | **off** | **4.533** | 1335 | 1806 | **0.201** | 10.67 | 16.75 | 0.42/0.46/0.56/0.42 (1.86) |

## What each arm tests

- `base` is the 0.3.3 preset at 15M scale.
- `A_dtcap` tightens `recurrent_dt_max` to 0.5. Learned per-group dt
  halves; under-steps the recurrence; worst greedy of the lot.
- `B_plain` drops `recurrent_depth` to 1. 1.7× more steps per second;
  wins by 0.21 nat at matched wall time, loses by 0.15 nat at matched
  steps. Direction: depth=4 is over-budget at this scale.
- `C_periter` swaps `recurrent_attention_mode` to `per_iteration`. CE
  tied with `base`; `max_run` 3.58 vs 13.08 is the cleanest greedy
  output. Costs: KV grows by `recurrent_depth` on the recurrent layers
  only (4× those layers, 3× the whole 6-layer model), inference-time
  depth override is refused, and speculative decoding falls back to
  plain.
- `obj_lm` zeroes every non-structural auxiliary loss (cluster_locality,
  router_locality, temporal, three lookahead terms, iteration_cluster,
  router_anchor, distill, stability, ponder — `balance` kept). 0.20 nat
  CE win over `base` at matched wall time, about 0.17 nat at matched
  step, on all five sources; best `unique_ratio`. The learned per-group
  `dt` is about twice base's ([0.42, 0.46, 0.56, 0.42] vs
  [0.25, 0.22, 0.25, 0.23]). Each group still interpolates (dt < 1), so
  this is not an overshoot. The arm removes all eleven terms at once, so
  it cannot say which term drives the CE gain or the larger `dt`.

## What 0.3.4 took from this

- Decoding-side: `repetition_penalty` kwarg across speculative /
  streaming / adapter / HTTP serving (default 1.0). CTRL rule applied on
  raw logits in both draft `pick` and the greedy verify step, so
  rejection sampling and greedy acceptance both stay consistent.
- `max_inference_depth` is exposed as an explicit `INFERENCE_OVERRIDE`
  key, so a caller can run the recurrent loop at a shallower depth than
  the one it was trained at. On the local 30M CPT checkpoint
  (`latest.state.pkl`), setting it to 3 lifts 60-token greedy
  `unique_ratio` from 0.129 to 0.220 without a penalty; combined with
  `repetition_penalty=1.15` it reaches 0.692.
- No kernel structure change. `shared_first` remains the default: the
  per_iteration win on decoding does not pay for its inference-time
  inflexibility (4× KV, no depth override, no speculative decoding).
- There is no "auto" depth selector in 0.3.4. An earlier draft of the
  release added one that would have read a trained `dt` schedule off
  `ChronosConfig`, but that schedule only lives on
  `recurrent_controller.step_sizes` in the state dict — the helper
  no-opped on every real load and was removed before this release.

## Files

- `design.md` — pre-registered arm definitions and pass rule.
- `arm_<name>.json` — per-step metrics the trainer wrote (`evals`,
  `train_curve`, final checkpoint path, `config` with the arm's resolved
  recipe).
- `arm_<name>.log` — the training-process stdout.
- `queue.log`, `queue_obj.log` — arm-level queue transitions.
- `probes/prompts.jsonl` — the 12 EN/ZH prompts used for greedy probing.
- `probes/<name>.json` — per-prompt greedy continuations and the
  `unique_ratio` / `max_run` / `max_4gram` averages in the arms table.
- `probes/probe_arm.py` — the greedy probe. Runs the Torch state dict
  through MLX via `ChronosMLXModel.from_chronos_pytorch` so experts are
  included.

The final `.pth` and training-output subdirectories are not committed;
`arm_<name>.json`'s `checkpoint` field is written relative
(`run_<name>/arm_<name>.pth`) for readers who rerun the arms locally.

## Reproducibility

Each arm's resolved recipe is captured in `arm_<name>.json` under
`config`. Only `recurrent_attention_mode` differs between `base` and
`C_periter`; only `recurrent_dt_init`/`recurrent_dt_max` differ for
`A_dtcap`; `B_plain` sets `recurrent_depth=1`; `obj_lm` zeroes the
auxiliary-loss weights listed above and keeps every other field at
`base`. All five arms share seed 1234, context 512, 65,536 tokens per
step, peak LR 1.5e-3, and the same 30-min wall budget.

To rerun a probe:

```
python experiments/loop_depth_15m/results_abc/probes/probe_arm.py \
    --pth run_<name>/arm_<name>.pth \
    --tokenizer <training tokenizer.json> \
    --prompts experiments/loop_depth_15m/results_abc/probes/prompts.jsonl \
    --max-tokens 60 \
    --out experiments/loop_depth_15m/results_abc/probes/<name>.json
```
