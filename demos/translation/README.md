# chronos-translate: an en↔zh translation demo on the Chronos kernel

The Chronos kernel (`chronos/`) trains general models and carries no
translation-specific code. Everything that builds the English↔Chinese
benchmark model lives here, as an extension that imports the kernel.

| Path | What it is |
|---|---|
| `chronos_translate/corpus/` | download, clean, dedup and shard en–zh data (WMT19, OPUS-100, FLORES+, …); `docmix` builds merged n-in-1 sets |
| `chronos_translate/run.py` | `TranslationLongRun`: the kernel `LongRun` plus FLORES+ BLEU/chrF validation |
| `chronos_translate/doc_run.py` | doc-mix variant: micro-batch auto-fit, `target_passes`, document-level eval |
| `chronos_translate/probe.py` | offline FLORES+ probe of a run checkpoint |
| `chronos_translate/bundle/` | self-contained model bundles: build, check, translate, prepare-data, fine-tune, serve |
| `chronos_translate/model_presets.py` | the 50M topology base and the tokenizer's language tokens |
| `recipes/enzh_50m_72h.json` | the 72h benchmark recipe (paths relative to this file) |
| `recipes/docmix_plan.json` | planner output for the doc-mix packages A/B/C |
| `scripts/` | `plan_docmix.py`, `make_docmix_packages.py` |
| `tests/` | extension tests (`pytest demos/translation/tests`) |

## Install

This extension is a demo inside the repository. It is not published on PyPI;
install it from a checkout. The kernel comes from PyPI (`pip install --pre`
while 0.3.1 is in beta) or from the same checkout.

```sh
git clone https://github.com/FonaTech/Project_Chronos.git && cd Project_Chronos
pip install -e '.[mlx,serve]'               # the kernel (or: pip install --pre 'Project_Chronos[mlx,serve]==0.3.2a0')
pip install -e demos/translation            # this extension (sacrebleu, sacremoses, pyarrow pinned)
chronos-translate --version
```

The extension pins `Project_Chronos==0.3.2a0`. Use the checkout's tag
(`git checkout v0.3.2a0`) so both sides match. A model bundle needs the two
as wheels; build them from the checkout:

```sh
python -m build --outdir dist .                     # dist/project_chronos-0.3.2a0-py3-none-any.whl
python -m build --outdir dist demos/translation     # dist/chronos_translate-0.3.2a0-py3-none-any.whl
```

The extension's tests run from the repo root: `pytest demos/translation/tests`.
The kernel CI does not run them, because they need sacrebleu and pyarrow.

## Commands

```sh
chronos tokenizer data/pairs.jsonl --out tok --vocab-size 16384 \
    --extra-special "<|en|>" "<|zh|>" "<|translate|>"    # kernel tokenizer + language tags
chronos-translate corpus download --help                 # also: build | shards | docmix
chronos-translate run --spec demos/translation/recipes/enzh_50m_72h.json
chronos-translate doc-run --spec <package>/recipe.json
chronos-translate inspect --run-dir runs/enzh_50m_v1 --samples 24
chronos-translate bundle build --run-dir runs/enzh_50m_v1 --wheel dist/project_chronos-*.whl \
    --wheel dist/chronos_translate-*.whl
```

`chronos serve`, `chronos monitor` and `chronos tokenizer` stay kernel commands.

## How it plugs into the kernel

This folder is also the template for other task extensions. It uses only
public kernel hooks and never patches kernel code:

| Kernel hook | What this extension does with it |
|---|---|
| `RunSpec` subclass | `TranslationRunSpec` adds `bleu_samples`, `probe_samples` and its own `export_prefix` |
| `LongRun.eval_samples_for` | sample count per phase (pretrain probe, SFT BLEU subset, final devtest) |
| `LongRun.task_eval` | returns `{metric: value}`; the keys land in the `validation` / `final` rows of `metrics.jsonl` and show up in the monitor |
| `LongRun.generate`, `write_samples` | greedy decoding with the KV cache; example outputs go to `samples.jsonl` |
| `LongRun.pinned_modules` | records the versions of sacrebleu and the extension modules in the run |
| `longrun.main(runner_cls=, spec_cls=, extra_path_keys=)` | the `chronos-translate run` CLI, with `extra.raw_root` resolved against the recipe folder |
| `special_tokens_with(...)` | the tokenizer's `<|en|>`, `<|zh|>`, `<|translate|>` slots |

A failing `task_eval` is logged as `task_eval_error` and never stops training.
A new task copies this layout: its own package, a `RunSpec` subclass, a
`LongRun` subclass and a CLI that calls the kernel `main`.

## Migrating from 0.3.1a1

| 0.3.1a1 (kernel) | 0.3.1a2 (this extension) |
|---|---|
| `chronos corpus …` | `chronos-translate corpus …` |
| `chronos bundle …` | `chronos-translate bundle …` (a bundle's `bin/chronos bundle …` still works) |
| `chronos longrun --spec enzh_50m_72h.json` | `chronos-translate run --spec …` |
| `python -m chronos.mlx.training.doc_run` | `chronos-translate doc-run` |
| `chronos.data.corpus.*` | `chronos_translate.corpus.*` |
| `chronos.bundle.*` | `chronos_translate.bundle.*` |
| `inspect_checkpoint.translation_probe` | `chronos_translate.probe.translation_probe` |
| `chronos.model.factory.ENZH_DEMO_BASE` | `chronos_translate.model_presets.ENZH_50M_BASE` |

Recipes keep their format: `bleu_samples` and `probe_samples` belong to
`TranslationRunSpec`, so the kernel runner rejects them with a pointer here.
Relative paths in a recipe now resolve against the recipe's own folder.

## Data and weights

WMT19 and OPUS-100 list their licences as unknown, so weights trained on them
are for research use only. FLORES+ (CC BY-SA 4.0) is used for evaluation only.

**Status of the 50M en→zh demo (in progress; `demos/translation`).**
- Model: 49.69M total / 12.09M active parameters, 36 experts, MLA latent 128, SWA window 128, 16k self-built vocabulary.
- Run: a 72-hour MLX bf16 run (pretrain, then SFT) started on 2026-11-05.
- Interim measurement: valid CE 2.477 at pretrain step 13,996 of about 24.7k planned (it was 3.012 at step 1,656). That checkpoint is packaged as `chronos-enzh-50m-pretrain-s13996` (see Model bundle).
- BLEU/chrF on FLORES+ and WMT22, and the ablations (attention mode, depth, lazy-expert budget, speculative γ), will be published after the run finishes. No quality numbers are claimed before then.

## Released model: chronos-enzh-50m-sft

[`models/chronos-enzh-50m-sft/`](models/chronos-enzh-50m-sft/) packages a 49.7M-parameter Chronos v3 model (12.1M active per token) trained for 69 hours on one Apple M5 Max: configuration, tokenizer, trained prompt, decoding settings, a driver (`translate.py`) and the evaluation behind every number. The weights (99.5 MB, bfloat16 safetensors) are attached to the release [`enzh-50m-sft-v1`](https://github.com/FonaTech/Project_Chronos/releases/tag/enzh-50m-sft-v1).

- **Boundary:** English → Chinese only, one sentence (≤ 128 tokens) at a time. Chinese → English is disabled: English-only fine-tuning erased it (FLORES+ BLEU 0.00 after SFT). The driver refuses input outside the boundary.
- **Measured** (MLX bf16, greedy): FLORES+ devtest BLEU 33.48 / chrF 29.52; WMT22 BLEU 32.60 / chrF 28.83; WMT24++ BLEU 26.89 / chrF 24.43 (split into sentences).
- **Research use only.** The training corpora's licences are unknown; see the folder's `NOTICE`.

```bash
cd demos/translation/models/chronos-enzh-50m-sft
pip install --pre "Project_Chronos[mlx]==0.3.2a0"   # or "Project_Chronos==0.3.2a0" without MLX
python translate.py --download && python translate.py "The meeting was moved to Friday."
```

## Model bundle (translation demo extension)

Bundles belong to the translation extension (`demos/translation`). Build the two wheels first (`python -m build --outdir dist .` and `python -m build --outdir dist demos/translation`). `chronos-translate bundle build` packages one exported checkpoint of a run into a folder that works right after unpacking. Building only reads the run directory.

```bash
chronos-translate bundle build --run-dir runs/enzh_50m_v1 --which best --pack \
    --wheel dist/project_chronos-0.3.2a0-py3-none-any.whl --wheel dist/chronos_translate-0.3.2a0-py3-none-any.whl
```

| Bundle path | Content |
|---|---|
| `model/` | weights + `.config.json` sidecar, tokenizer (`model/tokenizer/`) |
| `eval/` | FLORES+ dev/devtest (CC BY-SA 4.0) for `bundle eval` / `probe` |
| `data/sft_valid/` | the run's SFT validation shards |
| `wheels/`, `bin/setup.sh`, `bin/chronos` | offline install of the kernel and extension wheels into `.venv`; launcher that sets `CHRONOS_BUNDLE` and sends `bundle …` to `chronos-translate` |
| `recipes/source_run_spec.json`, `recommended.json` | the recipe that produced the weights; fine-tune tiers `smoke` / `quick` / `standard` / `full` |
| `bundle.json` | provenance (step, tokens, valid CE), measured quality probe, SHA-256 of every file |

On the recipient side:

```bash
bin/setup.sh                                              # venv + wheels + `bundle check`
bin/chronos bundle translate "The weather is very nice today."
bin/chronos bundle translate --backend mlx --speculative --file examples/translate_en.txt
bin/chronos bundle eval --split devtest -n 200
bin/chronos bundle prepare-data my_pairs.jsonl --out work/data      # FLORES+ is removed from your data
bin/chronos bundle finetune --data work/data --run-dir work/ft --tier quick --mode sft
bin/chronos bundle serve -- --port 11435
```

`--speculative` needs `--repetition-penalty 1.0` (its default), because the penalty changes the target distribution at every position. `finetune` writes a `chronos-translate run` spec with `init_checkpoint` set to the bundle weights and refuses a run directory that already holds a run. Bundles built from data with unknown licences (WMT19, OPUS-100) say in `NOTICE.md` that the weights are for research only.

## Kernel measurements taken on this demo

Three measurements in the main README use this demo's model or data:
- Self-speculative decoding table: pretrain step-13,996 checkpoint, greedy, batch 1, FLORES+ dev en→zh prompts, draft depth 1, γ 4 (reports in `reports/speculative/`).
- 0.3.2 native-kernel parity: the logits of the 50M en→zh checkpoint are bit-identical between 0.3.1a3 + minimind and the native kernel; the timings ran while the 50M en→zh SFT run was training on the GPU.
- MLX compiled step and activation recomputation (2026-11-08): the 50M en→zh recipe's model config from random init, real en→zh shards.
