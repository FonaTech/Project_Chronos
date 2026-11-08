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
