# chronos-translate：基于 Chronos 内核的英↔中翻译 demo

Chronos 内核（`chronos/`）用于训练通用模型，不含任何翻译特化代码。
构建英↔中标杆模型所需的全部内容都放在这里，作为调用内核的扩展。

| 路径 | 内容 |
|---|---|
| `chronos_translate/corpus/` | 下载、清洗、去重并分片英中数据（WMT19、OPUS-100、FLORES+ 等）；`docmix` 构建 n 合 1 合并集 |
| `chronos_translate/run.py` | `TranslationLongRun`：内核 `LongRun` 加 FLORES+ BLEU/chrF 验证 |
| `chronos_translate/doc_run.py` | doc-mix 版本：微批自适应、`target_passes`、篇章级评测 |
| `chronos_translate/probe.py` | 对运行检查点做离线 FLORES+ 探测 |
| `chronos_translate/bundle/` | 自包含模型包：构建、校验、翻译、准备数据、微调、服务 |
| `chronos_translate/model_presets.py` | 50M 拓扑基础配置和分词器语言标记 |
| `recipes/enzh_50m_72h.json` | 72 小时标杆配方（路径相对于本文件） |
| `recipes/docmix_plan.json` | doc-mix 训练包 A/B/C 的规划器输出 |
| `scripts/` | `plan_docmix.py`、`make_docmix_packages.py` |
| `tests/` | 扩展测试（`pytest demos/translation/tests`） |

## 安装

本扩展是仓库里的一个 demo，不发布到 PyPI，需要从仓库检出后安装。
内核可以从 PyPI 安装（0.3.1 处于 beta 阶段，需加 `pip install --pre`），也可以用同一份检出安装。

```sh
git clone https://github.com/FonaTech/Project_Chronos.git && cd Project_Chronos
pip install -e '.[mlx,serve]'               # 内核（或：pip install --pre 'Project_Chronos[mlx,serve]==0.3.1a2'）
pip install -e demos/translation            # 本扩展（sacrebleu、sacremoses、pyarrow 固定版本）
chronos-translate --version
```

扩展固定依赖 `Project_Chronos==0.3.1a2`。请检出对应标签（`git checkout v0.3.1a2`），使两边版本一致。
模型包需要这两个 wheel，可以从检出目录构建：

```sh
python -m build --outdir dist .                     # dist/project_chronos-0.3.1a2-py3-none-any.whl
python -m build --outdir dist demos/translation     # dist/chronos_translate-0.3.1a2-py3-none-any.whl
```

扩展测试在仓库根目录运行：`pytest demos/translation/tests`。
内核 CI 不运行这些测试，因为它们需要 sacrebleu 和 pyarrow。

## 命令

```sh
chronos tokenizer data/pairs.jsonl --out tok --vocab-size 16384 \
    --extra-special "<|en|>" "<|zh|>" "<|translate|>"    # 内核分词器 + 语言标记
chronos-translate corpus download --help                 # 另有 build | shards | docmix
chronos-translate run --spec demos/translation/recipes/enzh_50m_72h.json
chronos-translate doc-run --spec <训练包>/recipe.json
chronos-translate inspect --run-dir runs/enzh_50m_v1 --samples 24
chronos-translate bundle build --run-dir runs/enzh_50m_v1 --wheel dist/project_chronos-*.whl \
    --wheel dist/chronos_translate-*.whl
```

`chronos serve`、`chronos monitor` 和 `chronos tokenizer` 仍是内核命令。

## 如何接入内核

这个目录也是其他任务扩展的模板。它只使用内核公开的钩子，不修改内核代码：

| 内核钩子 | 本扩展的用法 |
|---|---|
| `RunSpec` 子类 | `TranslationRunSpec` 增加 `bleu_samples`、`probe_samples` 和自己的 `export_prefix` |
| `LongRun.eval_samples_for` | 每个阶段的样本数（预训练探测、SFT BLEU 子集、最终 devtest） |
| `LongRun.task_eval` | 返回 `{指标: 数值}`；这些键写入 `metrics.jsonl` 的 `validation` / `final` 行，并显示在 monitor 中 |
| `LongRun.generate`、`write_samples` | 带 KV cache 的贪心解码；示例输出写入 `samples.jsonl` |
| `LongRun.pinned_modules` | 在运行记录中写入 sacrebleu 和扩展模块的版本 |
| `longrun.main(runner_cls=, spec_cls=, extra_path_keys=)` | `chronos-translate run` 命令行；`extra.raw_root` 相对于配方目录解析 |
| `special_tokens_with(...)` | 分词器中的 `<|en|>`、`<|zh|>`、`<|translate|>` 位置 |

`task_eval` 出错时记为 `task_eval_error`，不会中断训练。
新任务照搬这个结构即可：独立的包、一个 `RunSpec` 子类、一个 `LongRun` 子类，以及调用内核 `main` 的命令行。

## 从 0.3.1a1 迁移

| 0.3.1a1（内核） | 0.3.1a2（本扩展） |
|---|---|
| `chronos corpus …` | `chronos-translate corpus …` |
| `chronos bundle …` | `chronos-translate bundle …`（模型包里的 `bin/chronos bundle …` 仍可用） |
| `chronos longrun --spec enzh_50m_72h.json` | `chronos-translate run --spec …` |
| `python -m chronos.mlx.training.doc_run` | `chronos-translate doc-run` |
| `chronos.data.corpus.*` | `chronos_translate.corpus.*` |
| `chronos.bundle.*` | `chronos_translate.bundle.*` |
| `inspect_checkpoint.translation_probe` | `chronos_translate.probe.translation_probe` |
| `chronos.model.factory.ENZH_DEMO_BASE` | `chronos_translate.model_presets.ENZH_50M_BASE` |

配方格式不变：`bleu_samples` 和 `probe_samples` 属于 `TranslationRunSpec`，
内核运行器遇到它们会报错并指向这里。配方中的相对路径现在相对于配方所在目录解析。

## 数据与权重

WMT19 和 OPUS-100 的许可证标注为 unknown，用它们训练的权重只能用于研究。
FLORES+（CC BY-SA 4.0）只用于评测。
