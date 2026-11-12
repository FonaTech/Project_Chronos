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
pip install -e '.[mlx,serve]'               # 内核（或：pip install --pre 'Project_Chronos[mlx,serve]==0.3.2a0'）
pip install -e demos/translation            # 本扩展（sacrebleu、sacremoses、pyarrow 固定版本）
chronos-translate --version
```

扩展固定依赖 `Project_Chronos==0.3.2a0`。请检出对应标签（`git checkout v0.3.2a0`），使两边版本一致。
模型包需要这两个 wheel，可以从检出目录构建：

```sh
python -m build --outdir dist .                     # dist/project_chronos-0.3.2a0-py3-none-any.whl
python -m build --outdir dist demos/translation     # dist/chronos_translate-0.3.2a0-py3-none-any.whl
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

**50M 英译中 demo（进行中；`demos/translation`）**
- 模型：总参数 49.69M，激活 12.09M；36 个专家；MLA latent 128；SWA 窗口 128；自研 16k 词表。
- 训练：72 小时 MLX bf16 任务（先预训练，后 SFT），2026-11-05 启动。
- 中途实测：预训练第 13,996 步（计划约 24.7k 步）valid CE 为 2.477（第 1,656 步时为 3.012）。该 checkpoint 已打包为 `chronos-enzh-50m-pretrain-s13996`（见“模型包”）。
- FLORES+ 和 WMT22 上的 BLEU/chrF，以及各项消融（注意力模式、深度、懒加载专家预算、推测 γ），等训练结束后公布。在此之前不给出任何质量数字。

## 已发布模型：chronos-enzh-50m-sft

[`models/chronos-enzh-50m-sft/`](models/chronos-enzh-50m-sft/) 打包了一个 4,969 万参数的 Chronos v3 模型（每个 token 激活 1,209 万），在一台 Apple M5 Max 上训练了 69 小时。目录中有模型配置、分词器、训练时的 prompt、解码参数、驱动脚本（`translate.py`），以及每个数字背后的评测。权重（99.5 MB，bfloat16 safetensors）作为附件放在 release [`enzh-50m-sft-v1`](https://github.com/FonaTech/Project_Chronos/releases/tag/enzh-50m-sft-v1) 中。

- **边界：** 只做英译中，每次一句（≤ 128 token）。中译英已屏蔽：只含英译中的微调抹掉了这一能力（SFT 后 FLORES+ BLEU 0.00）。驱动会拒绝边界以外的输入。
- **实测**（MLX bf16，贪心解码）：FLORES+ devtest BLEU 33.48 / chrF 29.52；WMT22 BLEU 32.60 / chrF 28.83；WMT24++ BLEU 26.89 / chrF 24.43（分句）。
- **仅供研究使用。** 训练语料的许可未知，见目录中的 `NOTICE`。

```bash
cd demos/translation/models/chronos-enzh-50m-sft
pip install --pre "Project_Chronos[mlx]==0.3.2a0"   # 不用 MLX 时装 "Project_Chronos==0.3.2a0"
python translate.py --download && python translate.py "The meeting was moved to Friday."
```

## 模型包（翻译 demo 扩展）

模型包属于翻译扩展（`demos/translation`）。先构建两个 wheel（`python -m build --outdir dist .` 和 `python -m build --outdir dist demos/translation`）。`chronos-translate bundle build` 把某次运行导出的一个 checkpoint 打包成解压即用的目录。打包过程只读取运行目录。

```bash
chronos-translate bundle build --run-dir runs/enzh_50m_v1 --which best --pack \
    --wheel dist/project_chronos-0.3.2a0-py3-none-any.whl --wheel dist/chronos_translate-0.3.2a0-py3-none-any.whl
```

| 包内路径 | 内容 |
|---|---|
| `model/` | 权重 + `.config.json` 配置文件，分词器（`model/tokenizer/`） |
| `eval/` | FLORES+ dev/devtest（CC BY-SA 4.0），供 `bundle eval` / `probe` 使用 |
| `data/sft_valid/` | 该次运行的 SFT 验证分片 |
| `wheels/`、`bin/setup.sh`、`bin/chronos` | 把内核和扩展两个 wheel 离线安装到 `.venv`；启动脚本设置 `CHRONOS_BUNDLE`，并把 `bundle …` 转给 `chronos-translate` |
| `recipes/source_run_spec.json`、`recommended.json` | 产生这份权重的配方；微调档位 `smoke` / `quick` / `standard` / `full` |
| `bundle.json` | 来源信息（步数、token 数、valid CE）、实测质量探针、每个文件的 SHA-256 |

接收方的用法：

```bash
bin/setup.sh                                              # 建 venv、装 wheels、运行 `bundle check`
bin/chronos bundle translate "The weather is very nice today."
bin/chronos bundle translate --backend mlx --speculative --file examples/translate_en.txt
bin/chronos bundle eval --split devtest -n 200
bin/chronos bundle prepare-data my_pairs.jsonl --out work/data      # 会从你的数据中剔除 FLORES+
bin/chronos bundle finetune --data work/data --run-dir work/ft --tier quick --mode sft
bin/chronos bundle serve -- --port 11435
```

`--speculative` 要求 `--repetition-penalty 1.0`（此时的默认值），因为惩罚会逐位置改变目标分布。`finetune` 写出一份 `init_checkpoint` 指向包内权重的 `chronos-translate run` spec；运行目录里已有训练时会拒绝写入。用许可未知的数据（WMT19、OPUS-100）训练的模型包，`NOTICE.md` 中写明权重仅供研究使用。

## 在本 demo 上做的内核实测

主 README 中有三项实测用到了本 demo 的模型或数据：
- 自推测解码表：预训练 step 13,996 checkpoint，greedy，batch 1，FLORES+ dev 英→中提示，起草深度 1，γ 4（报告在 `reports/speculative/`）。
- 0.3.2 原生内核一致性：50M 英译中 checkpoint 的 logits 在 0.3.1a3 + minimind 与原生内核之间逐位一致；计时时 50M 英译中 SFT 正在 GPU 上训练。
- MLX 编译步与激活重计算（2026-11-08）：50M 英译中 recipe 的模型配置，随机初始化，真实英译中分片。
