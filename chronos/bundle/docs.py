"""Bundle documents: README (en/zh), NOTICE, example data and launcher scripts."""
from __future__ import annotations

import json
import os

# Written for this bundle (not taken from any corpus); just enough to prove
# the prepare → fine-tune → translate path end to end.
EXAMPLE_PAIRS = [
    ("The weather is very nice today.", "今天天气很好。"),
    ("Please close the window before you leave.", "离开之前请关好窗户。"),
    ("I will call you when I arrive at the station.", "我到车站后会给你打电话。"),
    ("This book is more interesting than I expected.", "这本书比我想象的更有趣。"),
    ("We need to finish the report by Friday.", "我们需要在周五之前完成报告。"),
    ("The museum is closed on Mondays.", "博物馆周一闭馆。"),
    ("Could you tell me how to get to the library?", "你能告诉我怎么去图书馆吗？"),
    ("He has been learning Chinese for three years.", "他学中文已经三年了。"),
    ("The meeting has been moved to next Tuesday.", "会议改到了下周二。"),
    ("Drink more water when the weather is hot.", "天气热的时候要多喝水。"),
    ("The train was delayed by twenty minutes.", "火车晚点了二十分钟。"),
    ("She opened a small coffee shop near the university.", "她在大学附近开了一家小咖啡店。"),
    ("Remember to save your work before closing the program.", "关闭程序之前记得保存你的工作。"),
    ("The children are playing football in the park.", "孩子们在公园里踢足球。"),
    ("Prices in this market are usually lower.", "这个市场的价格通常更低。"),
    ("My phone battery is almost empty.", "我的手机快没电了。"),
    ("The new bridge will open to traffic next year.", "新桥将于明年通车。"),
    ("Please send me the file by email.", "请通过电子邮件把文件发给我。"),
    ("It took us two hours to climb the mountain.", "我们花了两个小时爬上这座山。"),
    ("The doctor advised him to get more sleep.", "医生建议他多睡觉。"),
    ("Our team won the first match of the season.", "我们队赢得了本赛季的第一场比赛。"),
    ("The restaurant is famous for its noodles.", "这家餐厅以面条闻名。"),
    ("I forgot to bring my umbrella this morning.", "我今天早上忘了带伞。"),
    ("The company plans to hire fifty new engineers.", "公司计划招聘五十名新工程师。"),
    ("Turn left at the second traffic light.", "在第二个红绿灯处左转。"),
    ("The river flows through the center of the city.", "这条河流经城市中心。"),
    ("He fixed the old bicycle by himself.", "他自己修好了那辆旧自行车。"),
    ("Students must hand in their homework on time.", "学生必须按时交作业。"),
    ("The flowers in the garden bloom every spring.", "花园里的花每年春天都会开。"),
    ("We watched a film together last night.", "我们昨晚一起看了一部电影。"),
    ("This road is very busy during rush hour.", "这条路在高峰时段非常拥堵。"),
    ("She speaks three languages fluently.", "她能流利地说三种语言。"),
    ("The hotel room has a view of the sea.", "酒店房间可以看到大海。"),
    ("Please turn off the lights when you leave the room.", "离开房间时请关灯。"),
    ("The price of vegetables rose this month.", "这个月蔬菜价格上涨了。"),
    ("My grandmother tells the best stories.", "我奶奶讲的故事最好听。"),
    ("The concert starts at eight in the evening.", "音乐会晚上八点开始。"),
    ("They planted trees along the new road.", "他们在新路两旁种了树。"),
    ("I am looking for a quiet place to study.", "我在找一个安静的地方学习。"),
    ("The airport is about forty kilometres from here.", "机场离这里大约四十公里。"),
]


def write_examples(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pairs_en_zh.jsonl"), "w", encoding="utf-8") as fh:
        for en, zh in EXAMPLE_PAIRS:
            fh.write(json.dumps({"en": en, "zh": zh}, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "translate_en.txt"), "w", encoding="utf-8") as fh:
        fh.write("The weather is very nice today.\nPlease send me the file by email.\n")


LAUNCHER = """#!/bin/sh
# Runs `chronos` with this bundle's settings (no git, no network needed).
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export CHRONOS_BUNDLE="${CHRONOS_BUNDLE:-$HERE}"
export CHRONOS_MINIMIND_DIR="${CHRONOS_MINIMIND_DIR:-$HERE/third_party/minimind}"
if [ -x "$HERE/.venv/bin/chronos" ]; then exec "$HERE/.venv/bin/chronos" "$@"; fi
exec chronos "$@"
"""

SETUP = """#!/bin/sh
# One-time setup: a virtualenv in .venv with Chronos and its pinned extras.
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python >= 3.10 required")'
[ -d "$HERE/.venv" ] || "$PY" -m venv "$HERE/.venv"
. "$HERE/.venv/bin/activate"
python -m pip install --upgrade pip
EXTRAS="translate,serve"
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then EXTRAS="$EXTRAS,mlx"; fi
WHEEL="$(ls "$HERE"/wheels/*.whl 2>/dev/null | head -n 1 || true)"
if [ -n "$WHEEL" ]; then
  python -m pip install "$WHEEL[$EXTRAS]"
else
  echo "no wheel in wheels/: install Chronos from source with  pip install -e '<repo>[$EXTRAS]'" >&2
  exit 1
fi
"$HERE/bin/chronos" bundle check
"""


def _fmt(value, digits: int = 2):
    return "n/a" if value is None else (f"{value:,}" if isinstance(value, int) else f"{value:.{digits}f}")


def _quality_rows(manifest: dict) -> list[str]:
    q = manifest.get("quality") or {}
    rows = []
    for key, res in (q.get("results") or {}).items():
        rp = key.split("_", 1)[1]
        for direction in ("en_zh", "zh_en"):
            r = res.get(direction)
            if r:
                rows.append(f"| {direction.replace('_', '→')} | {rp} | {r['n']} | {r['bleu']:.2f} | {r['chrf']:.2f} |")
    return rows


def _tier_rows(recs: dict) -> list[str]:
    rows = []
    for tier, t in recs["mlx_longrun"]["tiers"].items():
        sft = t["phases"]["sft"][0]
        rows.append(f"| `{tier}` | {t['hours']} | {sft['peak_lr']:g} | {sft['tokens_per_step']:,} | "
                    f"{sft['warmup_fraction']:g} | {t['why']} |")
    return rows


def _readme_en(m: dict, recs: dict) -> str:
    tr, arch, params = m.get("training") or {}, m["model"]["architecture"], m["model"].get("params") or {}
    status = ("**Interim checkpoint**: the training run was still in progress when this bundle was built. "
              "Rebuild from the finished run for the final weights.") if m.get("status") == "interim" else \
        "Final checkpoint of the training run."
    q = _quality_rows(m)
    lines = [
        f"# {m['name']}", "",
        f"Chronos v3 English↔Chinese **pretrained** model, packaged to run right after unpacking. {status}", "",
        "## What is inside", "",
        "| Item | Value |", "|---|---|",
        f"| Stage | {m['model']['stage']} ({m['model']['which']} validation checkpoint) |",
        f"| Training step | {_fmt(tr.get('step'))} of ~{_fmt(tr.get('planned_steps_at_export'))} planned |",
        f"| Tokens seen | {_fmt(tr.get('tokens_seen'))} |",
        f"| Held-out valid CE / PPL | {_fmt(tr.get('valid_ce'), 4)} / {_fmt(tr.get('valid_ppl'), 2)} (measured) |",
        f"| Parameters | total {_fmt(params.get('total'))}, active {_fmt(params.get('active'))} per token |",
        f"| Topology | hidden {arch['hidden_size']}, {arch['num_hidden_layers']} layers "
        f"(prelude {arch['prelude_layers']}, coda {arch['coda_layers']}), recurrent depth {arch['recurrent_depth']} "
        f"({arch['recurrent_attention_mode']}), {arch['num_experts']} experts top-{arch['num_experts_per_tok']} + "
        f"{arch['num_shared_experts']} shared |",
        f"| Vocabulary | {arch['vocab_size']:,} (own Chronos BPE tokenizer in `model/tokenizer/`) |", "",
        "Prompt format (pretrain): `<|en|>` English `<|zh|>` → Chinese, or `<|zh|>` Chinese `<|en|>` → English. "
        "The model has **not** been instruction-tuned: it translates single sentences, and long or unusual inputs "
        "can loop or drop entities. A repetition penalty of 1.2 (the CLI default) suppresses most loops.", "",
    ]
    if q:
        lines += [f"Quality on the first {m['quality']['first_n']} FLORES+ dev sentences ({m['quality']['label']}):", "",
                  "| Direction | Repetition penalty | n | BLEU | chrF |", "|---|---|---|---|---|", *q, ""]
    lines += [
        "## Quick start", "",
        "```sh",
        "tar xzf " + m["name"] + ".tar.gz && cd " + m["name"],
        "sh bin/setup.sh                      # .venv + pinned extras + `chronos bundle check`",
        "bin/chronos bundle translate \"The weather is very nice today.\"",
        "bin/chronos bundle translate --direction zh_en \"今天天气很好。\"",
        "bin/chronos bundle translate --file examples/translate_en.txt",
        "bin/chronos bundle translate --backend mlx --speculative --file examples/translate_en.txt",
        "bin/chronos bundle eval --split devtest -n 100   # BLEU/chrF on FLORES+",
        "```", "",
        "`--speculative` (recurrent-depth self-speculative decoding) gives the same greedy output as plain "
        "decoding and reports its acceptance rate on stderr. It needs `--repetition-penalty 1.0`, its default. "
        "On CPU at this model size it was measured slower than plain decoding.", "",
        "`bin/chronos` sets `CHRONOS_BUNDLE` and `CHRONOS_MINIMIND_DIR` (the vendored minimind in "
        "`third_party/`), so nothing is fetched from the network. Without `bin/`, export both variables and call "
        "`chronos bundle …` directly.", "",
        "## Next stage: SFT (fine-tuning) on Apple silicon (MLX)", "",
        "```sh",
        "# 1. your pairs: .jsonl {\"en\":…, \"zh\":…} | .csv | .tsv | .parquet",
        "bin/chronos bundle prepare-data my_pairs.jsonl --out work/data      # clean, dedup, tokenize, shard",
        "# add --unsegment-zh / --detokenize-en for word-segmented or Moses-tokenized corpora",
        "# 2. fine-tune from the bundled weights (resumable; rerun the same command after an interruption)",
        "bin/chronos bundle finetune --data work/data --run-dir work/run --tier quick",
        "# smoke test of the whole path on the bundled examples (minutes):",
        "bin/chronos bundle prepare-data examples/pairs_en_zh.jsonl --out work/smoke_data",
        "bin/chronos bundle finetune --data work/smoke_data --run-dir work/smoke_run --tier smoke",
        "```", "",
        "`--mode sft` (default) trains the chat prompt `<|im_start|>user\\n翻译成中文，只输出译文：\\n…`; "
        "`--mode pretrain` continues pretraining on new text; `--mode pretrain+sft` does both (40/60 time split). "
        "Monitor a run with `bin/chronos monitor --runs-dir work`. The fine-tuned checkpoint lands in "
        "`work/run/*_ft_sft.best.pth`; rebuild a bundle from it with `chronos bundle build --run-dir work/run "
        "--phase sft`.", "",
        "Tiers (from `recommended.json`; every value is labelled with its source):", "",
        "| Tier | Hours | SFT peak LR | Tokens / step | Warmup | Use |", "|---|---|---|---|---|---|",
        *_tier_rows(recs), "",
        "Torch (CUDA/CPU) path: `recommended.json → torch_stages` lists per-stage values for "
        "`train_chronos_sft.py` and the DPO/ORPO/GRPO/distill scripts; pass `--from_weight model/<name>.pth` "
        "(the tokenizer next to it is picked up automatically, or set `--tokenizer_path model/tokenizer`).", "",
        "## Serve (Ollama / OpenAI protocol)", "",
        "```sh",
        "bin/chronos bundle serve -- --port 11435        # loopback only, no auth",
        "curl -s localhost:11435/api/generate -d '{\"model\":\"" + m["model"]["checkpoint"].split("/")[-1][:-4]
        + "\",\"prompt\":\"<|en|>Good morning.<|zh|>\",\"raw\":true,\"stream\":false}'",
        "```", "",
        "The server binds 127.0.0.1 and has no authentication. Exposing it needs `--allow-remote` plus an API key "
        "in `CHRONOS_API_KEY`.", "",
        "## Files", "",
        "`bundle.json` (identity, provenance, sha256 of every file) · `model/` (weights, config sidecar, tokenizer) · "
        "`recipes/source_run_spec.json` (the exact training spec) · `recommended.json` (next-stage parameters) · "
        "`eval/` (FLORES+) · `data/sft_valid/` (held-out SFT validation shards) · `examples/` · `third_party/` · "
        "`wheels/` · `bin/`. Verify integrity any time with `bin/chronos bundle check`.", "",
        "See `NOTICE.md` before redistributing: the weights are for **research use only**.", "",
    ]
    return "\n".join(lines)


def _readme_zh(m: dict, recs: dict) -> str:
    tr, arch, params = m.get("training") or {}, m["model"]["architecture"], m["model"].get("params") or {}
    status = "**中期检查点**：打包时训练仍在进行，训练结束后请用最终 run 重新打包。" if m.get("status") == "interim" \
        else "训练 run 的最终检查点。"
    q = _quality_rows(m)
    lines = [
        f"# {m['name']}", "",
        f"Chronos v3 英↔中**预训练**模型，解压后即可使用。{status}", "",
        "## 包内容", "",
        "| 项目 | 数值 |", "|---|---|",
        f"| 阶段 | {m['model']['stage']}（{m['model']['which']} 验证检查点） |",
        f"| 训练步数 | {_fmt(tr.get('step'))} / 计划约 {_fmt(tr.get('planned_steps_at_export'))} |",
        f"| 已见 token | {_fmt(tr.get('tokens_seen'))} |",
        f"| 留出验证 CE / PPL | {_fmt(tr.get('valid_ce'), 4)} / {_fmt(tr.get('valid_ppl'), 2)}（实测） |",
        f"| 参数量 | 总计 {_fmt(params.get('total'))}，每 token 激活 {_fmt(params.get('active'))} |",
        f"| 结构 | hidden {arch['hidden_size']}，{arch['num_hidden_layers']} 层（prelude {arch['prelude_layers']}，"
        f"coda {arch['coda_layers']}），循环深度 {arch['recurrent_depth']}（{arch['recurrent_attention_mode']}），"
        f"{arch['num_experts']} 专家 top-{arch['num_experts_per_tok']} + {arch['num_shared_experts']} 共享专家 |",
        f"| 词表 | {arch['vocab_size']:,}（自研 Chronos BPE 分词器，位于 `model/tokenizer/`） |", "",
        "预训练格式：`<|en|>` 英文 `<|zh|>` → 中文，或 `<|zh|>` 中文 `<|en|>` → 英文。模型**未经指令微调**，"
        "适合单句翻译；长句或少见输入可能出现重复或实体错误。命令行默认 repetition penalty 1.2，可抑制大部分重复。", "",
    ]
    if q:
        lines += [f"FLORES+ dev 前 {m['quality']['first_n']} 句的质量（{m['quality']['label']}）：", "",
                  "| 方向 | repetition penalty | n | BLEU | chrF |", "|---|---|---|---|---|", *q, ""]
    lines += [
        "## 快速开始", "",
        "```sh",
        "tar xzf " + m["name"] + ".tar.gz && cd " + m["name"],
        "sh bin/setup.sh                      # 创建 .venv，安装固定版本依赖，并运行 bundle check",
        "bin/chronos bundle translate \"The weather is very nice today.\"",
        "bin/chronos bundle translate --direction zh_en \"今天天气很好。\"",
        "bin/chronos bundle translate --file examples/translate_en.txt",
        "bin/chronos bundle translate --backend mlx --speculative --file examples/translate_en.txt",
        "bin/chronos bundle eval --split devtest -n 100   # FLORES+ 上的 BLEU/chrF",
        "```", "",
        "`--speculative`（循环深度自推测解码）的 greedy 输出与普通解码一致，接受率打印到 stderr。"
        "它要求 `--repetition-penalty 1.0`（此时的默认值）。在这个规模的模型上用 CPU 实测，它比普通解码慢。", "",
        "`bin/chronos` 会设置 `CHRONOS_BUNDLE` 和 `CHRONOS_MINIMIND_DIR`（指向 `third_party/` 里随包附带的 minimind），"
        "全程不需要联网。不用 `bin/` 时，手动导出这两个变量后直接调用 `chronos bundle …`。", "",
        "## 下一阶段：SFT 微调（Apple silicon / MLX）", "",
        "```sh",
        "# 1. 准备平行语料：.jsonl {\"en\":…, \"zh\":…} | .csv | .tsv | .parquet",
        "bin/chronos bundle prepare-data my_pairs.jsonl --out work/data      # 清洗、去重、分词、分片",
        "# 中文已分词或英文经过 Moses tokenize 时，加 --unsegment-zh / --detokenize-en",
        "# 2. 从包内权重开始微调（可断点续训：中断后重跑同一条命令）",
        "bin/chronos bundle finetune --data work/data --run-dir work/run --tier quick",
        "# 用包内示例数据把整条流程跑通（几分钟）：",
        "bin/chronos bundle prepare-data examples/pairs_en_zh.jsonl --out work/smoke_data",
        "bin/chronos bundle finetune --data work/smoke_data --run-dir work/smoke_run --tier smoke",
        "```", "",
        "`--mode sft`（默认）训练对话格式 `<|im_start|>user\\n翻译成中文，只输出译文：\\n…`；"
        "`--mode pretrain` 在新文本上继续预训练；`--mode pretrain+sft` 两者都做（时间按 40/60 分配）。"
        "用 `bin/chronos monitor --runs-dir work` 监控训练。微调后的检查点在 `work/run/*_ft_sft.best.pth`，"
        "可用 `chronos bundle build --run-dir work/run --phase sft` 重新打包。", "",
        "档位（来自 `recommended.json`，每个数值都标注了来源）：", "",
        "| 档位 | 小时 | SFT 峰值 LR | 每步 token | Warmup | 用途 |", "|---|---|---|---|---|---|",
        *_tier_rows(recs), "",
        "Torch（CUDA/CPU）路径：`recommended.json → torch_stages` 给出 `train_chronos_sft.py` 及 DPO/ORPO/GRPO/"
        "distill 脚本的分阶段参数；传入 `--from_weight model/<name>.pth` 即可（同目录的分词器会被自动识别，"
        "也可以显式指定 `--tokenizer_path model/tokenizer`）。", "",
        "## 服务（Ollama / OpenAI 协议）", "",
        "```sh",
        "bin/chronos bundle serve -- --port 11435        # 只监听本机，无认证",
        "curl -s localhost:11435/api/generate -d '{\"model\":\"" + m["model"]["checkpoint"].split("/")[-1][:-4]
        + "\",\"prompt\":\"<|en|>Good morning.<|zh|>\",\"raw\":true,\"stream\":false}'",
        "```", "",
        "服务默认绑定 127.0.0.1，没有认证。对外开放需要 `--allow-remote`，并通过环境变量 `CHRONOS_API_KEY` 提供 API key。", "",
        "## 文件", "",
        "`bundle.json`（身份、来源、每个文件的 sha256）· `model/`（权重、配置 sidecar、分词器）· "
        "`recipes/source_run_spec.json`（产生这份权重的训练配置）· `recommended.json`（下一阶段参数）· "
        "`eval/`（FLORES+）· `data/sft_valid/`（留出的 SFT 验证分片）· `examples/` · `third_party/` · `wheels/` · "
        "`bin/`。随时可用 `bin/chronos bundle check` 校验完整性。", "",
        "再分发前请阅读 `NOTICE.md`：权重**仅供研究使用**。", "",
    ]
    return "\n".join(lines)


def _notice(m: dict) -> str:
    mm = (m.get("third_party") or {}).get("minimind") or {}
    return "\n".join([
        "# NOTICE", "",
        "## Weights: research use only", "",
        "The weights were trained on parallel text whose licences are not all known:", "",
        "| Source | Use | Licence as recorded at download |", "|---|---|---|",
        "| WMT19 zh-en (HF `wmt/wmt19`, news-commentary, UN, CWMT, wikititles) | training | unknown (dataset card); "
        "research use only |",
        "| OPUS-100 en-zh (HF `Helsinki-NLP/opus-100`) | training | unknown (dataset card) |",
        "| iic WMT Chinese-English corpus (ModelScope) | training | not recorded |",
        "| FLORES+ dev/devtest (shipped in `eval/`) | evaluation only; removed from training | CC BY-SA 4.0 |", "",
        "Because the training data licences are unknown, these weights are distributed **for research and "
        "evaluation only**. Do not use them commercially or redistribute them as a product without checking the "
        "licences of the underlying corpora yourself.", "",
        "## Code", "",
        "- Chronos (`wheels/`, `bin/`): Apache-2.0.",
        f"- minimind (`third_party/minimind/`, commit `{mm.get('commit') or 'unknown'}`): Apache-2.0, "
        "© jingyaogong. Its `LICENSE` file is included.",
        "- FLORES+ files in `eval/` keep their CC BY-SA 4.0 licence (attribution: the FLORES+ / OLDI project).", "",
        "## No warranty", "",
        "Provided as is. Machine translation output can be wrong; check it before relying on it.", "",
    ])


def write_docs(bundle_dir: str, manifest: dict, recs: dict | None = None) -> None:
    if recs is None:
        with open(os.path.join(bundle_dir, "recommended.json"), encoding="utf-8") as fh:
            recs = json.load(fh)
    for name, text in (("README.md", _readme_en(manifest, recs)), ("README_zh.md", _readme_zh(manifest, recs)),
                       ("NOTICE.md", _notice(manifest))):
        with open(os.path.join(bundle_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    bin_dir = os.path.join(bundle_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    for name, text in (("chronos", LAUNCHER), ("setup.sh", SETUP)):
        path = os.path.join(bin_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(path, 0o755)
