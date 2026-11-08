"""Self-contained MLX training packages for the doc-mix sets (A: 15%/4k/24h, B: 5%/1k/8h, C: 1%/1k/1h).

Each package carries its data, tokenizer, eval sets, Chronos source (no
third-party kernel since 0.3.2), a recipe with relative paths, pinned requirements, launch/monitor
scripts and SHA256SUMS.  Usage::

    python demos/translation/scripts/make_docmix_packages.py OUT_DIR [A B C]

The repository root defaults to three folders above this script (override with
``CHRONOS_REPO``); the planner output is ``recipes/docmix_plan.json`` (written by
``scripts/plan_docmix.py``).
"""
import json
import os
import shutil
import sys
import time

DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.environ.get("CHRONOS_REPO") or os.path.dirname(os.path.dirname(DEMO)))
sys.path[:0] = [REPO, DEMO]
from chronos_translate.bundle.builder import TOKENIZER_FILES, FLORES_REL, FLORES_FILES, sha256_file  # noqa: E402
from chronos.config_schema import model_config_kwargs  # noqa: E402

DATA = os.path.join(REPO, "Translation Training Data")
DOCMIX = os.path.join(DATA, "docmix_v1")
RAW = os.path.join(DATA, "raw")
TOKENIZER = os.path.join(DATA, "tokenizer_v1")
PLAN = os.path.join(DEMO, "recipes", "docmix_plan.json")
SKIP = {"__pycache__", ".DS_Store"}
# The extension's own recipes and tests are not needed inside a package.
SKIP_SRC = {os.path.join("demos", "translation", "recipes"), os.path.join("demos", "translation", "tests")}

# Measured on the built sets (docmix_manifest.json, trial_lengths.py); live run: 2.79 pretrain / ~1.67 SFT passes.
PACKAGES = {
    "A": dict(name="chronos_docmix_A_sci15_ctx4k_24h", set="sci15_ctx4096", hours=24.0, ctx=4096,
              validate_every_s=7200.0, checkpoint_every_s=1800.0, bleu_samples=100, final_eval_samples=1012,
              eval_reserve_fraction=0.06, doc_eval_docs=16, final_doc_eval_docs=1 << 30, tensor_stats_every_s=1800.0,
              probe_samples=16, lengths={"p50": 139, "p95": 1826, "p99": 3568, "max": 4071}),
    "B": dict(name="chronos_docmix_B_sci5_ctx1k_8h", set="sci5_ctx1024", hours=8.0, ctx=1024,
              validate_every_s=3600.0, checkpoint_every_s=1200.0, bleu_samples=100, final_eval_samples=1012,
              eval_reserve_fraction=0.08, doc_eval_docs=16, final_doc_eval_docs=1 << 30, tensor_stats_every_s=1800.0,
              probe_samples=16, lengths={"p50": 138, "p95": 695, "p99": 943, "max": 1014}),
    "C": dict(name="chronos_docmix_C_sci1_ctx1k_1h", set="sci1_ctx1024", hours=1.0, ctx=1024,
              validate_every_s=1200.0, checkpoint_every_s=600.0, bleu_samples=50, final_eval_samples=200,
              eval_reserve_fraction=0.12, doc_eval_docs=8, final_doc_eval_docs=40, tensor_stats_every_s=900.0,
              probe_samples=8, lengths={"p50": 140, "p95": 696, "p99": 944, "max": 1013}),
}
TARGET_PASSES = {"pretrain": 4.0, "sft": 3.0}
# Exact versions resolved from the environment the packages were tested in (pip metadata walk).
REQUIREMENTS = """\
annotated-types==0.7.0
anyio==4.15.1
certifi==2026.4.22
click==8.2.1
exceptiongroup==1.3.0
fastapi==0.115.9
filelock==3.18.0
fsspec==2025.3.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.10.2
idna==3.20
jinja2==3.1.6
lxml==5.4.0
markdown-it-py==3.0.0
markupsafe==3.0.2
mdurl==0.1.2
mlx==0.32.4
mlx-metal==0.32.4
mpmath==1.3.0
networkx==3.4.2
numpy==1.26.4
packaging==24.2
portalocker==4.4.2
psutil==7.2.2
pyarrow==23.0.1
pydantic==2.11.5
pydantic-core==2.33.2
pygments==2.19.1
pyyaml==6.0
regex==2026.4.4
rich==14.0.0
sacrebleu==2.6.0
safetensors==0.5.3
shellingham==1.5.4
starlette==0.45.3
sympy==1.14.0
tabulate==0.10.0
tokenizers==0.22.2
torch==2.10.0
tqdm==4.67.1
transformers==5.5.0
typer==0.16.0
typing-extensions==4.16.0
typing-inspection==0.4.1
uvicorn==0.34.3
"""


def _ignore(folder, names):
    out = {n for n in names if n in SKIP or n.endswith((".pyc", ".egg-info"))}
    for n in names:
        if os.path.relpath(os.path.join(folder, n), REPO) in SKIP_SRC:
            out.add(n)
    return out


def _version(module):
    import importlib
    return getattr(importlib.import_module(module), "__version__", "unknown")


def _copytree(src, dst):
    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=False)


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _write(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, mode)


def _json(path, obj):
    _write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _download_manifest(src, dst):
    m = json.load(open(src, encoding="utf-8"))
    for f in m.get("files", []):
        f.pop("local", None)                       # local absolute/relative paths of this machine
    _json(dst, m)


def recipe(key, cfg, plan, docmix):
    model = dict(plan["model"])
    model.update(max_position_embeddings=cfg["ctx"], rope_theta=1e6, bos_token_id=1, eos_token_id=2,
                 extra_metadata={"docmix_package": cfg["name"]})
    model = model_config_kwargs(model)
    assert model["num_experts"] == 18 and model["max_position_embeddings"] == cfg["ctx"]
    assert model["rope_theta"] == 1e6 and model["eos_token_id"] == 2
    phases = [dict(p) for p in plan["phases"]]
    for p in phases:
        p["seq_len"] = cfg["ctx"]
    train = docmix["splits"]["train"]
    extra = {
        "raw_root": "data/raw", "micro_fit": "auto", "micro_fit_fraction": 0.7,
        "target_passes": TARGET_PASSES, "doc_eval_docs": cfg["doc_eval_docs"],
        "package": cfg["name"],
        "params": {k: plan["arch"][k] for k in ("total_params", "active_params", "active_union_upper",
                                                "virtual_layers", "experts", "top_k", "expert_width")},
        "data": {"set": cfg["set"], "ctx_len": cfg["ctx"], "train_rows": train["rows"], "sft_records": train["records"],
                 "pretrain_tokens": train["pretrain_tokens"], "sft_tokens": train["sft_tokens"],
                 "sft_supervised_tokens": train["sft_supervised_tokens"], "record_len_tokens": cfg["lengths"],
                 "selection": docmix["selection"]},
        "planner": {"memory_estimate": plan["memory"], "scaling": plan["scaling"],
                    "windows": {"sliding_window_size": model["sliding_window_size"],
                                "cluster_locality_window": model["cluster_locality_window"],
                                "rule": "SWA = pow2 ≥ p95/4 (min 128, ≤ ctx); cluster = pow2 nearest p50"}},
        "note": ("target_passes: live 72h run reached 2.79 pretrain / ~1.67 SFT passes; +1 pass each, rounded. "
                 "A phase ends at its target or its time budget, whichever comes first."),
    }
    if cfg["final_doc_eval_docs"] < (1 << 30):
        extra["final_doc_eval_docs"] = cfg["final_doc_eval_docs"]
    return {
        "_comment": f"Chronos doc-mix package {key}: paths are relative to this file.",
        "run_dir": "run", "shards_dir": "data/shards", "tokenizer_dir": "data/tokenizer",
        "model_config": model, "phases": phases, "budget_hours": cfg["hours"],
        "eval_reserve_fraction": cfg["eval_reserve_fraction"], "log_every_steps": 20,
        "validate_every_s": cfg["validate_every_s"], "checkpoint_every_s": cfg["checkpoint_every_s"],
        "bleu_samples": cfg["bleu_samples"], "final_eval_samples": cfg["final_eval_samples"], "seed": 1234,
        "dtype": "bfloat16", "schedule": "time", "tensor_stats_every_s": cfg["tensor_stats_every_s"],
        "probe_samples": cfg["probe_samples"], "export_prefix": "chronos_docmix", "extra": extra,
    }


def shards_manifest(cfg, docmix):
    """Small summary the monitor shows under "data" (``shards_manifest.json`` in shards_dir)."""
    tr, va = docmix["splits"]["train"], docmix["splits"]["valid"]
    return {"format": "chronos-docmix-v1", "set": cfg["set"], "tokenizer_dir": "../tokenizer",
            "tokenizer_fingerprint": docmix.get("tokenizer_fingerprint"), "created_utc": docmix.get("created_utc"),
            "sft_max_len": cfg["ctx"], "pretrain_train_tokens": tr["pretrain_tokens"],
            "pretrain_valid_tokens": va["pretrain_tokens"], "sft_train_records": tr["records"],
            "sft_valid_records": va["records"], "sft_train_tokens": tr["sft_tokens"],
            "sft_train_supervised_tokens": tr["sft_supervised_tokens"], "prompts": docmix.get("prompts"),
            "size_share_records": tr.get("size_share_records"), "dir_share_records": tr.get("dir_share_records"),
            "token_share_by_len": tr.get("token_share_by_len"), "ordered_records": tr.get("ordered_records"),
            "random_concat_records": tr.get("random_concat_records"), "spec": docmix.get("spec")}


LAUNCH = r"""#!/bin/bash
# Chronos doc-mix training package: venv + pinned deps + resumable MLX run.
#   ./launch.sh                 full run (rerun the same command to resume from run/checkpoints/latest.state.pkl)
#   ./launch.sh --smoke         2 updates per phase + short eval in run_smoke/<time>/ (run/ untouched)
#   ./launch.sh --verify        check SHA256SUMS only
#   ./launch.sh --budget-hours 20   any other flags go to chronos_translate.doc_run
# Env: PYTHON=python3.x (venv base, 3.10–3.12), PIP_INDEX_URL (default Tsinghua mirror),
#      CHRONOS_PY=/path/to/python (use an existing environment instead of .venv; no pip install)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "[pkg] 需要 Apple silicon (arm64) 的 macOS：MLX 只支持这个平台" >&2; exit 1
fi
verify() { echo "[pkg] 校验 SHA256SUMS ..."; shasum -a 256 -c --quiet SHA256SUMS && echo "[pkg] 校验通过"; }
if [ "${1:-}" = "--verify" ]; then verify; exit $?; fi
[ -f .verified ] || { verify && touch .verified; }

if [ -n "${CHRONOS_PY:-}" ]; then
  PY="$CHRONOS_PY"
else
  BASE="${PYTHON:-python3}"
  if [ ! -x .venv/bin/python ]; then
    "$BASE" -c 'import sys; v = sys.version_info[:2]; sys.exit(0 if (3, 10) <= v <= (3, 12) else 1)' || {
      echo "[pkg] 需要 Python 3.10–3.12（numpy 1.26.4 的 wheel 范围），用 PYTHON=/path/to/python3.x 指定" >&2; exit 1; }
    "$BASE" -m venv .venv
  fi
  PY="$HERE/.venv/bin/python"
  if ! cmp -s requirements-train.txt .venv/.deps_ok 2>/dev/null; then
    "$PY" -m pip install -r requirements-train.txt -i "$INDEX"
    cp requirements-train.txt .venv/.deps_ok
  fi
fi
export PYTHONPATH="$HERE/src"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
"$PY" -c 'import mlx.core as mx; i = mx.device_info(mx.gpu); print("[pkg] GPU:", i.get("device_name"), "| 内存", round(i.get("memory_size", 0) / 2**30), "GiB | 推荐工作集", round(i.get("max_recommended_working_set_size", 0) / 2**30), "GiB")'

if [ "${1:-}" = "--smoke" ]; then
  shift
  OUT="run_smoke/$(date +%Y%m%d_%H%M%S)"
  echo "[pkg] 冒烟测试 → $OUT"
  exec "$PY" -m chronos_translate.doc_run --spec recipe.json --max-steps 2 --run-dir "$OUT" "$@"
fi
mkdir -p run
echo "[pkg] 训练开始（可断点续训）。日志：$HERE/run/train.log；监控页：./monitor.sh"
caffeinate -i -s "$PY" -m chronos_translate.doc_run --spec recipe.json "$@" 2>&1 | tee -a run/train.log
"""

MONITOR = r"""#!/bin/bash
# Read-only training monitor: http://127.0.0.1:${PORT:-7870}  (loopback only; there is no login on loopback).
# Remote viewing needs --allow-remote --api-key <key> (see README_zh.md); prefer an SSH tunnel instead.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${CHRONOS_PY:-$HERE/.venv/bin/python}"
export PYTHONPATH="$HERE/src"
export PYTHONNOUSERSITE=1
exec "$PY" -m chronos.monitor.server --runs-dir "$HERE" --host 127.0.0.1 --port "${PORT:-7870}" "$@"
"""


def readme(key, cfg, rec, docmix, size_gb):
    m, tr, x = rec["model_config"], docmix["splits"]["train"], rec["extra"]
    ph = {p["name"]: p for p in rec["phases"]}
    sel = docmix["selection"]
    share = lambda d: "、".join(f"{k} {v:.1%}" for k, v in d.items())  # noqa: E731
    return f"""# {cfg['name']}

Chronos v3 双向翻译（英→中 2/3，中→英 1/3）MLX 训练包 {key}：{cfg['set']}，上下文 {cfg['ctx']}，墙钟预算 {cfg['hours']:g} 小时。
数据、分词器、评测集、源码和依赖清单都在包里，不需要访问原机器。

## 运行

要求：Apple silicon Mac（MLX 只支持 arm64 macOS），Python 3.10–3.12，解压后约 {size_gb:.1f} GB，外加 venv 约 2 GB。

```bash
shasum -a 256 -c {cfg['name']}.tar.gz.sha256   # 先校验压缩包
tar -xzf {cfg['name']}.tar.gz && cd {cfg['name']}
./launch.sh --verify      # 校验包内文件（SHA256SUMS；首次 ./launch.sh 也会自动校验）
./launch.sh --smoke       # 冒烟：每阶段 2 步 + 小评测，写到 run_smoke/，不影响正式 run/
./launch.sh               # 正式训练；中断后再次执行同一命令即从 run/checkpoints/latest.state.pkl 续训
./monitor.sh              # 另开终端：只读监控页 http://127.0.0.1:7870
```

- 首次运行会在 `.venv/` 建虚拟环境，并按 `requirements-train.txt`（全部固定版本）从清华镜像安装依赖。换镜像用 `PIP_INDEX_URL=…`；用已有环境用 `CHRONOS_PY=/path/to/python ./launch.sh`（不会再 pip 安装）。
- `launch.sh` 用 `caffeinate` 防止睡眠，日志同时写到 `run/train.log`。
- 监控页只监听 127.0.0.1，没有登录。远程查看请用 SSH 隧道：`ssh -L 7870:127.0.0.1:7870 <训练机>`。

## 数据（{cfg['set']}，构建于 {docmix.get('created_utc')}）

- 随机抽取 {sel['rows_selected'] / sel['train_rows_total']:.1%} 的训练行：{sel['rows_selected']:,} / {sel['train_rows_total']:,}。按 256 行连续块抽取，保留上下文顺序。
- 科学类加权：每块按关键词命中分档，高 ×2.0、中 ×1.2、其余 ×1.0。高档行占比从全量的 {sel['tier_row_share_all'][0]:.1%} 升到 {sel['tier_row_share_selected'][0]:.1%}；每行平均命中从 {sel['mean_hits_per_row_all']:.3f} 升到 {sel['mean_hits_per_row_selected']:.3f}。
- SFT 记录 {tr['records']:,} 条，n-in-1 合并占比（按记录数）：{share(tr['size_share_records'])}；方向：{share(tr['dir_share_records'])}。
- 合并来源：{tr['ordered_records']:,} 条来自原文有序的相邻句（连贯篇章），{tr['random_concat_records']:,} 条来自已打乱的来源（随机拼接，只练长度不练篇章）。
- 记录长度（token）：p50 {cfg['lengths']['p50']}，p95 {cfg['lengths']['p95']}，p99 {cfg['lengths']['p99']}，最长 {cfg['lengths']['max']}；上限 {cfg['ctx']}。token 占比按长度：{share(tr['token_share_by_len'])}。
- 预训练 {tr['pretrain_tokens']:,} token；SFT {tr['sft_tokens']:,} token，其中被监督的译文 {tr['sft_supervised_tokens']:,}。
- 长度采样：SFT 每个 batch 先按 1/行数(长度) 抽目标长度，再取同长度桶。这样每条记录在每个长度段被看到的次数相同，长记录不会因为一批只装几行而少练。

## 模型与参数（planner 按实测长度给出）

- 总参数 {x['params']['total_params']:,}，激活 {x['params']['active_params']:,}（循环各轮专家并集上界 {x['params']['active_union_upper']:,}）。
- H={m['hidden_size']}，物理层 {m['num_hidden_layers']}（prelude {m['prelude_layers']} + {m['recurrent_groups']} 个循环组 + coda {m['coda_layers']}），循环深度 {m['recurrent_depth']}（最少 {m['min_recurrent_depth']}，ACT 自适应），虚拟层 {x['params']['virtual_layers']}；heads {m['num_attention_heads']}/{m['num_key_value_heads']}。
- MoE：{m['num_experts']} 个专家，top-{m['num_experts_per_tok']} + {m['num_shared_experts']} 个共享专家，专家宽度 {m['moe_intermediate_size']}。
- 注意力：MLA/SWA 混合，`{m['recurrent_attention_mode']}` KV，kv_latent {m['kv_latent_dim']}，rope {m['rope_dim']}，SWA 窗口 {m['sliding_window_size']}；cluster locality λ={m['lambda_cluster_locality']}，窗口 {m['cluster_locality_window']}。
- 预训练：峰值 LR {ph['pretrain']['peak_lr']}，每步 {ph['pretrain']['tokens_per_step']:,} token，序列 {ph['pretrain']['seq_len']}。SFT：峰值 LR {ph['sft']['peak_lr']}，每步 {ph['sft']['tokens_per_step']:,} 个记录 token。
- 显存：启动时实测两步峰值，自动选 micro-batch，目标不超过推荐工作集的 70%；每步 token 数由梯度累积补足。结果写在 `run/train.log` 的 `[doc] micro-batch fit` 行。

## 时间与轮数

- 预算 {cfg['hours']:g} 小时，其中 {rec['eval_reserve_fraction']:.0%} 留给最终评测，其余预训练和 SFT 各一半。学习率按墙钟 warmup + cosine。
- 目标轮数：预训练 {TARGET_PASSES['pretrain']:g} 轮、SFT {TARGET_PASSES['sft']:g} 轮（当前 72h 计划约 2.8 / 1.7 轮，各加 1 轮后取整）。阶段先到目标轮数就提前结束，剩余时间留给下一阶段；吞吐不够时按时间结束。
- 训练 10 分钟后日志会打出 `[doc] … passes fit the phase budget`：按实测吞吐预计本阶段能跑几轮。每次验证还会在 metrics.jsonl 写一条 `data_passes`。

## 评测

- 验证 CE；SFT 期间每 {rec['validate_every_s'] / 3600:g} 小时在 FLORES+ devtest 前 {rec['bleu_samples']} 句上评测英→中、中→英 BLEU/chrF。
- 篇章级：FLORES+ devtest 按文章拼接（双向），WMT24++ en→zh 按文档拼接；放不进上下文的文档跳过，评测条数写在 `*_n`。
- 结束时：FLORES+ devtest {rec['final_eval_samples']} 句，加上篇章级评测（{'全部文档' if 'final_doc_eval_docs' not in x else f"每个集合前 {x['final_doc_eval_docs']} 篇"}）。

## 输出

`run/`：metrics.jsonl、train.log、samples.jsonl（评测译文样例）、checkpoints/、`chronos_docmix_pretrain*.pth`、`chronos_docmix_sft*.pth`（`.best` 为验证 CE 最低）。

## 注意

- WMT19、OPUS-100 的许可在数据卡上是 unknown：权重只用于研究，不能对外发布。FLORES+ 是 CC BY-SA 4.0，WMT24++ 见 `data/raw/google__wmt24pp__en-zh_CN/README.md`，两者只用于评测。
- 现有 bundle/serve 推理默认使用英→中提示；中→英提示是 `翻译成英文，只输出译文：\\n`，运行时接入是后续工作。
- 数字来源：数据统计来自 `data/shards/docmix_manifest.json`（实测）；参数量为 meta-device 实测；显存与吞吐需要在训练机上实测。
"""


def _du(root):
    return sum(os.path.getsize(os.path.join(b, f)) for b, _d, fs in os.walk(root) for f in fs)


def build(key, out_root, plan):
    cfg = PACKAGES[key]
    pkg = os.path.join(out_root, cfg["name"])
    if os.path.exists(pkg):
        raise FileExistsError(f"{pkg} exists (remove it first)")
    free = shutil.disk_usage(out_root).free / 2**30
    if free < 8 + 2 * _du(os.path.join(DOCMIX, cfg["set"])) / 2**30:
        raise SystemExit(f"[pkg] only {free:.1f} GiB free")
    tmp = pkg + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)                                       # our own unfinished output
    t0 = time.time()
    src_set = os.path.join(DOCMIX, cfg["set"])
    docmix = json.load(open(os.path.join(src_set, "docmix_manifest.json"), encoding="utf-8"))
    assert docmix["spec"]["ctx_len"] == cfg["ctx"]
    _copytree(src_set, os.path.join(tmp, "data", "shards"))
    _json(os.path.join(tmp, "data", "shards", "shards_manifest.json"), shards_manifest(cfg, docmix))
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(TOKENIZER, name)):
            _copy(os.path.join(TOKENIZER, name), os.path.join(tmp, "data", "tokenizer", name))
    flores = os.path.join(RAW, "facebook__flores__eng-zho")
    for name in FLORES_FILES:
        _copy(os.path.join(RAW, FLORES_REL, name), os.path.join(tmp, "data", "raw", FLORES_REL, name))
    _download_manifest(os.path.join(flores, "download_manifest.json"),
                       os.path.join(tmp, "data", "raw", "facebook__flores__eng-zho", "download_manifest.json"))
    wmt = os.path.join(RAW, "google__wmt24pp__en-zh_CN")
    for name in ("en-zh_CN.jsonl", "README.md"):
        _copy(os.path.join(wmt, name), os.path.join(tmp, "data", "raw", "google__wmt24pp__en-zh_CN", name))
    _download_manifest(os.path.join(wmt, "download_manifest.json"),
                       os.path.join(tmp, "data", "raw", "google__wmt24pp__en-zh_CN", "download_manifest.json"))
    _copytree(os.path.join(REPO, "chronos"), os.path.join(tmp, "src", "chronos"))
    _copytree(os.path.join(DEMO, "chronos_translate"), os.path.join(tmp, "src", "chronos_translate"))
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml"):
        _copy(os.path.join(REPO, name), os.path.join(tmp, "src", name))
    rec = recipe(key, cfg, plan[key], docmix)
    _json(os.path.join(tmp, "recipe.json"), rec)
    _write(os.path.join(tmp, "requirements-train.txt"), REQUIREMENTS)
    _write(os.path.join(tmp, "launch.sh"), LAUNCH, 0o755)
    _write(os.path.join(tmp, "monitor.sh"), MONITOR, 0o755)
    _write(os.path.join(tmp, "README_zh.md"), readme(key, cfg, rec, docmix, _du(tmp) / 1e9))
    _json(os.path.join(tmp, "PACKAGE.json"), {
        "name": cfg["name"], "format": "chronos-docmix-package/1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chronos_version": _version("chronos"), "chronos_translate_version": _version("chronos_translate"), "set": cfg["set"], "ctx_len": cfg["ctx"], "budget_hours": cfg["hours"],
        "tokenizer_fingerprint": docmix.get("tokenizer_fingerprint"),
        "licence_note": "WMT19/OPUS-100 licences unknown: research-only weights; FLORES+ CC BY-SA 4.0 (eval only)"})
    lines = []
    for base, dirs, files in os.walk(tmp):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(base, name), tmp)
            if name not in SKIP and rel != "SHA256SUMS":
                lines.append(f"{sha256_file(os.path.join(base, name))}  {rel}")
    _write(os.path.join(tmp, "SHA256SUMS"), "\n".join(lines) + "\n")
    os.replace(tmp, pkg)
    print(json.dumps({"package": cfg["name"], "files": len(lines), "gb": round(_du(pkg) / 1e9, 3),
                      "s": round(time.time() - t0, 1)}), flush=True)
    return pkg


def main():
    out_root = os.path.abspath(sys.argv[1])
    keys = sys.argv[2:] or list(PACKAGES)
    os.makedirs(out_root, exist_ok=True)
    plan = json.load(open(PLAN, encoding="utf-8"))
    for key in keys:
        build(key, out_root, plan)


if __name__ == "__main__":
    main()
