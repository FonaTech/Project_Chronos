"""Field schema for the Web UI: every setting the pages render, in one place.

Each field carries its group, type, bounds / choices, the code default and —
where the project has one — a recommended value with its *source*:

* ``code``     – the default in ``ChronosConfig`` / the runtime module;
* ``planner``  – chosen by the adaptive planner (``chronos.model.planner``);
* ``measured`` – backed by a measurement in ``reports/`` (named in ``ref``);
* ``rule``     – derived from the model / data (e.g. "= recurrent_depth").

Labels, hover help and choice names live in ``chronos/webui/i18n/<lang>.json``
under ``field.<key>.label`` / ``field.<key>.help`` / ``choice.<key>.<value>``;
``tests/test_webui.py`` checks every field here has them in all four
languages.  The recommended value is rendered from this schema, never
duplicated in the catalogs, so the two cannot drift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Field:
    key: str
    group: str
    type: str                          # int | float | bool | choice | text | path | multi
    default: object = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple = ()
    recommended: object = None
    source: str | None = None          # code | planner | measured | rule
    ref: str | None = None             # report path / rule id for the recommendation
    advanced: bool = False
    scope: tuple = ("config",)         # which forms show it
    nullable: bool = False             # "auto" / empty allowed

    def as_dict(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if v not in (None, (), False) or k in ("default",)}
        out["choices"] = list(self.choices) if self.choices else None
        out["scope"] = list(self.scope)
        return {k: v for k, v in out.items() if v is not None or k == "default"}


def _f(key, group, type_, default=None, **kw) -> Field:
    for name in ("choices", "scope"):
        if name in kw and isinstance(kw[name], list):
            kw[name] = tuple(kw[name])
    return Field(key, group, type_, default, **kw)


LANGUAGES = ("zh-Hans", "zh-Hant", "en", "ja")
DEFAULT_LANGUAGE = "en"

# Config-page groups, in page order.
CONFIG_GROUPS = ("model", "recurrent", "moe", "attention", "objectives", "memory", "optimizer", "runtime", "rl")
# Every group id (config + other pages); each needs ``group.<id>`` in the catalogs.
GROUPS = CONFIG_GROUPS + ("train", "stage_data", "probe", "inference", "lazy", "speculative", "export", "serve",
                          "tokenizer", "benchmark", "autotune", "planner")

RES = "reports/residency_20261107"
SPEC = "reports/speculative"

CONFIG_FIELDS: tuple[Field, ...] = (
    # ---- model shape -------------------------------------------------------
    _f("hidden_size", "model", "int", 768, min=8, max=16384, step=8),
    _f("num_hidden_layers", "model", "int", 8, min=1, max=128, step=1),
    _f("vocab_size", "model", "int", 6400, min=32, max=200000, step=32,
       recommended="tokenizer", source="rule", ref="vocab_matches_tokenizer"),
    _f("max_position_embeddings", "model", "int", 32768, min=128, max=1048576, step=128, advanced=True),
    _f("tie_word_embeddings", "model", "bool", True, recommended=True, source="code"),
    _f("dropout", "model", "float", 0.1, min=0.0, max=0.5, step=0.01, advanced=True),
    _f("rope_theta", "model", "float", 1e6, min=1e3, max=1e8, advanced=True),
    _f("dtype", "model", "choice", "auto", choices=["auto", "float32", "bfloat16", "float16"],
       recommended="auto", source="code"),
    _f("modalities", "model", "text", "", advanced=True),
    # ---- recurrent depth ---------------------------------------------------
    _f("recurrent_enabled", "recurrent", "bool", True, recommended=True, source="code"),
    _f("recurrent_groups", "recurrent", "int", 8, min=1, max=128, step=1, nullable=True),
    _f("prelude_layers", "recurrent", "int", 0, min=0, max=16, step=1, advanced=True),
    _f("coda_layers", "recurrent", "int", 0, min=0, max=16, step=1, advanced=True),
    _f("recurrent_depth", "recurrent", "int", 1, min=1, max=16, step=1, recommended=4, source="planner"),
    _f("min_recurrent_depth", "recurrent", "int", 1, min=1, max=16, step=1),
    _f("max_inference_depth", "recurrent", "int", 1, min=1, max=32, step=1,
       recommended="recurrent_depth", source="rule", ref="equals_recurrent_depth"),
    _f("recurrent_attention_mode", "recurrent", "choice", "shared_first",
       choices=["shared_first", "per_iteration", "first_only"], recommended="shared_first", source="code"),
    _f("adaptive_halting", "recurrent", "bool", False),
    _f("halting_threshold", "recurrent", "float", 0.99, min=0.5, max=1.0, step=0.001),
    _f("depth_schedule", "recurrent", "choice", "fixed_then_adaptive",
       choices=["fixed_then_adaptive", "fixed", "adaptive"], recommended="fixed_then_adaptive", source="code"),
    _f("depth_schedule_warmup_fraction", "recurrent", "float", 0.25, min=0.0, max=1.0, step=0.01),
    _f("recurrent_strategy", "recurrent", "choice", "scheduled", choices=["fixed", "scheduled", "adaptive"],
       advanced=True),
    _f("recurrent_depth_jitter", "recurrent", "float", 0.0, min=0.0, max=1.0, step=0.05, advanced=True),
    _f("ponder_loss_coef", "recurrent", "float", 0.01, min=0.0, max=1.0, step=0.001),
    _f("recurrent_distill_coef", "recurrent", "float", 0.05, min=0.0, max=1.0, step=0.01),
    _f("recurrent_stability_coef", "recurrent", "float", 0.001, min=0.0, max=1.0, step=0.001),
    _f("draft_exit_kl_coef", "recurrent", "float", 0.0, min=0.0, max=1.0, step=0.01, advanced=True),
    _f("draft_exit_depth", "recurrent", "int", 1, min=1, max=16, step=1, advanced=True),
    # ---- mixture of experts ------------------------------------------------
    _f("num_experts", "moe", "int", 4, min=1, max=1024, step=1),
    _f("num_experts_per_tok", "moe", "int", 1, min=1, max=32, step=1, recommended=2, source="planner"),
    _f("num_shared_experts", "moe", "int", 1, min=0, max=32, step=1, recommended=1, source="code"),
    _f("moe_intermediate_size", "moe", "int", 0, min=0, max=65536, step=8, nullable=True),
    _f("router_scope", "moe", "choice", "per_iteration", choices=["per_iteration", "shared"],
       recommended="per_iteration", source="code"),
    _f("lookahead_steps", "moe", "int", 2, min=0, max=16, step=1, recommended=2, source="code"),
    _f("layer_aware_cache", "moe", "bool", False),
    _f("prefetch_depth", "moe", "int", 2, min=0, max=64, step=1, recommended=2, source="code"),
    _f("fallback_mask_prob", "moe", "float", 0.08, min=0.0, max=1.0, step=0.01, recommended=0.08, source="code"),
    # ---- attention / KV ----------------------------------------------------
    _f("num_attention_heads", "attention", "int", 8, min=1, max=256, step=1),
    _f("num_key_value_heads", "attention", "int", 4, min=1, max=256, step=1),
    _f("use_hybrid_attention", "attention", "bool", True, recommended=True, source="code"),
    _f("kv_latent_dim", "attention", "int", 64, min=4, max=1024, step=4),
    _f("rope_dim", "attention", "int", 32, min=2, max=256, step=2),
    _f("sliding_window_size", "attention", "int", 2048, min=8, max=65536, step=8),
    _f("kv_cache_dtype", "attention", "choice", "auto", choices=["auto", "int8"], recommended="auto", source="code"),
    _f("kv_quant_group_size", "attention", "int", 32, min=8, max=256, step=8, advanced=True),
    _f("mla_decode", "attention", "choice", "expanded", choices=["expanded", "absorbed", "auto"],
       recommended="expanded", source="code", ref="mla_auto_gap"),
    _f("mla_cache_rope", "attention", "bool", False, advanced=True),
    # ---- training objectives -----------------------------------------------
    _f("lambda_balance", "objectives", "float", 5e-4, min=0.0, max=1.0, step=1e-4, recommended=5e-4, source="code"),
    _f("lambda_temporal", "objectives", "float", 3e-3, min=0.0, max=1.0, step=1e-4, recommended=3e-3, source="code"),
    _f("lambda_lookahead", "objectives", "float", 0.1, min=0.0, max=10.0, step=0.01, recommended=0.1, source="code"),
    _f("lambda_lookahead_topk", "objectives", "float", 0.15, min=0.0, max=10.0, step=0.01, advanced=True),
    _f("lambda_lookahead_union", "objectives", "float", 0.05, min=0.0, max=10.0, step=0.01, advanced=True),
    _f("lambda_router_locality", "objectives", "float", 0.02, min=0.0, max=10.0, step=0.01, advanced=True),
    _f("lambda_router_anchor", "objectives", "float", 0.0, min=0.0, max=10.0, step=0.01),
    _f("lambda_cluster_locality", "objectives", "float", 0.3, min=0.0, max=10.0, step=0.01,
       recommended=0.3, source="measured", ref=f"{RES}/sweep"),
    _f("cluster_locality_clusters", "objectives", "int", 0, min=0, max=1024, step=1,
       recommended=0, source="rule", ref="clusters_auto"),
    _f("cluster_locality_window", "objectives", "int", 32, min=0, max=65536, step=1,
       recommended="data", source="rule", ref="window_median_record"),
    # ---- memory / lazy experts ---------------------------------------------
    _f("vram_budget_gb", "memory", "float", 4.0, min=0.01, max=256.0, step=0.01),
    _f("pinned_memory_max_fraction", "memory", "float", 0.25, min=0.01, max=0.95, step=0.01),
    _f("max_resident_experts", "memory", "int", None, min=1, max=4096, step=1, nullable=True),
    _f("ram_budget_mb", "memory", "float", None, min=1.0, max=1048576.0, step=1.0, nullable=True),
    _f("storage_format", "memory", "choice", "safetensors", choices=["safetensors", "pt"],
       recommended="safetensors", source="code"),
    _f("training_paging", "memory", "bool", False),
    _f("optimizer_paging", "memory", "bool", False),
    _f("page_size_mb", "memory", "int", 64, min=1, max=4096, step=1, advanced=True),
    _f("optimizer_page_dir", "memory", "path", "./optimizer_pages", advanced=True),
    _f("activation_checkpointing", "memory", "bool", False),
    # ---- optimizer ---------------------------------------------------------
    _f("learning_rate", "optimizer", "float", 5e-4, min=1e-7, max=1.0, step=1e-5,
       recommended="planner", source="planner"),
    _f("batch_size", "optimizer", "int", 16, min=1, max=512, step=1),
    _f("accumulation_steps", "optimizer", "int", 8, min=1, max=256, step=1),
    _f("max_seq_len", "optimizer", "int", 512, min=8, max=131072, step=8),
    _f("epochs", "optimizer", "int", 2, min=1, max=1000, step=1),
    _f("max_steps", "optimizer", "int", 0, min=0, max=10_000_000, step=1, recommended=0, source="code"),
    _f("weight_decay", "optimizer", "float", 0.01, min=0.0, max=1.0, step=0.001),
    _f("grad_clip", "optimizer", "float", 1.0, min=0.0, max=100.0, step=0.1),
    # ---- runtime -----------------------------------------------------------
    _f("save_dir", "runtime", "path", "./out"),
    _f("tokenizer_path", "runtime", "path", ""),
    _f("save_interval", "runtime", "int", 1000, min=1, max=1_000_000, step=1),
    _f("log_interval", "runtime", "int", 10, min=1, max=100000, step=1),
    _f("cpu_threads", "runtime", "text", "auto", recommended="auto", source="code"),
    _f("cpu_budget_percent", "runtime", "float", 100, min=10, max=100, step=1),
    _f("num_workers", "runtime", "text", "auto", recommended="auto", source="code"),
    # ---- RL / preference ---------------------------------------------------
    _f("beta", "rl", "float", 0.1, min=0.0, max=10.0, step=0.01),
    _f("alpha", "rl", "float", 0.7, min=0.0, max=1.0, step=0.01),
    _f("lambda_or", "rl", "float", 0.1, min=0.0, max=10.0, step=0.01),
    _f("reward_spec", "rl", "text", "toy"),
    _f("max_gen_len", "rl", "int", 24, min=1, max=4096, step=1),
    _f("num_generations", "rl", "int", 4, min=1, max=64, step=1),
    _f("temperature", "rl", "float", 1.0, min=0.0, max=2.0, step=0.05),
)

TRAIN = ("train",)
PROBE = ("train", "probe")
INFER = ("inference",)

FORM_FIELDS: tuple[Field, ...] = (
    # ---- training job ------------------------------------------------------
    _f("train.backend", "train", "choice", "auto", choices=["auto", "mlx", "cuda", "xpu", "mps", "cpu"],
       recommended="auto", source="code", scope=TRAIN),
    _f("train.stages", "train", "multi", ["pretrain"], choices=["pretrain", "sft", "dpo", "orpo", "grpo", "distill"],
       scope=TRAIN),
    _f("train.data_path", "train", "path", "", scope=TRAIN),
    # Later stages need their own data format (conversations, preference pairs, prompts); the Train
    # page shows the row of each selected stage after the first. Empty = reuse the first stage's data.
    *(_f(f"train.data_{m}", "stage_data", "path", "", scope=TRAIN, nullable=True)
      for m in ("sft", "dpo", "orpo", "grpo", "distill")),
    _f("train.init_weight", "train", "path", "", scope=TRAIN, nullable=True),
    _f("train.teacher_path", "train", "path", "", scope=TRAIN, nullable=True),
    _f("train.max_steps", "train", "int", 0, min=0, max=10_000_000, step=1, scope=TRAIN),
    _f("train.val_ratio", "train", "float", 0.05, min=0.0, max=0.5, step=0.01, recommended=0.05,
       source="code", scope=TRAIN),
    _f("train.sample_prompt", "train", "text", "", scope=TRAIN),
    # ---- lazy-expert probe after each validation ---------------------------
    _f("probe.enabled", "probe", "bool", True, recommended=True, source="code", scope=PROBE),
    _f("probe.backends", "probe", "multi", ["auto"], choices=["auto", "mlx", "cuda", "xpu", "mps", "cpu"],
       recommended=["auto"], source="code", scope=PROBE),
    _f("probe.paging", "probe", "multi", ["auto", "expert"], choices=["auto", "cluster", "expert"],
       recommended=["auto", "expert"], source="code", scope=PROBE),
    _f("probe.residency", "probe", "multi", ["planned"], choices=["planned", "window"],
       recommended=["planned"], source="measured", ref=f"{RES}/bench_residency_50m_s13996_flores16_cpu.json",
       scope=PROBE),
    _f("probe.ram_ratios", "probe", "multi", [0.5, 1.0],
       choices=[0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9, 1.0], scope=PROBE),
    _f("probe.miss_policies", "probe", "multi", ["sync_on_demand"],
       choices=["sync_on_demand", "on_demand", "fallback_diagnostic"], recommended=["sync_on_demand"],
       source="rule", ref="sync_is_exact", scope=PROBE),
    _f("probe.residency_decay", "probe", "float", 0.7, min=0.0, max=0.95, step=0.05, recommended=0.7,
       source="code", scope=PROBE, advanced=True),
    _f("probe.decode_wait_s", "probe", "float", 0.012, min=0.0, max=1.0, step=0.001, recommended=0.012,
       source="code", scope=PROBE, advanced=True),
    _f("probe.verify_wait_s", "probe", "float", 0.02, min=0.0, max=1.0, step=0.001, recommended=0.02,
       source="code", scope=PROBE, advanced=True),
    _f("probe.speculative", "probe", "multi", [False], choices=[False, True], scope=PROBE),
    _f("probe.draft_depth", "probe", "int", 1, min=1, max=16, step=1, scope=PROBE, advanced=True),
    _f("probe.gamma", "probe", "int", 4, min=1, max=8, step=1, scope=PROBE, advanced=True),
    _f("probe.max_new_tokens", "probe", "int", 32, min=1, max=512, step=1, recommended=32, source="code",
       scope=PROBE),
    _f("probe.max_variants", "probe", "int", 12, min=1, max=256, step=1, recommended=12, source="code",
       scope=PROBE),
    # ---- inference ---------------------------------------------------------
    _f("infer.model_path", "inference", "path", "", scope=INFER),
    _f("infer.backend", "inference", "choice", "auto", choices=["auto", "mlx", "cuda", "xpu", "mps", "cpu"],
       recommended="auto", source="code", scope=INFER),
    _f("infer.mode", "inference", "choice", "compare", choices=["compare", "offload", "fullload"], scope=INFER),
    _f("infer.max_tokens", "inference", "int", 128, min=1, max=4096, step=1, scope=INFER),
    _f("infer.temperature", "inference", "float", 0.85, min=0.0, max=2.0, step=0.05, scope=INFER),
    _f("infer.raw_prompt", "inference", "bool", False, scope=INFER),
    _f("infer.ram_load_ratio", "lazy", "float", 1.0, min=0.1, max=1.25, step=0.01, scope=INFER),
    _f("infer.sweep_ram_ratios", "lazy", "bool", False, scope=INFER),
    _f("infer.miss_policy", "lazy", "choice", "sync_on_demand",
       choices=["on_demand", "sync_on_demand", "fallback_diagnostic"], recommended="sync_on_demand",
       source="rule", ref="sync_is_exact", scope=INFER),
    _f("infer.paging", "lazy", "choice", "auto", choices=["auto", "cluster", "expert"], recommended="auto",
       source="code", scope=INFER),
    _f("infer.residency_planner", "lazy", "bool", True, recommended=True, source="measured",
       ref=f"{RES}/bench_residency_50m_s13996_flores16_cpu.json", scope=INFER),
    _f("infer.residency_decay", "lazy", "float", 0.7, min=0.0, max=0.95, step=0.05, recommended=0.7,
       source="code", scope=INFER, advanced=True),
    _f("infer.decode_wait_s", "lazy", "float", 0.012, min=0.0, max=1.0, step=0.001, recommended=0.012,
       source="code", scope=INFER, advanced=True),
    _f("infer.verify_wait_s", "lazy", "float", 0.02, min=0.0, max=1.0, step=0.001, recommended=0.02,
       source="code", scope=INFER, advanced=True),
    _f("infer.speculative", "speculative", "bool", False, recommended=False, source="measured",
       ref=SPEC, scope=INFER),
    _f("infer.draft_depth", "speculative", "int", 1, min=1, max=8, step=1, scope=INFER),
    _f("infer.gamma", "speculative", "int", 4, min=1, max=8, step=1, scope=INFER),
    # ---- export ------------------------------------------------------------
    _f("export.model_path", "export", "path", "", scope=("export",)),
    _f("export.output_dir", "export", "path", "./exports/chronos", scope=("export",)),
    _f("export.formats", "export", "multi", ["fp16-safetensors", "q8_0-gguf"],
       choices=["fp16-safetensors", "q8_0-safetensors", "fp16-gguf", "q8_0-gguf", "chronos-native-gguf"],
       scope=("export",)),
    _f("export.config_path", "export", "path", "", scope=("export",), nullable=True, advanced=True),
    _f("export.expert_cache_dir", "export", "path", "", scope=("export",), nullable=True, advanced=True),
    _f("export.copy_cache", "export", "bool", True, scope=("export",)),
    _f("export.auto_cluster", "export", "bool", True, recommended=True, source="code", scope=("export",)),
    _f("export.calibration_data_path", "export", "path", "", scope=("export",), nullable=True),
    _f("export.cluster_device", "export", "choice", "cpu", choices=["cpu", "mps", "cuda"], scope=("export",),
       advanced=True),
    _f("export.cluster_max_batches", "export", "int", 50, min=1, max=10000, step=1, scope=("export",),
       advanced=True),
    _f("export.cluster_batch_size", "export", "int", 4, min=1, max=256, step=1, scope=("export",), advanced=True),
    _f("export.cluster_max_seq_len", "export", "int", 256, min=8, max=8192, step=8, scope=("export",),
       advanced=True),
    # ---- serve (Ollama / OpenAI protocol) ----------------------------------
    _f("serve.models_dir", "serve", "path", "./exports", scope=("serve",)),
    _f("serve.host", "serve", "text", "127.0.0.1", recommended="127.0.0.1", source="rule", ref="loopback",
       scope=("serve",)),
    _f("serve.port", "serve", "int", 11435, min=1024, max=65535, step=1, scope=("serve",)),
    _f("serve.device", "serve", "choice", "auto", choices=["auto", "mlx", "cuda", "mps", "cpu"], scope=("serve",)),
    _f("serve.dtype", "serve", "choice", "auto", choices=["auto", "float32", "bfloat16", "float16"],
       scope=("serve",)),
    _f("serve.lazy_experts", "serve", "bool", True, recommended=True, source="code", scope=("serve",)),
    _f("serve.ram_expert_fraction", "serve", "float", 0.5, min=0.05, max=1.0, step=0.05, scope=("serve",)),
    _f("serve.paging", "serve", "choice", "auto", choices=["auto", "cluster", "expert"], recommended="auto",
       source="code", scope=("serve",)),
    _f("serve.kv_cache_dtype", "serve", "choice", "", choices=["", "auto", "int8"], scope=("serve",),
       advanced=True),
    _f("serve.mla_decode", "serve", "choice", "", choices=["", "expanded", "absorbed", "auto"], scope=("serve",),
       advanced=True),
    _f("serve.router_scope", "serve", "choice", "", choices=["", "per_iteration", "shared"], scope=("serve",),
       advanced=True),
    _f("serve.mla_cache_rope", "serve", "bool", False, scope=("serve",), advanced=True),
    _f("serve.max_resident_experts", "serve", "int", None, min=1, max=4096, step=1, nullable=True,
       scope=("serve",), advanced=True),
    _f("serve.ram_budget_mb", "serve", "float", None, min=1.0, max=1048576.0, step=1.0, nullable=True,
       scope=("serve",), advanced=True),
    _f("serve.allow_remote", "serve", "bool", False, recommended=False, source="rule", ref="loopback",
       scope=("serve",)),
    # ---- tokenizer builder -------------------------------------------------
    _f("tok.paths", "tokenizer", "text", "", scope=("tokenizer",)),
    _f("tok.out_dir", "tokenizer", "path", "./out/tokenizer", scope=("tokenizer",)),
    _f("tok.vocab_size", "tokenizer", "int", 16384, min=256, max=65535, step=64, scope=("tokenizer",)),
    _f("tok.per_source", "tokenizer", "int", 400_000, min=1000, max=100_000_000, step=1000, scope=("tokenizer",)),
    _f("tok.temperature", "tokenizer", "float", 0.5, min=0.1, max=1.0, step=0.05, recommended=0.5, source="code",
       scope=("tokenizer",)),
    _f("tok.fields", "tokenizer", "text", "", scope=("tokenizer",), advanced=True),
    # ---- auto-tune ---------------------------------------------------------
    _f("tune.data_path", "autotune", "path", "", scope=("autotune",)),
    _f("tune.n_trials", "autotune", "int", 20, min=2, max=200, step=1, scope=("autotune",)),
    _f("tune.probe_steps", "autotune", "int", 80, min=10, max=1000, step=10, scope=("autotune",)),
    _f("tune.search", "autotune", "multi", ["lambda_balance", "lambda_temporal", "lr"],
       choices=["lambda_balance", "lambda_temporal", "lambda_lookahead", "lookahead_steps", "lr", "hidden_size",
                "num_experts", "num_shared_experts", "kv_latent_dim"], scope=("autotune",)),
    # ---- planner -----------------------------------------------------------
    _f("plan.target_m", "planner", "float", 50.0, min=0.1, max=100000.0, step=0.1, scope=("planner",)),
    _f("plan.use_moe", "planner", "bool", True, scope=("planner",)),
    _f("plan.num_experts", "planner", "int", 32, min=1, max=1024, step=1, scope=("planner",)),
    _f("plan.active_mode", "planner", "choice", "experts", choices=["experts", "params"], scope=("planner",)),
    _f("plan.active_value", "planner", "float", 2, min=0, max=100000, step=1, scope=("planner",)),
    _f("plan.memory_saving", "planner", "choice", "none", choices=["none", "balanced", "aggressive", "extreme"],
       scope=("planner",)),
    _f("plan.data_paths", "planner", "text", "", scope=("planner",)),
    _f("plan.ram_gb", "planner", "float", None, min=1, max=4096, step=1, nullable=True, scope=("planner",)),
    _f("plan.token_budget_b", "planner", "float", None, min=0.001, max=100000, step=0.1, nullable=True,
       scope=("planner",)),
    _f("plan.depth", "planner", "int", 4, min=1, max=16, step=1, scope=("planner",)),
    _f("plan.vocab_size", "planner", "int", None, min=256, max=200000, step=32, nullable=True, scope=("planner",)),
    _f("plan.seq_len", "planner", "int", None, min=16, max=131072, step=16, nullable=True, scope=("planner",)),
)

ALL_FIELDS: tuple[Field, ...] = CONFIG_FIELDS + FORM_FIELDS
FIELDS = {f.key: f for f in ALL_FIELDS}
# Recommendation ids rendered as text (``rule.<ref>`` in the catalogs).
RULE_REFS = tuple(sorted({f.ref for f in ALL_FIELDS if f.source == "rule" and f.ref}))
SOURCES = ("code", "planner", "measured", "rule")


# Technical identifiers shown verbatim in every language (backend, dtype, file-format and
# stage names); every other choice value needs ``choice.<field>.<value>`` in each catalog.
LITERAL_CHOICES = frozenset({
    "mlx", "cuda", "xpu", "mps", "cpu", "float32", "bfloat16", "float16", "int8", "safetensors", "pt",
    "fp16-safetensors", "q8_0-safetensors", "fp16-gguf", "q8_0-gguf", "chronos-native-gguf",
    "pretrain", "sft", "dpo", "orpo", "grpo", "distill", "shared_first", "per_iteration", "first_only",
    "lambda_balance", "lambda_temporal", "lambda_lookahead", "lookahead_steps", "lr", "hidden_size",
    "num_experts", "num_shared_experts", "kv_latent_dim", "sync_on_demand", "on_demand", "fallback_diagnostic",
})


def choice_keys() -> list[str]:
    """``choice.<value>`` keys every catalog must define.  Values mean the same thing in every
    field that offers them (``auto``, ``cluster``, ``fixed`` …), so they are translated once;
    booleans, numbers, blanks and :data:`LITERAL_CHOICES` are shown verbatim."""
    out = []
    for f in ALL_FIELDS:
        if f.type in ("choice", "multi"):
            for value in f.choices:
                if (isinstance(value, bool) or value == "" or isinstance(value, (int, float))
                        or value in LITERAL_CHOICES):
                    continue
                out.append(f"choice.{value}")
    return list(dict.fromkeys(out))


def required_keys() -> set[str]:
    keys = {f"group.{g}" for g in GROUPS} | {f"source.{s}" for s in SOURCES} | {f"rule.{r}" for r in RULE_REFS}
    for f in ALL_FIELDS:
        keys.add(f"field.{f.key}.label")
        keys.add(f"field.{f.key}.help")
    keys.update(choice_keys())
    return keys


# What the browse button next to a path field opens: (kind, multiple). ``kind`` is a filter of
# ``chronos.workbench.files.KINDS``; ``multiple`` fields hold one path per line and several datasets
# are merged into one JSONL before training or calibration reads them.
PICK = {
    "optimizer_page_dir": ("dir", False), "save_dir": ("dir", False), "tokenizer_path": ("dir", False),
    "train.data_path": ("data", True),
    **{f"train.data_{m}": ("data", True) for m in ("sft", "dpo", "orpo", "grpo", "distill")},
    "train.init_weight": ("checkpoint", False), "train.teacher_path": ("checkpoint", False),
    "infer.model_path": ("checkpoint", False),
    "export.model_path": ("checkpoint", False), "export.output_dir": ("dir", False),
    "export.config_path": ("config", False), "export.expert_cache_dir": ("dir", False),
    "export.calibration_data_path": ("data", True),
    "serve.models_dir": ("dir", False),
    "tok.paths": ("data", True), "tok.out_dir": ("dir", False),
    "tune.data_path": ("data", True),
    "plan.data_paths": ("data", True),
}
MULTI_PATH = frozenset(k for k, (_kind, many) in PICK.items() if many)
# Multi-dataset fields whose selection is merged into one JSONL (training stages, export calibration).
# The tokenizer samples each source on its own and the planner measures each, so they are not merged.
MERGE_PATH = frozenset({"train.data_path", "export.calibration_data_path"}
                       | {f"train.data_{m}" for m in ("sft", "dpo", "orpo", "grpo", "distill")})
MAX_TEXT = 4096
MAX_MULTI_TEXT = 65536


def schema_payload() -> dict:
    fields = []
    for f in ALL_FIELDS:
        item = f.as_dict()
        if f.key in PICK:
            item["pick"] = {"kind": PICK[f.key][0], "multiple": PICK[f.key][1], "merge": f.key in MERGE_PATH}
        fields.append(item)
    return {"version": 1, "languages": list(LANGUAGES), "default_language": DEFAULT_LANGUAGE,
            "config_groups": list(CONFIG_GROUPS), "groups": list(GROUPS), "fields": fields}


def config_defaults() -> dict:
    return {f.key: f.default for f in CONFIG_FIELDS}


def coerce(key: str, value):
    """Validate one value against its field; raises ``ValueError(key)``."""
    f = FIELDS[key]
    if value is None or value == "":
        if f.nullable or f.type in ("text", "path") or f.default in (None, ""):
            return None if f.type not in ("text", "path") else ""
        raise ValueError(key)
    if f.type == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(key)
    if f.type in ("int", "float"):
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(key) from None
        if num != num or (f.min is not None and num < f.min) or (f.max is not None and num > f.max):
            raise ValueError(key)
        if f.type == "int":
            if num != int(num):
                raise ValueError(key)
            return int(num)
        return num
    if f.type == "choice":
        if value not in f.choices:
            raise ValueError(key)
        return value
    if f.type == "multi":
        items = value if isinstance(value, list) else [value]
        if not items or any(v not in f.choices for v in items):
            raise ValueError(key)
        return list(dict.fromkeys(items))
    limit = MAX_MULTI_TEXT if key in MULTI_PATH else MAX_TEXT
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValueError(key)
    return value


def coerce_config(values: dict) -> tuple[dict, list[str]]:
    """Model/runtime config from the Config page: ``(clean, bad_keys)``; unknown keys are dropped."""
    clean, bad = {}, []
    for key, value in (values or {}).items():
        if key not in FIELDS or FIELDS[key] not in CONFIG_FIELDS:
            continue
        try:
            clean[key] = coerce(key, value)
        except ValueError:
            bad.append(key)
    return clean, bad
