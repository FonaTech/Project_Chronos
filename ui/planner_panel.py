"""ui/planner_panel.py — Config tab "auto plan" accordion.

The user states a target (total parameters, MoE on/off, expert count, active
parameters *or* active experts, memory-saving level, optional datasets) and
:func:`chronos.model.planner.plan_training` returns a v3 architecture plus
independent hyperparameters for every training stage.  The stage table is
editable; edits are written back to ``config_state["stage_overrides"]``,
which the Train tab merges over the shared values for the selected stage.

The handlers are plain functions so they can be tested without Gradio.
"""
from __future__ import annotations

import json
import os

from chronos.model.planner import (
    MEMORY_LEVELS, STAGE_KEYS, PlanRequest, overrides_from_table, plan_training, stage_table,
)
from ui.gradio_compat import gr
from ui.i18n import register_translatable, t
from ui.tabs.tokenizer_tab import parse_paths

ACTIVE_MODES = ("experts", "params")
STAGE_HEADERS = ["stage"] + list(STAGE_KEYS)


def run_plan(target_m, use_moe, num_experts, active_mode, active_value, memory_saving, data_paths,
             ram_gb, token_budget_b, depth, vocab_size, seq_len) -> dict:
    """Build a PlanRequest from UI values (millions / billions) and run the planner."""
    target = float(target_m or 0) * 1e6
    if target <= 0:
        raise ValueError("target")
    use_moe = bool(use_moe)
    value = float(active_value or 0)
    active_params = value * 1e6 if use_moe and active_mode == "params" and value > 0 else None
    active_experts = int(value) if use_moe and active_mode == "experts" and value >= 1 else None
    if use_moe and active_experts and active_experts > int(num_experts or 1):
        raise ValueError("active_experts")
    request = PlanRequest(
        target_total=target, use_moe=use_moe, num_experts=int(num_experts or 1),
        active_params=active_params, active_experts=active_experts,
        memory_saving=memory_saving if memory_saving in MEMORY_LEVELS else "none",
        data_paths=tuple(parse_paths(data_paths or "")),
        ram_gb=float(ram_gb) if ram_gb else None,
        token_budget=float(token_budget_b) * 1e9 if token_budget_b else None,
        recurrent_depth=int(depth or 4),
        vocab_size=int(vocab_size) if vocab_size else None,
        seq_len=int(seq_len) if seq_len else None,
    )
    return plan_training(request)


def plan_summary(plan: dict) -> str:
    """Short Markdown summary; every figure carries its label."""
    arch, scal, mem = plan["report"]["architecture"], plan["report"]["scaling"], plan["report"]["memory"]
    lines = [
        f"**{arch['total_params'] / 1e6:.2f}M** total · **{arch['active_params'] / 1e6:.2f}M** active "
        f"(union ≤ {arch['active_union_upper'] / 1e6:.2f}M) — {arch['params_source']}",
        f"h={arch['hidden_size']} · L={arch['physical_layers']} (groups {arch['recurrent_groups']} × depth "
        f"{arch['recurrent_depth']} → {arch['virtual_layers']} virtual) · heads {arch['heads']} · "
        f"E={arch['experts']} top-{arch['top_k']} width {arch['expert_width']}",
        f"LR {scal['chosen_lr']:g} · batch {scal['chosen_batch_tokens']:,} tokens · "
        f"{scal['tokens']:.3g} tokens ({scal['tokens_source']}) — estimated, {scal['law']}",
        f"micro {mem['micro_tokens']:,} tokens × accum {mem['accumulation_steps']} · training peak ≈ "
        f"{mem['training_peak_gb']} GiB (estimated) of {mem['target_gb']} GiB target · KV {mem['kv_cache_dtype']} · "
        f"router {mem['router_scope']}",
    ]
    lines += [f"- {note}" for note in plan["report"]["notes"]]
    return "\n\n".join(lines)


def apply_stage_rows(cfg: dict, rows) -> dict:
    """Write an edited stage table back into the config (pretrain row also sets the shared values)."""
    cfg = dict(cfg or {})
    if hasattr(rows, "values") and hasattr(rows, "columns"):          # pandas DataFrame
        rows = rows.values.tolist()
    overrides = overrides_from_table(rows)
    cfg["stage_overrides"] = overrides
    cfg.update(overrides.get("pretrain", {}))
    return cfg


def stage_rows(cfg: dict) -> list[list]:
    return stage_table((cfg or {}).get("stage_overrides") or {})[1]


def save_plan(plan: dict, path: str) -> str:
    """Write the full plan (config, per-stage values, MLX long-run phases, report) as JSON."""
    path = os.path.abspath(os.path.expanduser(path or "./out/chronos_plan.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False, default=str)
    return path


def build_planner_panel(config_state, config_result, config_outputs: list) -> None:
    """Render the accordion inside the Config tab and wire it to ``config_state``.

    ``config_result(cfg)`` must return the values for ``config_outputs`` (state,
    JSON view, every editor widget and the estimator panel).
    """
    with gr.Accordion(t("plan.title"), open=False) as acc:
        register_translatable(acc, "plan.title")
        intro = gr.Markdown(t("plan.intro"))
        with gr.Row():
            target = gr.Number(label=t("plan.target"), value=200, precision=2, minimum=0.1)
            use_moe = gr.Checkbox(label=t("plan.use_moe"), value=True)
            experts = gr.Number(label=t("plan.experts"), value=32, precision=0, minimum=1)
            depth = gr.Number(label=t("plan.depth"), value=4, precision=0, minimum=1, maximum=16)
        with gr.Row():
            active_mode = gr.Radio(choices=[(t("plan.mode.experts"), "experts"), (t("plan.mode.params"), "params")],
                                   value="experts", label=t("plan.active_mode"))
            active_value = gr.Number(label=t("plan.active_value"), value=2, precision=2, minimum=0)
            memory = gr.Dropdown(choices=[(t(f"plan.mem.{lvl}"), lvl) for lvl in MEMORY_LEVELS],
                                 value="balanced", label=t("plan.memory"))
        with gr.Row():
            data_paths = gr.Textbox(label=t("plan.data"), lines=2, scale=3,
                                    placeholder="./dataset/shards/\n./dataset/sft_mini_512.jsonl")
            with gr.Column(scale=2):
                ram_gb = gr.Number(label=t("plan.ram"), value=0, precision=1, minimum=0)
                token_budget = gr.Number(label=t("plan.tokens"), value=0, precision=3, minimum=0)
            with gr.Column(scale=2):
                vocab = gr.Number(label=t("plan.vocab"), value=0, precision=0, minimum=0)
                seq_len = gr.Number(label=t("plan.seq_len"), value=0, precision=0, minimum=0)
        with gr.Row():
            plan_btn = gr.Button(t("plan.run"), variant="primary")
            plan_path = gr.Textbox(value="./out/chronos_plan.json", label=t("plan.path"), scale=2)
            save_btn = gr.Button(t("plan.save"), variant="secondary")
        summary = gr.Markdown("")
        table_md = gr.Markdown(t("plan.table_help"))
        table = gr.Dataframe(headers=STAGE_HEADERS, value=[], type="array", interactive=True,
                             label=t("plan.table"), wrap=False, static_columns=[0])
        report = gr.JSON(label=t("plan.report"), value={})
        plan_state = gr.State({})
        for comp, key in ((intro, "plan.intro"), (target, "plan.target"), (use_moe, "plan.use_moe"),
                          (experts, "plan.experts"), (depth, "plan.depth"), (active_mode, "plan.active_mode"),
                          (active_value, "plan.active_value"), (memory, "plan.memory"), (data_paths, "plan.data"),
                          (ram_gb, "plan.ram"), (token_budget, "plan.tokens"), (vocab, "plan.vocab"),
                          (seq_len, "plan.seq_len"), (plan_btn, "plan.run"), (plan_path, "plan.path"),
                          (save_btn, "plan.save"), (table_md, "plan.table_help"), (table, "plan.table"),
                          (report, "plan.report")):
            register_translatable(comp, key)

    def on_plan(*values):
        try:
            plan = run_plan(*values)
        except (ValueError, TypeError, OSError) as exc:
            key = f"plan.err.{exc}" if str(exc) in ("target", "active_experts") else None
            message = t(key) if key else f"❌ {type(exc).__name__}: {exc}"
            return [message, gr.update(), gr.update(), gr.update()] + [gr.update() for _ in config_outputs]
        cfg = plan["config"]
        return [plan_summary(plan), stage_rows(cfg), plan["report"], plan] + config_result(cfg)

    def on_apply(cfg, rows):
        try:
            cfg = apply_stage_rows(cfg, rows)
        except (ValueError, TypeError) as exc:
            return [f"❌ {exc}"] + [gr.update() for _ in config_outputs]
        return [t("plan.applied")] + config_result(cfg)

    def on_save(plan, cfg, path):
        if not plan:
            return t("plan.err.no_plan")
        plan = dict(plan, config=dict(cfg), stage_overrides=dict((cfg or {}).get("stage_overrides") or {}))
        return f"✅ {save_plan(plan, path)}"

    plan_inputs = [target, use_moe, experts, active_mode, active_value, memory, data_paths, ram_gb, token_budget,
                   depth, vocab, seq_len]
    plan_btn.click(fn=on_plan, inputs=plan_inputs, outputs=[summary, table, report, plan_state] + config_outputs)
    # Cell edits apply immediately, so the table never holds un-applied values
    # that a later refresh could discard.
    table.input(fn=on_apply, inputs=[config_state, table], outputs=[summary] + config_outputs)
    save_btn.click(fn=on_save, inputs=[plan_state, config_state, plan_path], outputs=[summary])
    # Presets, loaded configs and shared-widget edits (which keep the pretrain
    # row in sync, see merge_config_inputs) refresh the table.
    config_state.change(fn=stage_rows, inputs=[config_state], outputs=[table])
