"""
ui/tabs/inference_tab.py — legacy Gradio inference tab (runners: chronos.workbench.inference)
"""
from ui.gradio_compat import gr
import pandas as pd

from ui.i18n import t, register_translatable
from chronos.trainer.device_utils import configure_cpu_threads, cpu_thread_snapshot  # noqa: F401
from chronos.workbench.inference import (  # noqa: F401  (re-exported for callers and tests)
    _actual_active_expert_count,
    _apply_offload_expert_budget,
    _attach_decode_diagnostics,
    _available_inference_backend_choices,
    _backend_memory_snapshot,
    _best_torch_inference_backend,
    _bounded_offload_expert_budget,
    _build_model_cfg,
    _checkpoint_expert_cache_dir,
    _checkpoint_sidecar_metadata,
    _clone_model_cfg,
    _configure_inference_cpu_threads_if_needed,
    _decode_generation_diagnostics,
    _default_inference_backend_value,
    _encode_prompt,
    _format_inference_notices,
    _format_inference_stats,
    _format_offload_stats,
    _inference_tokenizer_dir,
    _memory_delta,
    _memory_snapshot_gb,
    _next_token_from_logits,
    _normalize_inference_mode,
    _normalize_miss_policy,
    _normalize_ram_load_ratio,
    _populate_mlx_cache_from_reader,
    _populate_mlx_cache_from_state_dict,
    _rebuild_mlx_lazy_cache,
    _replace_mlx_experts_with_placeholders,
    _replace_torch_experts_with_placeholders,
    _resolve_inference_backend,
    _row_from_stats,
    _rows_to_chart_records,
    _run_inference_modes,
    _run_mlx_full_dram,
    _run_mlx_lazy_inference,
    _run_torch_full_dram,
    _run_torch_inference,
    _speculative_options,
    _speculative_row_fields,
    _stage_warning_from_checkpoint,
    _store_torch_dtype,
    _text_diff_summary,
    generate_api,
    INFERENCE_BACKEND_CHOICES,
    INFERENCE_CHART_METRICS,
    INFERENCE_MODE_CHOICES,
    INFERENCE_SWEEP_CHART_METRICS,
    MISS_POLICY_CHOICES,
    RAM_LOAD_RATIO_CHOICES,
    RAM_LOAD_SWEEP_RATIOS,
    TORCH_INFERENCE_PRIORITY,
)


def _empty_inference_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["metric", "mode", "x", "value", "normalized_value", "unit"])


def _rows_to_chart_df(rows: list[dict]) -> pd.DataFrame:
    data = _rows_to_chart_records(rows)
    return pd.DataFrame(data) if data else _empty_inference_df()


def build_inference_tab(config_state: gr.State):
    with gr.Tab(t("tab.inference")) as tab:
        register_translatable(tab, "tab.inference")

        model_path = gr.Textbox(
            label=t("infer.model_path"), placeholder="./out/chronos_512_moe.pth"
        )
        register_translatable(model_path, "infer.model_path")

        prompt = gr.Textbox(
            label=t("infer.prompt"), lines=4,
            placeholder="Explain how Chronos handles expert prefetch during prefill..."
        )
        register_translatable(prompt, "infer.prompt")

        with gr.Row():
            inference_backend = gr.Dropdown(
                choices=_available_inference_backend_choices(),
                value=_default_inference_backend_value(),
                label=t("infer.backend"),
            )
            inference_mode = gr.Dropdown(
                choices=list(INFERENCE_MODE_CHOICES),
                value="compare",
                label=t("infer.mode"),
            )
            max_tokens  = gr.Slider(16, 512, value=128, step=16, label=t("infer.max_tokens"))
            temperature = gr.Slider(0.1, 2.0, value=0.85, step=0.05, label=t("infer.temperature"))
            register_translatable(inference_backend, "infer.backend")
            register_translatable(inference_mode, "infer.mode")
            register_translatable(max_tokens,  "infer.max_tokens")
            register_translatable(temperature, "infer.temperature")
        raw_prompt = gr.Checkbox(value=False, label=t("infer.raw_prompt"))
        register_translatable(raw_prompt, "infer.raw_prompt")
        with gr.Row():
            miss_policy = gr.Dropdown(
                choices=list(MISS_POLICY_CHOICES),
                value="on_demand",
                label=t("infer.miss_policy"),
            )
            ram_load_ratio = gr.Dropdown(
                choices=list(RAM_LOAD_RATIO_CHOICES),
                value="1.00",
                label=t("infer.ram_load_ratio"),
                allow_custom_value=True,
                filterable=True,
            )
            sweep_ratios = gr.Checkbox(
                value=False,
                label=t("infer.sweep_ram_load_ratios"),
            )
            register_translatable(miss_policy, "infer.miss_policy")
            register_translatable(ram_load_ratio, "infer.ram_load_ratio")
            register_translatable(sweep_ratios, "infer.sweep_ram_load_ratios")
        with gr.Row():
            speculative = gr.Checkbox(value=False, label=t("infer.speculative"))
            draft_depth = gr.Slider(1, 8, value=1, step=1, label=t("infer.draft_depth"))
            spec_gamma = gr.Slider(1, 8, value=4, step=1, label=t("infer.spec_gamma"))
            register_translatable(speculative, "infer.speculative")
            register_translatable(draft_depth, "infer.draft_depth")
            register_translatable(spec_gamma, "infer.spec_gamma")
            residency_planner = gr.Checkbox(value=True, label=t("infer.residency_planner"))
            register_translatable(residency_planner, "infer.residency_planner")
        with gr.Row():
            from chronos.lazy_experts import (
                DEFAULT_DECODE_WAIT_S, DEFAULT_RESIDENCY_DECAY, DEFAULT_VERIFY_WAIT_S,
            )
            paging_mode = gr.Dropdown(choices=["auto", "cluster", "expert"], value="auto",
                                      label=t("infer.paging"))
            residency_decay = gr.Slider(0.0, 0.95, value=DEFAULT_RESIDENCY_DECAY, step=0.05,
                                        label=t("infer.residency_decay"))
            decode_wait = gr.Number(value=DEFAULT_DECODE_WAIT_S, minimum=0.0, label=t("infer.decode_wait"))
            verify_wait = gr.Number(value=DEFAULT_VERIFY_WAIT_S, minimum=0.0, label=t("infer.verify_wait"))
            for comp, key in ((paging_mode, "infer.paging"), (residency_decay, "infer.residency_decay"),
                              (decode_wait, "infer.decode_wait"), (verify_wait, "infer.verify_wait")):
                register_translatable(comp, key)
        spec_note = gr.Markdown(t("infer.spec_note"))
        register_translatable(spec_note, "infer.spec_note")

        gen_btn = gr.Button(t("infer.generate"), variant="primary")
        register_translatable(gen_btn, "infer.generate")

        output_box = gr.Textbox(
            label=t("infer.output"), lines=10, interactive=False, visible=False,
        )
        with gr.Row():
            lazy_output_box = gr.Textbox(
                label=t("infer.lazy_output"), lines=10, interactive=False, visible=True,
            )
            full_output_box = gr.Textbox(
                label=t("infer.full_output"), lines=10, interactive=False, visible=True,
            )
        stats_md = gr.Markdown(value=_format_inference_stats([]))
        stats_chart = gr.LinePlot(
            value=_empty_inference_df(),
            x="x", y="normalized_value", color="metric",
            title="Inference mode comparison",
            x_title="Mode / RAM load ratio",
            y_title="Normalized value",
            tooltip=["mode", "metric", "value", "unit"],
            sort=None,
            height=520,
        )
        raw_stats = gr.JSON(label=t("infer.compare_table"), value={}, visible=False)
        register_translatable(output_box, "infer.output")
        register_translatable(lazy_output_box, "infer.lazy_output")
        register_translatable(full_output_box, "infer.full_output")
        register_translatable(raw_stats, "infer.compare_table")

        def _sniff_checkpoint(path: str) -> dict:
            """Inspect a Chronos .pth and return the topology fields encoded
            in the saved tensor shapes. Returns {} if the file is unreadable
            or doesn't look like a Chronos checkpoint.

            Sniffed fields:
                vocab_size, hidden_size, num_hidden_layers, num_experts,
                moe_intermediate_size, num_attention_heads, num_key_value_heads.
            """
            import os as _os
            if not path or not _os.path.exists(path):
                return {}
            try:
                import torch as _t
                sd = _t.load(path, map_location="cpu")
                if not isinstance(sd, dict):
                    return {}
                out = {}

                embed = sd.get("model.embed_tokens.weight")
                if embed is not None:
                    out["vocab_size"], out["hidden_size"] = int(embed.shape[0]), int(embed.shape[1])

                # Count layers via highest layer index in keys.
                layer_idxs = set()
                for k in sd.keys():
                    if k.startswith("model.layers."):
                        try:
                            layer_idxs.add(int(k.split(".")[2]))
                        except (ValueError, IndexError):
                            pass
                if layer_idxs:
                    out["num_hidden_layers"] = max(layer_idxs) + 1

                # Count experts on layer 0.
                expert_idxs = set()
                for k in sd.keys():
                    if k.startswith("model.layers.0.mlp.experts."):
                        try:
                            expert_idxs.add(int(k.split(".")[5]))
                        except (ValueError, IndexError):
                            pass
                if expert_idxs:
                    out["num_experts"] = max(expert_idxs) + 1

                # Per-expert FFN width from gate_proj shape (intermediate, hidden).
                gate = sd.get("model.layers.0.mlp.experts.0.gate_proj.weight")
                if gate is not None:
                    out["moe_intermediate_size"] = int(gate.shape[0])

                # Attention head counts from q/k_proj shapes.
                qw = sd.get("model.layers.0.self_attn.q_proj.weight")
                kw = sd.get("model.layers.0.self_attn.k_proj.weight")
                if qw is not None and kw is not None and out.get("hidden_size"):
                    h = out["hidden_size"]
                    # qw: (n_heads * head_dim, hidden); kw: (n_kv * head_dim, hidden)
                    # head_dim is typically hidden_size // n_heads, but we can't
                    # split q without knowing head_dim. Conservative: assume the
                    # ratio matches MiniMind defaults (head_dim = h // num_heads).
                    # Solve via the kw / qw ratio assuming both share head_dim.
                    out["num_attention_heads_total_dim"] = int(qw.shape[0])
                    out["num_kv_heads_total_dim"] = int(kw.shape[0])

                # MLA attention uses split query and latent KV projections:
                #   q_nope_proj: (n_heads * nope_dim, hidden)
                #   q_rope_proj: (n_heads * rope_dim, hidden)
                #   kv_down_proj: (kv_latent_dim, hidden)
                #   v_proj: (n_kv_heads * head_dim, kv_latent_dim)
                qnope = sd.get("model.layers.0.self_attn.q_nope_proj.weight")
                qrope = sd.get("model.layers.0.self_attn.q_rope_proj.weight")
                kvdown = sd.get("model.layers.0.self_attn.kv_down_proj.weight")
                vproj = sd.get("model.layers.0.self_attn.v_proj.weight")
                if qnope is not None and qrope is not None and out.get("hidden_size"):
                    total_q = int(qnope.shape[0] + qrope.shape[0])
                    h = out["hidden_size"]
                    if total_q > 0 and h % total_q == 0:
                        # For MiniMind/Chronos checkpoints head_dim is
                        # hidden_size / num_attention_heads, so total_q equals
                        # hidden_size and n_heads is recovered by divisibility.
                        pass
                    candidates = [
                        n for n in range(1, out["hidden_size"] + 1)
                        if out["hidden_size"] % n == 0
                        and int(qrope.shape[0]) % n == 0
                        and int(qnope.shape[0]) % n == 0
                    ]
                    if candidates:
                        # Prefer the configured/default 8 when valid; else
                        # choose the largest plausible head count.
                        preferred = 8 if 8 in candidates else max(candidates)
                        head_dim = out["hidden_size"] // preferred
                        rope_dim = int(qrope.shape[0]) // preferred
                        nope_dim = int(qnope.shape[0]) // preferred
                        if rope_dim + nope_dim == head_dim:
                            out["num_attention_heads"] = preferred
                            out["rope_dim"] = rope_dim
                            out["num_attention_heads_total_dim"] = preferred * head_dim
                if kvdown is not None:
                    out["kv_latent_dim"] = int(kvdown.shape[0])
                if vproj is not None and out.get("kv_latent_dim"):
                    head_dim = out.get("hidden_size", 0) // max(out.get("num_attention_heads", 8), 1)
                    if head_dim > 0:
                        out["num_key_value_heads"] = max(1, int(vproj.shape[0]) // head_dim)
                        out["num_kv_heads_total_dim"] = int(vproj.shape[0])

                # Odd layers are SlidingWindowAttention in the hybrid stack.
                # If layer 1 exists, its q/k projection shapes are a direct
                # source of head counts and should override weaker defaults.
                sw_q = sd.get("model.layers.1.self_attn.q_proj.weight")
                sw_k = sd.get("model.layers.1.self_attn.k_proj.weight")
                if sw_q is not None and sw_k is not None and out.get("hidden_size"):
                    out["num_attention_heads_total_dim"] = int(sw_q.shape[0])
                    out["num_kv_heads_total_dim"] = int(sw_k.shape[0])
                return out
            except Exception:
                return {}


        def generate_unified(cfg, selected_backend, selected_mode, model_path_val,
                             prompt_val, max_tok, temp, raw_prompt_val,
                             miss_policy_val, ram_load_ratio_val, sweep_ratios_val,
                             speculative_val=False, draft_depth_val=1, gamma_val=4,
                             residency_planner_val=True, paging_val="auto",
                             residency_decay_val=None, decode_wait_val=None, verify_wait_val=None):
            raw = generate_api(
                cfg,
                selected_backend=selected_backend,
                selected_mode=selected_mode,
                model_path_val=model_path_val,
                prompt_val=prompt_val,
                max_tok=max_tok,
                temp=temp,
                raw_prompt_val=raw_prompt_val,
                ram_load_ratio=ram_load_ratio_val,
                sweep_ram_load_ratios=sweep_ratios_val,
                miss_policy=miss_policy_val,
                speculative=speculative_val,
                draft_depth=draft_depth_val,
                gamma=gamma_val,
                residency_planner=residency_planner_val,
                paging=paging_val,
                residency_decay=residency_decay_val,
                decode_wait_s=decode_wait_val,
                verify_wait_s=verify_wait_val,
            )
            if raw.get("error"):
                text = f"Error: {raw.get('error')}"
                if raw.get("traceback"):
                    text = f"{text}\n{raw['traceback']}"
                return (
                    gr.update(value=text, visible=True),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    f"_error: {raw.get('error')}_",
                    _empty_inference_df(),
                    gr.update(value=raw, visible=True),
                )

            rows = raw.get("rows", [])
            diff = raw.get("output_diff")
            mode = _normalize_inference_mode(raw.get("mode"))
            outputs = raw.get("outputs", {})
            table_md = _format_inference_notices(raw) + _format_inference_stats(rows, diff)
            chart_df = _rows_to_chart_df(rows)
            if mode == "compare":
                return (
                    gr.update(value="", visible=False),
                    gr.update(value=outputs.get("lazy", ""), visible=True),
                    gr.update(value=outputs.get("full_dram", ""), visible=True),
                    table_md,
                    chart_df,
                    gr.update(value=raw, visible=True),
                )
            return (
                gr.update(value=outputs.get("single", ""), visible=True),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                table_md,
                chart_df,
                gr.update(value=raw, visible=True),
            )

        gen_btn.click(
            fn=generate_unified,
            inputs=[
                config_state, inference_backend, inference_mode, model_path, prompt,
                max_tokens, temperature, raw_prompt, miss_policy, ram_load_ratio, sweep_ratios,
                speculative, draft_depth, spec_gamma, residency_planner,
                paging_mode, residency_decay, decode_wait, verify_wait,
            ],
            outputs=[output_box, lazy_output_box, full_output_box, stats_md, stats_chart, raw_stats],
            api_name=None,
        )

        api_btn = gr.Button("api_generate", visible=False)
        api_out = gr.JSON(value={}, visible=False)
        api_btn.click(
            fn=generate_api,
            inputs=[
                config_state, inference_backend, inference_mode, model_path, prompt,
                max_tokens, temperature, raw_prompt, ram_load_ratio, sweep_ratios, miss_policy,
                speculative, draft_depth, spec_gamma, residency_planner,
                paging_mode, residency_decay, decode_wait, verify_wait,
            ],
            outputs=[api_out],
            api_name="generate",
        )

    return tab
