"""
ui/tabs/config_tab.py — ChronosConfig parameter editor (i18n-wired)
"""
import gradio as gr
from chronos.model.config import ChronosConfig
from ui.i18n import t, register_translatable


def build_config_tab():
    """Returns (config_state, list_of_input_components)"""
    config_state = gr.State(ChronosConfig().__dict__.copy())

    with gr.Tab(t("tab.config")) as tab:
        register_translatable(tab, "tab.config")

        arch_md = gr.Markdown(f"### {t('config.arch')}")

        with gr.Row():
            hidden_size     = gr.Slider(128, 2048, value=512,  step=128, label=t("config.hidden_size"))
            num_layers      = gr.Slider(2,   32,   value=8,    step=2,   label=t("config.num_layers"))
            num_experts     = gr.Slider(2,   16,   value=4,    step=2,   label=t("config.num_experts"))
            experts_per_tok = gr.Slider(1,   4,    value=1,    step=1,   label=t("config.experts_per_tok"))
            register_translatable(hidden_size,     "config.hidden_size")
            register_translatable(num_layers,      "config.num_layers")
            register_translatable(num_experts,     "config.num_experts")
            register_translatable(experts_per_tok, "config.experts_per_tok")

        with gr.Row():
            num_shared     = gr.Slider(0, 4,    value=1,    step=1,   label=t("config.shared_experts"))
            lookahead      = gr.Slider(1, 4,    value=2,    step=1,   label=t("config.lookahead"))
            kv_latent_dim  = gr.Slider(16, 256, value=64,   step=16,  label=t("config.kv_latent"))
            sliding_window = gr.Slider(128, 8192, value=2048, step=128, label=t("config.sliding_window"))
            register_translatable(num_shared,     "config.shared_experts")
            register_translatable(lookahead,      "config.lookahead")
            register_translatable(kv_latent_dim,  "config.kv_latent")
            register_translatable(sliding_window, "config.sliding_window")

        loss_md = gr.Markdown(f"### {t('config.loss')}")

        with gr.Row():
            lambda_balance  = gr.Number(value=5e-4, label=t("config.lambda_balance"),  precision=6)
            lambda_temporal = gr.Number(value=1e-3, label=t("config.lambda_temporal"), precision=6)
            register_translatable(lambda_balance,  "config.lambda_balance")
            register_translatable(lambda_temporal, "config.lambda_temporal")

        hw_md = gr.Markdown(f"### {t('config.hw')}")

        with gr.Row():
            vram_budget     = gr.Slider(1.0, 48.0, value=4.0,  step=0.5,  label=t("config.vram_budget"))
            pinned_frac     = gr.Slider(0.05, 0.5, value=0.25, step=0.05, label=t("config.pinned_frac"))
            use_hybrid_attn = gr.Checkbox(value=True, label=t("config.hybrid_attn"))
            register_translatable(vram_budget,     "config.vram_budget")
            register_translatable(pinned_frac,     "config.pinned_frac")
            register_translatable(use_hybrid_attn, "config.hybrid_attn")

        train_md = gr.Markdown(f"### {t('config.train')}")

        with gr.Row():
            learning_rate = gr.Number(value=5e-4, label=t("config.lr"), precision=6)
            batch_size    = gr.Slider(1, 64,   value=16,  step=1,   label=t("config.batch_size"))
            accum_steps   = gr.Slider(1, 32,   value=8,   step=1,   label=t("config.accum"))
            max_seq_len   = gr.Slider(64, 4096, value=512, step=64,  label=t("config.max_seq_len"))
            register_translatable(learning_rate, "config.lr")
            register_translatable(batch_size,    "config.batch_size")
            register_translatable(accum_steps,   "config.accum")
            register_translatable(max_seq_len,   "config.max_seq_len")

        with gr.Row():
            epochs        = gr.Slider(1, 20,    value=2,    step=1,   label=t("config.epochs"))
            save_interval = gr.Slider(100, 5000, value=1000, step=100, label=t("config.save_interval"))
            data_path     = gr.Textbox(label=t("config.data_path"), placeholder="./dataset/pretrain.jsonl")
            save_dir      = gr.Textbox(label=t("config.save_dir"),  value="./out")
            register_translatable(epochs,        "config.epochs")
            register_translatable(save_interval, "config.save_interval")
            register_translatable(data_path,     "config.data_path")
            register_translatable(save_dir,      "config.save_dir")

        config_display = gr.JSON(label="Current Config", value={})

        all_inputs = [
            hidden_size, num_layers, num_experts, experts_per_tok,
            num_shared, lookahead, kv_latent_dim, sliding_window,
            lambda_balance, lambda_temporal,
            vram_budget, pinned_frac, use_hybrid_attn,
            learning_rate, batch_size, accum_steps, max_seq_len,
            epochs, save_interval, data_path, save_dir,
        ]

        def update_config(*vals):
            (hs, nl, ne, ept, ns, la, kv, sw,
             lb, lt, vb, pf, ha,
             lr, bs, ac, msl, ep, si, dp, sd) = vals
            cfg = {
                "hidden_size": int(hs), "num_hidden_layers": int(nl),
                "num_experts": int(ne), "num_experts_per_tok": int(ept),
                "num_shared_experts": int(ns), "lookahead_steps": int(la),
                "kv_latent_dim": int(kv), "sliding_window_size": int(sw),
                "lambda_balance": float(lb), "lambda_temporal": float(lt),
                "vram_budget_gb": float(vb), "pinned_memory_max_fraction": float(pf),
                "use_hybrid_attention": bool(ha),
                "learning_rate": float(lr), "batch_size": int(bs),
                "accumulation_steps": int(ac), "max_seq_len": int(msl),
                "epochs": int(ep), "save_interval": int(si),
                "data_path": dp, "save_dir": sd,
            }
            return cfg, cfg

        for inp in all_inputs:
            inp.change(fn=update_config, inputs=all_inputs, outputs=[config_state, config_display])

    return config_state, all_inputs, data_path, save_dir
