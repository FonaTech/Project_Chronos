"""
ui/tabs/inference_tab.py — Real-time generation with tokens/s display
"""
import time

import gradio as gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable


def build_inference_tab(config_state: gr.State):
    with gr.Tab(t("tab.inference")) as tab:
        register_translatable(tab, "tab.inference")

        model_path = gr.Textbox(
            label=t("infer.model_path"), placeholder="./out/chronos_512_moe.pth"
        )
        register_translatable(model_path, "infer.model_path")

        prompt = gr.Textbox(
            label=t("infer.prompt"), lines=4,
            placeholder="Once upon a time..."
        )
        register_translatable(prompt, "infer.prompt")

        with gr.Row():
            max_tokens  = gr.Slider(16, 512, value=128, step=16, label=t("infer.max_tokens"))
            temperature = gr.Slider(0.1, 2.0, value=0.85, step=0.05, label=t("infer.temperature"))
            register_translatable(max_tokens,  "infer.max_tokens")
            register_translatable(temperature, "infer.temperature")

        gen_btn = gr.Button(t("infer.generate"), variant="primary")
        register_translatable(gen_btn, "infer.generate")

        output_box = gr.Textbox(label=t("infer.output"), lines=10, interactive=False)
        tps_box    = gr.Number(label=t("infer.tps"), value=0.0, interactive=False, precision=1)
        register_translatable(output_box, "infer.output")
        register_translatable(tps_box,    "infer.tps")

        def generate(cfg, model_path_val, prompt_val, max_tok, temp):
            from transformers import AutoTokenizer
            from chronos.model.config import ChronosConfig
            from chronos.backend import get_backend

            if not prompt_val.strip():
                return "Please enter a prompt.", 0.0

            try:
                model_cfg = ChronosConfig(
                    hidden_size=cfg.get("hidden_size", 512),
                    num_hidden_layers=cfg.get("num_hidden_layers", 8),
                    num_experts=cfg.get("num_experts", 4),
                    use_moe=True,
                    use_hybrid_attention=cfg.get("use_hybrid_attention", True),
                    kv_latent_dim=cfg.get("kv_latent_dim", 64),
                    sliding_window_size=cfg.get("sliding_window_size", 2048),
                    vram_budget_gb=cfg.get("vram_budget_gb", 4.0),
                )

                backend = get_backend()
                tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
                token_ids = tokenizer.encode(prompt_val)
                generated = []
                t0 = time.monotonic()

                if backend == "mlx":
                    import mlx.core as mx
                    from chronos.mlx.model import ChronosMLXModel
                    from chronos.mlx.inference import ChronosMLXInferenceEngine

                    model = ChronosMLXModel(model_cfg)
                    if model_path_val and __import__("os").path.exists(model_path_val):
                        import numpy as np, torch
                        sd = torch.load(model_path_val, map_location="cpu")
                        # best-effort weight copy via numpy bridge
                        for k, v in sd.items():
                            arr = mx.array(v.float().numpy())
                            # walk model attributes by key path
                            try:
                                parts = k.split(".")
                                obj = model
                                for p in parts[:-1]:
                                    obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
                                setattr(obj, parts[-1], arr)
                            except Exception:
                                pass

                    engine = ChronosMLXInferenceEngine(model, model_cfg)
                    input_arr = mx.array([token_ids])
                    for tok_id in engine.generate(input_arr, max_new_tokens=int(max_tok),
                                                  temperature=float(temp)):
                        generated.append(tok_id)
                        if tok_id == tokenizer.eos_token_id:
                            break
                    engine.stop()

                else:
                    import torch
                    from chronos.model.model_chronos import ChronosForCausalLM

                    model = ChronosForCausalLM(model_cfg)
                    if model_path_val and __import__("os").path.exists(model_path_val):
                        weights = torch.load(model_path_val, map_location="cpu")
                        model.load_state_dict(weights, strict=False)

                    device_map = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}[backend]
                    model = model.to(device_map).eval()
                    input_ids = torch.tensor([token_ids]).to(device_map)

                    with torch.no_grad():
                        out, lp = model(input_ids, use_cache=True)
                        past_kv = out.past_key_values
                        for _ in range(int(max_tok)):
                            logits = out.logits[:, -1, :] / float(temp)
                            next_tok = torch.multinomial(
                                torch.softmax(logits, dim=-1), num_samples=1
                            )
                            generated.append(next_tok.item())
                            if next_tok.item() == tokenizer.eos_token_id:
                                break
                            out2, lp2 = model(next_tok, past_key_values=past_kv, use_cache=True)
                            past_kv = out2.past_key_values
                            out = out2

                elapsed = time.monotonic() - t0
                tps = len(generated) / max(elapsed, 1e-6)
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                return f"[{backend}] {decoded}", round(tps, 1)

            except Exception as e:
                import traceback
                return f"Error: {e}\n{traceback.format_exc()}", 0.0

        gen_btn.click(
            fn=generate,
            inputs=[config_state, model_path, prompt, max_tokens, temperature],
            outputs=[output_box, tps_box],
        )

    return tab
