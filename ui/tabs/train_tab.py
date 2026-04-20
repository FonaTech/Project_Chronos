"""
ui/tabs/train_tab.py — Full training loop: Pretrain / SFT / RL / ORPO
"""
import os
import time
import threading
import queue

import gradio as gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable


# ── Background trainer thread ─────────────────────────────────────

class TrainSession:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._log_q: queue.Queue = queue.Queue(maxsize=5000)
        self.status = "idle"
        self.step = 0
        self.loss = None
        # Separate metric history for charts
        self._metrics: list[dict] = []   # [{step, total, ce, aux, temporal, tps}]
        self._metric_lock = threading.Lock()
        # Progress + ETA — set when training kicks off
        self.total_steps: int = 0
        self.t_start: float = 0.0

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: dict, mode: str):
        if self.is_running():
            return
        self._stop.clear()
        self.status = "running"
        self.step = 0
        self.loss = None
        self._metrics.clear()
        self.total_steps = 0
        self.t_start = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, args=(cfg, mode), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.status = "stopped"

    def _put(self, msg: str):
        try:
            self._log_q.put_nowait(msg)
        except queue.Full:
            pass

    def _put_metric(self, m: dict):
        with self._metric_lock:
            self._metrics.append(m)

    def drain_log(self):
        """Return all pending log lines as a single string (no newline prefix)."""
        lines = []
        while True:
            try:
                lines.append(self._log_q.get_nowait())
            except queue.Empty:
                break
        return "\n".join(lines)

    def get_metrics(self):
        with self._metric_lock:
            return list(self._metrics)

    def _run(self, cfg: dict, mode: str):
        t_start = time.monotonic()
        try:
            import torch
            import torch.optim as optim
            from torch.utils.data import DataLoader

            from chronos.model.config import ChronosConfig
            from chronos.model.model_chronos import ChronosForCausalLM
            from chronos.model.temporal_loss import total_loss
            from chronos.model.moe_chronos import ChronosMOEFeedForward

            self._put(f"[{mode.upper()}] Building model...")
            # Pass the FULL config dict so user-set arch overrides (e.g.
            # moe_intermediate_size, num_attention_heads, vocab_size) are
            # actually honored. Previously this method only forwarded a
            # hardcoded subset, so the trainer silently used minimind's
            # auto-derived intermediate_size while the UI estimator showed
            # the user's slider — causing 10x param-count mismatches.
            model_cfg_kwargs = {
                "hidden_size":               cfg.get("hidden_size", 512),
                "num_hidden_layers":         cfg.get("num_hidden_layers", 8),
                "num_experts":               cfg.get("num_experts", 4),
                "num_experts_per_tok":       cfg.get("num_experts_per_tok", 1),
                "num_shared_experts":        cfg.get("num_shared_experts", 1),
                "lookahead_steps":           cfg.get("lookahead_steps", 2),
                "kv_latent_dim":             cfg.get("kv_latent_dim", 64),
                "sliding_window_size":       cfg.get("sliding_window_size", 2048),
                "lambda_balance":            cfg.get("lambda_balance", 5e-4),
                "lambda_temporal":           cfg.get("lambda_temporal", 1e-3),
                "vram_budget_gb":            cfg.get("vram_budget_gb", 4.0),
                "use_hybrid_attention":      cfg.get("use_hybrid_attention", True),
                "use_moe":                   True,
            }
            # Optional overrides: only forward when the user actually set them.
            # For the "auto" sentinel fields (0 = let MiniMindConfig derive
            # it), we skip forwarding when value is 0 so ceil(H·π/64)·64
            # takes effect instead of Linear(H, 0) crashing MPS init.
            AUTO_SENTINEL_KEYS = {"intermediate_size", "moe_intermediate_size"}
            for opt_key, cfg_key in [
                ("num_attention_heads",   "num_attention_heads"),
                ("num_key_value_heads",   "num_key_value_heads"),
                ("rope_dim",              "rope_dim"),
                ("vocab_size",            "vocab_size"),
                ("max_position_embeddings", "max_seq_len"),
                ("intermediate_size",     "moe_intermediate_size"),
                ("moe_intermediate_size", "moe_intermediate_size"),
                ("lambda_lookahead",      "lambda_lookahead"),
                ("lambda_router_anchor",  "lambda_router_anchor"),
                ("pinned_memory_max_fraction", "pinned_memory_max_fraction"),
                ("storage_format",        "storage_format"),
                ("tie_word_embeddings",   "tie_word_embeddings"),
            ]:
                if cfg_key not in cfg:
                    continue
                val = cfg[cfg_key]
                if val in (None, ""):
                    continue
                if opt_key in AUTO_SENTINEL_KEYS and val == 0:
                    continue
                model_cfg_kwargs[opt_key] = val
            model_cfg = ChronosConfig(**model_cfg_kwargs)
            model = ChronosForCausalLM(model_cfg)
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            self._put(f"Model: {params_m:.1f}M params  "
                      f"(H={model_cfg.hidden_size}, L={model_cfg.num_hidden_layers}, "
                      f"E={model_cfg.num_experts}, ffn={model_cfg.intermediate_size}, "
                      f"vocab={model_cfg.vocab_size})")

            save_dir = cfg.get("save_dir", "./out")
            ckp_path = os.path.join(save_dir, f"chronos_{model_cfg.hidden_size}_moe.pth")
            if os.path.exists(ckp_path):
                weights = torch.load(ckp_path, map_location="cpu")
                model.load_state_dict(weights, strict=False)
                self._put(f"Resumed from {ckp_path}")

            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self._put(f"Device: {device}")
            model = model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 5e-4))

            data_path = cfg.get("data_path", "")
            max_seq_len = cfg.get("max_seq_len", 512)
            batch_size = cfg.get("batch_size", 4)
            accum = cfg.get("accumulation_steps", 8)
            epochs = cfg.get("epochs", 1)
            save_interval = cfg.get("save_interval", 500)
            log_interval = max(1, cfg.get("log_interval", 10))

            if not data_path or not os.path.exists(data_path):
                self._put("No dataset found — running synthetic smoke test (50 steps)")
                loader = self._synthetic_loader(model_cfg.vocab_size, max_seq_len, batch_size, n=50)
            else:
                tokenizer = self._load_tokenizer()
                try:
                    if mode == "pretrain":
                        from dataset.lm_dataset import PretrainDataset
                        ds = PretrainDataset(data_path, tokenizer, max_length=max_seq_len)
                    else:
                        from dataset.lm_dataset import SFTDataset
                        ds = SFTDataset(data_path, tokenizer, max_length=max_seq_len)
                    _ = ds[0]
                except (KeyError, Exception) as e:
                    self._put(f"Dataset load failed ({e}), falling back to FlexibleDataset")
                    from chronos.data.flexible_dataset import FlexibleDataset
                    ds = FlexibleDataset(data_path, tokenizer, max_length=max_seq_len)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                    num_workers=0, pin_memory=False)

            model.train()
            global_step = 0
            step_t = time.monotonic()
            # Steps remaining for ETA / progress bar
            try:
                self.total_steps = max(1, len(loader) * epochs)
            except Exception:
                self.total_steps = 0

            for epoch in range(epochs):
                if self._stop.is_set():
                    break
                for step, (input_ids, labels) in enumerate(loader):
                    if self._stop.is_set():
                        break
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)

                    out, lp = model(input_ids, labels=labels)
                    moe_layers = [l.mlp for l in model.model.layers
                                  if isinstance(l.mlp, ChronosMOEFeedForward)]

                    aux_val = out.aux_loss.item() if hasattr(out, "aux_loss") else 0.0

                    if moe_layers and moe_layers[0].last_router_probs is not None:
                        probs = torch.stack(
                            [l.last_router_probs for l in moe_layers], dim=2
                        ).mean(dim=2)
                        from chronos.model.temporal_loss import temporal_locality_loss
                        temp_val = temporal_locality_loss(probs).item()
                        loss = total_loss(out.loss, out.aux_loss, probs,
                                          model_cfg.lambda_balance, model_cfg.lambda_temporal)
                    else:
                        temp_val = 0.0
                        loss = out.loss + out.aux_loss

                    (loss / accum).backward()
                    global_step += 1
                    self.step = global_step
                    self.loss = loss.item()

                    if global_step % accum == 0:
                        import torch.nn as nn
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    if global_step % log_interval == 0:
                        now = time.monotonic()
                        tps = log_interval / max(now - step_t, 1e-6)
                        step_t = now
                        ce_val = out.loss.item()
                        tot_val = loss.item()

                        self._put(
                            f"Epoch {epoch+1}/{epochs}  Step {global_step}  "
                            f"loss={tot_val:.4f}  ce={ce_val:.4f}  "
                            f"aux={aux_val:.4f}  temporal={temp_val:.4f}  "
                            f"steps/s={tps:.2f}"
                        )
                        self._put_metric({
                            "step": global_step,
                            "total": tot_val,
                            "ce": ce_val,
                            "aux": aux_val,
                            "temporal": temp_val,
                            "tps": tps,
                        })

                    if global_step % save_interval == 0:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(
                            {k: v.half().cpu() for k, v in model.state_dict().items()},
                            ckp_path
                        )
                        self._put(f"Saved checkpoint → {ckp_path}")

            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {k: v.half().cpu() for k, v in model.state_dict().items()},
                ckp_path
            )
            elapsed = time.monotonic() - t_start
            self._put(f"Training complete in {elapsed:.1f}s. Saved → {ckp_path}")
            self.status = "finished"

        except Exception as e:
            import traceback
            self._put(f"[ERROR] {e}\n{traceback.format_exc()}")
            self.status = "error"

    def _synthetic_loader(self, vocab_size, seq_len, batch_size, n=50):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        ids = torch.randint(0, vocab_size, (n * batch_size, seq_len))
        ds = TensorDataset(ids, ids.clone())
        return DataLoader(ds, batch_size=batch_size)

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())


_session = TrainSession()


def _fmt_eta(seconds: float) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def _make_loss_chart(metrics: list[dict], total_steps: int = 0, t_start: float = 0.0):
    """Build matplotlib figure: loss curves + throughput + progress/ETA."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Close any prior figures held by this thread to avoid the
        # "More than 20 figures opened" warning Gradio's Plot component triggers.
        plt.close("all")

        if not metrics:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="#888")
            ax.set_axis_off()
            plt.tight_layout()
            return fig

        steps = [m["step"] for m in metrics]
        cur_step = steps[-1]
        elapsed = max(0.0, time.monotonic() - t_start) if t_start else 0.0
        # ETA from average step rate so far (not last point — smoother).
        rate = cur_step / elapsed if elapsed > 0 else 0.0
        remaining = (total_steps - cur_step) / rate if rate > 0 and total_steps else float("nan")
        progress_pct = min(100.0, 100.0 * cur_step / total_steps) if total_steps else 0.0

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5),
                                 gridspec_kw={"width_ratios": [4, 3, 3]})
        fig.patch.set_facecolor("#1a1a2e")

        # ── Left: loss curves
        ax = axes[0]
        ax.set_facecolor("#16213e")
        ax.plot(steps, [m["total"] for m in metrics],    color="#e94560", lw=1.5, label="total")
        ax.plot(steps, [m["ce"] for m in metrics],       color="#0f3460", lw=1.2, label="ce")
        ax.plot(steps, [m["aux"] for m in metrics],      color="#533483", lw=1.2, label="aux")
        ax.plot(steps, [m["temporal"] for m in metrics], color="#e2b714", lw=1.0, label="temporal", ls="--")
        ax.set_title("Loss Curves", color="white", fontsize=11)
        ax.set_xlabel("Step", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=8, facecolor="#16213e", labelcolor="white")

        # ── Middle: throughput
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        ax2.plot(steps, [m["tps"] for m in metrics], color="#00b4d8", lw=1.5)
        ax2.fill_between(steps, [m["tps"] for m in metrics], alpha=0.2, color="#00b4d8")
        ax2.set_title("Throughput (steps/s)", color="white", fontsize=11)
        ax2.set_xlabel("Step", color="#aaa")
        ax2.tick_params(colors="#aaa")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")

        # ── Right: progress bar + numeric panel
        ax3 = axes[2]
        ax3.set_facecolor("#16213e")
        # Horizontal bar
        bar_y = 0.65
        ax3.barh([bar_y], [100], height=0.18, color="#2a2a44", edgecolor="#444")
        ax3.barh([bar_y], [progress_pct], height=0.18, color="#39d98a")
        ax3.set_xlim(0, 100); ax3.set_ylim(0, 1)
        ax3.set_xticks([0, 25, 50, 75, 100])
        ax3.set_yticks([])
        ax3.tick_params(colors="#aaa")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#333")

        # Big text overlay
        ax3.text(50, 0.92, f"Progress  {progress_pct:5.1f}%",
                 ha="center", va="center", color="white", fontsize=12, weight="bold")
        eta_str = _fmt_eta(remaining)
        elapsed_str = _fmt_eta(elapsed)
        ax3.text(50, 0.42, f"step {cur_step:,} / {total_steps or '?':,}",
                 ha="center", va="center", color="#cfd8e3", fontsize=11)
        ax3.text(50, 0.25, f"elapsed {elapsed_str}    ETA {eta_str}",
                 ha="center", va="center", color="#9aa6b2", fontsize=10)
        ax3.text(50, 0.10, f"~{rate:.2f} steps/s avg",
                 ha="center", va="center", color="#9aa6b2", fontsize=9)
        ax3.set_title("Progress & ETA", color="white", fontsize=11)

        plt.tight_layout(pad=1.5)
        return fig

    except Exception:
        return None


def build_train_tab(config_state: gr.State, cfg_save_dir: gr.Textbox):
    with gr.Tab(t("tab.train")) as tab:
        register_translatable(tab, "tab.train")

        # ── Train owns its dataset choice (moved out of Config tab) ─
        with gr.Row():
            data_path = gr.Textbox(
                label=t("config.data_path"),
                placeholder="./tests/fixtures/tiny_pretrain.jsonl",
                scale=3,
            )
            register_translatable(data_path, "config.data_path")
            dataset_upload = gr.File(
                label=t("train.dataset_upload"),
                file_types=[".jsonl"], type="filepath", scale=2,
            )
            register_translatable(dataset_upload, "train.dataset_upload")
        dataset_upload.change(
            fn=lambda fp: fp or "",
            inputs=dataset_upload,
            outputs=data_path,
        )

        # ── Mode + controls ───────────────────────────────────────
        with gr.Row():
            mode = gr.Radio(
                ["pretrain", "sft", "rl", "orpo"],
                value="pretrain", label=t("train.mode")
            )
            register_translatable(mode, "train.mode")
            status_box = gr.Textbox(
                value="idle", label=t("train.status"),
                interactive=False, scale=1,
            )
            register_translatable(status_box, "train.status")

        with gr.Row():
            start_btn = gr.Button(t("train.start"), variant="primary")
            stop_btn  = gr.Button(t("train.stop"),  variant="stop")
            clear_btn = gr.Button(t("train.clear_log"), variant="secondary")
            register_translatable(start_btn, "train.start")
            register_translatable(stop_btn,  "train.stop")
            register_translatable(clear_btn, "train.clear_log")

        # ── Metrics chart ─────────────────────────────────────────
        chart = gr.Plot(label=t("train.chart"), show_label=True)
        register_translatable(chart, "train.chart")

        # ── Scalar readouts ───────────────────────────────────────
        with gr.Row():
            step_box     = gr.Number(label=t("train.step"),     value=0,   interactive=False, precision=0)
            loss_box     = gr.Number(label=t("train.loss"),     value=0.0, interactive=False, precision=4)
            ce_box       = gr.Number(label=t("train.ce_loss"),  value=0.0, interactive=False, precision=4)
            aux_box      = gr.Number(label=t("train.aux_loss"), value=0.0, interactive=False, precision=4)
            tps_box      = gr.Number(label=t("train.tps"),      value=0.0, interactive=False, precision=2)
            register_translatable(step_box, "train.step")
            register_translatable(loss_box, "train.loss")
            register_translatable(ce_box,   "train.ce_loss")
            register_translatable(aux_box,  "train.aux_loss")
            register_translatable(tps_box,  "train.tps")

        # ── Scrollable log ────────────────────────────────────────
        log_box = gr.Textbox(
            label=t("train.log"), lines=18, max_lines=30,
            interactive=False, autoscroll=True,
        )
        register_translatable(log_box, "train.log")

        # ── Callbacks ─────────────────────────────────────────────

        def start_training(cfg, train_mode, dpath):
            if _session.is_running():
                return "already running", 0, 0.0, 0.0, 0.0, 0.0, "Already running.\n", None
            cfg = dict(cfg) if cfg else {}
            cfg["data_path"] = dpath or cfg.get("data_path", "")
            _session.start(cfg, train_mode)
            return "running", 0, 0.0, 0.0, 0.0, 0.0, "Starting...\n", None

        def stop_training():
            _session.stop()
            return "stopped"

        def clear_log():
            return ""

        def poll(current_log: str):
            new_lines = _session.drain_log()
            # Append new lines to existing log; keep last 500 lines to avoid unbounded growth
            if new_lines:
                combined = (current_log + "\n" + new_lines).lstrip("\n")
            else:
                combined = current_log

            lines = combined.split("\n")
            if len(lines) > 500:
                combined = "\n".join(lines[-500:])

            metrics = _session.get_metrics()
            fig = _make_loss_chart(metrics, _session.total_steps, _session.t_start)

            # Latest scalar values from last metric entry
            if metrics:
                last = metrics[-1]
                ce_v = last.get("ce", 0.0)
                aux_v = last.get("aux", 0.0)
                tps_v = last.get("tps", 0.0)
            else:
                ce_v = aux_v = tps_v = 0.0

            return (
                _session.status,
                _session.step or 0,
                _session.loss or 0.0,
                ce_v,
                aux_v,
                tps_v,
                combined,
                fig,
            )

        start_btn.click(
            fn=start_training,
            inputs=[config_state, mode, data_path],
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box, log_box, chart],
        )
        stop_btn.click(fn=stop_training, outputs=[status_box])
        clear_btn.click(fn=clear_log, outputs=[log_box])

        timer = gr.Timer(value=2.0)
        timer.tick(
            fn=poll,
            inputs=[log_box],   # pass current log so we can append
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box, log_box, chart],
        )

    return tab
