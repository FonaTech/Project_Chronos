"""
chronos/tuning/chronos_auto_tuner.py — ChronosAutoTuner: λ1/λ2/lookahead Optuna search.

Extends the vendored AutoTuner base with Chronos-specific hyperparameters.
Uses a lightweight training loop (no external unsloth/trl dependency).
"""
import os
import time
import dataclasses
from dataclasses import dataclass, field
from typing import List


# Knobs that only change a looped model: searching any of them makes the probe loop.
LOOP_KEYS = ("recurrent_order", "recurrent_dt_param", "recurrent_dt_min", "recurrent_dt_max", "recurrent_dt_init",
    "recurrent_later_topk", "recurrent_later_groups", "lambda_iteration_cluster", "lookahead_teacher_scope")

from chronos.tuning._base_tuner import (
    AutoTuner, SearchSpaceConfig, AutoTuneResult, _gc_collect,
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class ChronosSearchSpaceConfig(SearchSpaceConfig):
    tune_lambda_balance: bool = True
    lambda_balance_min: float = 1e-4
    lambda_balance_max: float = 1e-2

    tune_lambda_temporal: bool = True
    lambda_temporal_min: float = 1e-4
    lambda_temporal_max: float = 1e-2

    tune_lookahead_steps: bool = False
    lookahead_steps_choices: List[int] = field(default_factory=lambda: [1, 2])

    tune_lr: bool = True

    # Extended (UI fix 6): structural knobs that can also be searched.
    tune_hidden_size: bool = False
    hidden_size_choices: List[int] = field(default_factory=lambda: [128, 256, 384, 512])

    tune_num_experts: bool = False
    num_experts_choices: List[int] = field(default_factory=lambda: [2, 4, 6, 8])

    tune_num_shared_experts: bool = False
    num_shared_experts_choices: List[int] = field(default_factory=lambda: [0, 1, 2])

    tune_kv_latent_dim: bool = False
    kv_latent_dim_choices: List[int] = field(default_factory=lambda: [32, 64, 96, 128])

    tune_lambda_lookahead: bool = False
    lambda_lookahead_min: float = 1e-3
    lambda_lookahead_max: float = 1.0

    # Loop knobs. They only matter for a looped model, so searching any of them makes the probe model run
    # `probe_recurrent_depth` iterations over `probe_recurrent_layers` block layers (other probes stay at depth 1).
    # The probe score comes from a few steps and cannot resolve differences as small as dt 0.2-0.5 at depth 4:
    # treat the loop knobs as a coarse screen.
    tune_recurrent_order: bool = False
    recurrent_order_choices: List[str] = field(default_factory=lambda: ["layer", "block"])

    tune_recurrent_dt_init: bool = False
    recurrent_dt_init_min: float = 0.1
    recurrent_dt_init_max: float = 1.0

    tune_recurrent_dt_param: bool = False
    recurrent_dt_param_choices: List[str] = field(default_factory=lambda: ["clamp", "sigmoid"])

    tune_recurrent_dt_min: bool = False
    recurrent_dt_min_min: float = 0.01
    recurrent_dt_min_max: float = 0.2

    tune_recurrent_dt_max: bool = False
    recurrent_dt_max_min: float = 0.5
    recurrent_dt_max_max: float = 1.0

    # Cost knobs: they remove expert routes or layer calls from later iterations, so the cross-entropy score
    # alone favours "off"; each trial also logs its median step time. later_groups never offers 1: a single
    # layer looping alone is no longer a block loop.
    tune_recurrent_later_topk: bool = False
    recurrent_later_topk_choices: List[int] = field(default_factory=lambda: [0, 1])

    tune_recurrent_later_groups: bool = False
    recurrent_later_groups_choices: List[int] = field(default_factory=lambda: [0, 2])

    tune_lambda_iteration_cluster: bool = False
    lambda_iteration_cluster_choices: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 1.0])

    tune_lookahead_teacher_scope: bool = False
    lookahead_teacher_scope_choices: List[str] = field(default_factory=lambda: ["first", "iterations"])

    probe_recurrent_depth: int = 4
    probe_recurrent_layers: int = 4


class ChronosAutoTuner(AutoTuner):
    """
    Extends AutoTuner to search over Chronos-specific hyperparameters
    (λ1, λ2, lookahead_steps) using a lightweight native training loop
    (no trl/unsloth dependency required).
    """

    def _sample_params(self, trial, ss) -> dict:
        params = {}

        if ss.tune_lr:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", ss.lr_min, ss.lr_max, log=True
            )
        else:
            params["learning_rate"] = 2e-4

        if isinstance(ss, ChronosSearchSpaceConfig):
            if ss.tune_lambda_balance:
                params["lambda_balance"] = trial.suggest_float(
                    "lambda_balance", ss.lambda_balance_min, ss.lambda_balance_max, log=True
                )
            else:
                params["lambda_balance"] = 5e-4

            if ss.tune_lambda_temporal:
                params["lambda_temporal"] = trial.suggest_float(
                    "lambda_temporal", ss.lambda_temporal_min, ss.lambda_temporal_max, log=True
                )
            else:
                params["lambda_temporal"] = 1e-3

            if ss.tune_lookahead_steps:
                params["lookahead_steps"] = trial.suggest_categorical(
                    "lookahead_steps", ss.lookahead_steps_choices
                )
            else:
                params["lookahead_steps"] = 2

            # Extended (UI fix 6) — categorical structural knobs
            if ss.tune_hidden_size:
                params["hidden_size"] = trial.suggest_categorical(
                    "hidden_size", ss.hidden_size_choices)
            if ss.tune_num_experts:
                params["num_experts"] = trial.suggest_categorical(
                    "num_experts", ss.num_experts_choices)
            if ss.tune_num_shared_experts:
                params["num_shared_experts"] = trial.suggest_categorical(
                    "num_shared_experts", ss.num_shared_experts_choices)
            if ss.tune_kv_latent_dim:
                params["kv_latent_dim"] = trial.suggest_categorical(
                    "kv_latent_dim", ss.kv_latent_dim_choices)
            if ss.tune_lambda_lookahead:
                params["lambda_lookahead"] = trial.suggest_float(
                    "lambda_lookahead", ss.lambda_lookahead_min, ss.lambda_lookahead_max, log=True)

            # Loop knobs: the probe model needs more than one iteration for them to change anything.
            if ss.tune_recurrent_order:
                params["recurrent_order"] = trial.suggest_categorical(
                    "recurrent_order", ss.recurrent_order_choices)
            if ss.tune_recurrent_dt_param:
                params["recurrent_dt_param"] = trial.suggest_categorical(
                    "recurrent_dt_param", ss.recurrent_dt_param_choices)
            if ss.tune_recurrent_dt_min:
                params["recurrent_dt_min"] = trial.suggest_float(
                    "recurrent_dt_min", ss.recurrent_dt_min_min, ss.recurrent_dt_min_max, log=True)
            if ss.tune_recurrent_dt_max:
                params["recurrent_dt_max"] = trial.suggest_float(
                    "recurrent_dt_max", ss.recurrent_dt_max_min, ss.recurrent_dt_max_max)
            if ss.tune_recurrent_dt_init:
                params["recurrent_dt_init"] = trial.suggest_float(
                    "recurrent_dt_init", ss.recurrent_dt_init_min, ss.recurrent_dt_init_max, log=True)
            if ss.tune_recurrent_later_topk:
                params["recurrent_later_topk"] = trial.suggest_categorical(
                    "recurrent_later_topk", ss.recurrent_later_topk_choices)
            if ss.tune_recurrent_later_groups and params.get("recurrent_order", "block") == "block":
                # Prefix blocks exist only in block order; a trial that drew layer order skips this knob.
                params["recurrent_order"] = "block"
                params["recurrent_later_groups"] = trial.suggest_categorical(
                    "recurrent_later_groups", ss.recurrent_later_groups_choices)
            if ss.tune_lambda_iteration_cluster:
                params["lambda_iteration_cluster"] = trial.suggest_categorical(
                    "lambda_iteration_cluster", ss.lambda_iteration_cluster_choices)
            if ss.tune_lookahead_teacher_scope:
                params["lookahead_teacher_scope"] = trial.suggest_categorical(
                    "lookahead_teacher_scope", ss.lookahead_teacher_scope_choices)
            if any(key in params for key in LOOP_KEYS):
                params["recurrent_depth"] = max(2, int(ss.probe_recurrent_depth))
                params["num_hidden_layers"] = 2 + max(1, int(ss.probe_recurrent_layers))
        else:
            params["lambda_balance"] = 5e-4
            params["lambda_temporal"] = 1e-3
            params["lookahead_steps"] = 2

        return params

    def _probe_tokens(self, dataset_path, tokenizer_dir):
        """Real tokens of the user's dataset for the probe, read once per run; ``None`` means random ids."""
        cache = self.__dict__.setdefault("_probe_cache", {})
        key = (dataset_path, tokenizer_dir)
        if key not in cache:
            from chronos.tuning.probe_data import load_probe_tokens
            cache[key] = load_probe_tokens(dataset_path, tokenizer_dir)
            if cache[key] is None:
                line = "[probe] no text could be read from the dataset; trials use random token ids"
            else:
                line = f"[probe] {cache[key][0].size} real tokens from the dataset (vocab {cache[key][1]})"
            self._put({"type": "log", "line": line})
        return cache[key]

    def _run_probe_trial(
        self, trial, model_id, dataset_path, train_ratio,
        probe_steps, output_dir, seed, params, **kwargs
    ) -> float:
        import torch
        from chronos.backend import resolve_training_device
        from chronos.model.config import ChronosConfig
        from chronos.model.model_chronos import ChronosForCausalLM
        from chronos.model.checkpoint import (
            chronos_config_from_checkpoint,
            load_checkpoint_state_dict,
            load_state_dict_controlled,
        )
        from chronos.trainer.loss_mixin import chronos_loss_term

        _, device = resolve_training_device("auto")
        if device == "mlx":
            # The probe trains a torch model and "mlx" is not a torch device: use the best torch device instead.
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        lr = params.get("learning_rate", 2e-4)
        lb = params.get("lambda_balance", 5e-4)
        lt = params.get("lambda_temporal", 1e-3)
        la = params.get("lookahead_steps", 2)
        hs = int(params.get("hidden_size", 256))
        ne = int(params.get("num_experts", 4))
        ns = int(params.get("num_shared_experts", 1))
        kv = int(params.get("kv_latent_dim", 64))
        real = self._probe_tokens(dataset_path, kwargs.get("tokenizer_dir"))
        vocab_kw = {"vocab_size": real[1]} if real else {}
        layers = int(params.get("num_hidden_layers", 4))
        # The searched objective weights reach the model config, so the probe trains on the objective they set.
        loop = {key: params[key] for key in ("lambda_lookahead",) if key in params}
        if params.get("recurrent_depth"):
            depth = int(params["recurrent_depth"])
            loop.update(recurrent_enabled=True, recurrent_depth=depth, min_recurrent_depth=max(1, depth // 2),
                    max_inference_depth=depth, prelude_layers=1, coda_layers=1)
            for key in LOOP_KEYS:
                if key in params:
                    loop[key] = params[key]

        if os.path.exists(model_id):
            try:
                cfg, _sources = chronos_config_from_checkpoint(
                    model_id,
                    overrides={
                        "hidden_size": hs,
                        "num_experts": ne,
                        "num_shared_experts": ns,
                        "kv_latent_dim": kv,
                        "lookahead_steps": la,
                        "lambda_balance": lb,
                        "lambda_temporal": lt,
                        **loop,
                    },
                    require_unsniffable=True,
                )
            except Exception:
                cfg = ChronosConfig(
                    hidden_size=hs, num_hidden_layers=layers, num_experts=ne,
                    num_experts_per_tok=2, num_shared_experts=ns,
                    kv_latent_dim=kv, lookahead_steps=la,
                    lambda_balance=lb, lambda_temporal=lt,
                    use_hybrid_attention=True, use_moe=True, **vocab_kw, **loop,
                )
        else:
            cfg = ChronosConfig(
                hidden_size=hs,
                num_hidden_layers=layers,
                num_experts=ne,
                num_experts_per_tok=2,
                num_shared_experts=ns,
                kv_latent_dim=kv,
                lookahead_steps=la,
                lambda_balance=lb,
                lambda_temporal=lt,
                use_hybrid_attention=True,
                use_moe=True,
                **vocab_kw,
                **loop,
            )
        if real and real[1] > int(cfg.vocab_size):
            # A checkpoint's smaller vocabulary cannot embed this tokenizer's ids.
            self._put({"type": "log", "line": f"[probe] model vocab {cfg.vocab_size} < tokenizer vocab {real[1]}; trials use random token ids"})
            real = None
        if any(key in loop for key in ("recurrent_dt_init", "recurrent_dt_min", "recurrent_dt_max")):
            # Independently drawn bounds can exclude the drawn init; start inside the range instead.
            cfg.recurrent_dt_init = min(max(float(cfg.recurrent_dt_init), float(cfg.recurrent_dt_min)), float(cfg.recurrent_dt_max))
        model = ChronosForCausalLM(cfg).to(device)
        if os.path.exists(model_id):
            try:
                w = load_checkpoint_state_dict(model_id, map_location="cpu")
                load_state_dict_controlled(model, w)
            except Exception:
                pass

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        vocab_size = cfg.vocab_size
        seq_len = 64
        batch = 2
        model.train()
        losses, step_times = [], []
        for step in range(probe_steps):
            if real:
                from chronos.tuning.probe_data import probe_batch
                ids = torch.from_numpy(probe_batch(real[0], step, batch, seq_len)).to(device)
            else:
                ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
            t0 = time.perf_counter()
            out, lookahead_probs = model(ids, labels=ids, legacy_output=True)
            # Train on the same objective the trainers use (every weighted term of the config), but score the
            # trial by cross-entropy: a score that includes the weighted terms rewards small weights for their own sake.
            loss = chronos_loss_term(model, out.loss, lookahead_probs, cfg, aux_loss=out.aux_loss,
                                input_ids=ids, labels=ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ce = float(out.loss.item())
            step_times.append(time.perf_counter() - t0)
            losses.append(ce)
            trial.report(ce, step=step)
            if trial.should_prune():
                del model
                _gc_collect()
                raise optuna.exceptions.TrialPruned()

        final_loss = sum(losses[-max(1, probe_steps // 5):]) / max(1, probe_steps // 5)
        step_s = sorted(step_times)[len(step_times) // 2]
        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("step_time_s", step_s)
        self._put({"type": "log", "line": f"  probe cross-entropy {final_loss:.4f}, median step {step_s * 1000:.0f} ms"})
        del model
        _gc_collect()
        return final_loss

    def get_best_chronos_config_patch(self) -> dict:
        patch = self.get_best_config_patch()
        if self.result and self.result.best_params:
            p = self.result.best_params
            for key in (
                "lambda_balance", "lambda_temporal", "lambda_lookahead",
                "lookahead_steps", "hidden_size", "num_experts",
                "num_shared_experts", "kv_latent_dim",
                *LOOP_KEYS,
            ):
                if key in p:
                    patch[key] = p[key]
        return patch
