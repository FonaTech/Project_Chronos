import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from chronos.model.kernel import Logger, is_main_process, save_resume_state, setup_seed
from chronos.trainer.optim_utils import get_lr  # warmup→cosine override
from chronos.data.flexible_dataset import (
    FlexibleDataset as PretrainDataset,
    StreamingSFTDataset as SFTDataset,
)

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.temporal_loss import lookahead_supervision_loss
from chronos.trainer.loss_mixin import (
    chronos_loss_term, collect_router_probs, first_iteration_router_probs, lookahead_teacher,
)
from chronos.model.checkpoint import save_state_dict_with_config
from chronos.trainer.device_utils import (
    autocast_context,
    configure_cpu_threads,
    grad_scaler,
    optimizer_step_with_scaler,
    runtime_summary,
)
from chronos.trainer.paged_store import PagedTrainingController


class ChronosTrainer:
    """
    Training orchestrator for Project Chronos.

    Pretraining loop (native kernel utilities) that injects:
    - temporal_locality_loss via λ2 * ||E_t - E_{t-1}||^2
    - load_balance_loss via λ1 (already computed inside ChronosMOEFeedForward)
    - LookaheadRouter accuracy logging
    """

    def __init__(self, config: ChronosConfig, args):
        self.config = config
        self.args = args
        self.device = args.device
        seed = getattr(args, "seed", None)
        if seed is not None:
            setup_seed(int(seed))
        configure_cpu_threads(
            getattr(args, "cpu_threads", None),
            budget_percent=getattr(args, "cpu_budget_percent", 100),
        )
        self.autocast_ctx = autocast_context(self.device, getattr(args, "dtype", "float32"))
        self.scaler = grad_scaler(self.device, getattr(args, "dtype", "float32"))
        summary = runtime_summary(self.device, getattr(args, "dtype", "float32"))
        Logger(
            "Runtime: "
            f"device={summary.device} type={summary.device_type} dtype={summary.dtype} "
            f"cpu_threads={summary.cpu_threads} autocast={summary.autocast} scaler={summary.scaler}"
        )

        self.model = ChronosForCausalLM(config).to(self.device)
        self.paged_training = PagedTrainingController(self.model, config)
        from chronos.trainer.optim_utils import build_configured_optimizer
        self.optimizer = build_configured_optimizer(
            self.model, config, lr=args.learning_rate,
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
        )
        # One compute dtype end to end; FP32 masters live in the optimizer.
        from chronos.trainer.precision_training import apply_training_precision
        apply_training_precision(self, models=())

    def _collect_router_probs(self) -> torch.Tensor:
        """Gather last_router_probs from all MoE layers → [B, S, L, E]."""
        return collect_router_probs(self.model)

    def train_step(self, input_ids, labels, step, total_steps, *, modality_inputs=None, modality_mask=None):
        if hasattr(self.model.model, "recurrent_controller"):
            self.model.model.recurrent_controller.set_training_progress(step, total_steps)
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if modality_inputs:
            modality_inputs = {k: v.to(self.device) for k, v in modality_inputs.items()}
        if modality_mask is not None:
            modality_mask = modality_mask.to(self.device)

        lr = get_lr(step, total_steps, self.args.learning_rate)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

        with self.autocast_ctx:
            paged_masks = self.paged_training.masks_for_forward(self.device)
            result = self.model(
                input_ids, labels=labels,
                modality_inputs=modality_inputs, modality_mask=modality_mask,
                available_expert_masks=paged_masks,
            )
            outputs, lookahead_probs = result.legacy_tuple()
            ce_loss = outputs.loss
            aux_loss = outputs.aux_loss

            loss = chronos_loss_term(
                self.model, ce_loss, lookahead_probs, self.config,
                aux_loss=aux_loss, input_ids=input_ids, labels=labels,
            )

            # Standalone lookahead loss value for logging
            lookahead_loss_val = 0.0
            router_4d = first_iteration_router_probs(self.model)
            if (
                router_4d is not None
                and router_4d.shape[1] > 1
                and lookahead_probs is not None
                and self.config.lookahead_steps > 0
            ):
                teacher = lookahead_teacher(router_4d, lookahead_probs, self.config)
                la_term = lookahead_supervision_loss(
                    lookahead_probs, teacher, self.config.lookahead_steps,
                )
                lookahead_loss_val = float(la_term.item())

            loss = loss / self.args.accumulation_steps

        self.scaler.scale(loss).backward()

        if step % self.args.accumulation_steps == 0:
            optimizer_step_with_scaler(
                self.scaler,
                self.optimizer,
                self.model.parameters(),
                self.args.grad_clip,
            )
            self.paged_training.observe()

        return loss.item() * self.args.accumulation_steps, ce_loss.item(), aux_loss.item(), lookahead_loss_val

    def train_epoch(self, epoch, loader, iters, start_step=0, wandb=None, max_steps=None):
        self.model.train()
        start_time = time.time()
        cap = int(max_steps) if max_steps is not None else None
        total_steps = cap or (self.args.epochs * iters)

        for local_step, batch in enumerate(loader, start=1):
            if local_step > iters:
                break
            step = int(start_step) + local_step
            if cap is not None and step > cap:
                break
            if isinstance(batch, dict):
                input_ids, labels = batch["input_ids"], batch["labels"]
                media, media_mask = batch.get("modality_inputs"), batch.get("modality_mask")
            else:
                input_ids, labels = batch
                media = media_mask = None
            loss, ce_loss, aux_loss, la_loss = self.train_step(
                input_ids, labels, step, total_steps,
                modality_inputs=media, modality_mask=media_mask,
            )

            if local_step % self.args.log_interval == 0 or local_step == iters:
                elapsed = time.time() - start_time
                eta = elapsed / max(local_step, 1) * (iters - local_step) // 60
                lr = self.optimizer.param_groups[-1]['lr']
                Logger(
                    f'Epoch:[{epoch+1}/{self.args.epochs}]({local_step}/{iters}) '
                    f'loss:{loss:.4f} ce:{ce_loss:.4f} aux:{aux_loss:.4f} '
                    f'la:{la_loss:.4f} lr:{lr:.2e} eta:{eta:.1f}min'
                )
                if wandb:
                    wandb.log({
                        "loss": loss, "ce_loss": ce_loss, "aux_loss": aux_loss,
                        "lookahead_loss": la_loss, "lr": lr,
                    })

            if (step % self.args.save_interval == 0 or local_step == iters) and is_main_process():
                self._save(epoch, step, wandb)

    def _save(self, epoch, step, wandb=None):
        self.model.eval()
        os.makedirs(self.args.save_dir, exist_ok=True)
        ckp = f'{self.args.save_dir}/chronos_{self.config.hidden_size}_moe.pth'
        save_state_dict_with_config(
            self.model,
            ckp,
            self.config,
            stage="chronos",
        )
        # Resume state only; the release checkpoint above keeps its dtype
        # policy (BF16 stays BF16) and is not rewritten here.
        save_resume_state(
            f'{self.args.save_dir}/chronos_{self.config.hidden_size}_moe_resume.pth',
            model=self.model, optimizer=self.optimizer, epoch=epoch, step=step,
            scaler=self.scaler, wandb_id=getattr(wandb, 'id', None) if wandb else None,
        )
        self.model.train()
