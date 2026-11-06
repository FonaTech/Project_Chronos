import sys
import chronos.deps  # ensure minimind on sys.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from model.model_minimind import FeedForward, MOEFeedForward
from .config import ChronosConfig


class LazyExpertPlaceholder(nn.Module):
    """Inference-only placeholder for experts that live on SSD/RAM.

    If this module executes directly, Chronos failed to materialize the
    expert before use or failed to route the token through the shared
    expert fallback. That is a correctness bug, so we raise loudly.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.loaded = False

    def forward(self, _x):
        raise RuntimeError("lazy expert placeholder executed before materialization")


def _run_expert(expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a materialized expert.  Expert pages are stored and promoted in the
    model's compute dtype (``ExpertStore`` follows the trunk), so there is no
    cast here; a mismatch is a loading bug and fails loudly."""
    return expert(x)


class ChronosMOEFeedForward(nn.Module):
    """
    Extends MOEFeedForward with:
    1. shared_experts: always resident in VRAM, used as soft fallback
    2. soft_gate: when target experts are unavailable (cache miss), blend
       their routing weight into shared experts instead of blocking on I/O

    Training (available_expert_mask=None): identical to MOEFeedForward,
    plus stores per-token routing probs for temporal_locality_loss.

    Inference (available_expert_mask provided): routes cache-miss tokens
    to shared experts with zero latency.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_aux_loss_coef = config.router_aux_loss_coef

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.shared_experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_shared_experts)
        ])
        self.aux_loss = torch.zeros(1)
        # raw, *unscaled* load-balance value computed in forward(). The
        # trainer-side λ_balance lives in chronos_loss_term so we don't
        # double-scale (router_aux_loss_coef × λ_balance gave a 2 000×
        # under-weighting and let the gate degenerate). Set to None until
        # the first training forward populates it.
        self.aux_loss_raw: torch.Tensor = None
        # Stores routing probs for downstream loss / metrics.
        # We keep TWO refs:
        #   last_router_probs       — detached, safe for logging / cluster
        #                              layout / inference engine prefetch
        #                              (consumers that must NOT change grad).
        #   last_router_probs_grad  — live tensor with autograd attached,
        #                              consumed by chronos_loss_term so the
        #                              λ_temporal / λ_lookahead penalties
        #                              actually shape the gate.
        self.last_router_probs: torch.Tensor = None
        self.last_router_probs_grad: torch.Tensor = None
        # A recurrent block calls this layer once per loop iteration; every
        # call is a real dispatch decision, so routing statistics and the
        # load-balance term must cover all of them (reset per model forward).
        self.iteration_router_probs: list[torch.Tensor] = []
        self.iteration_router_probs_grad: list[torch.Tensor] = []
        self._iteration_aux_raw: list[torch.Tensor] = []
        # router_scope="shared": iterations after the first reuse the first
        # iteration's expert ids (weights are re-gathered from the current
        # scores, so the gate still trains every iteration).
        self.router_scope = str(getattr(config, "router_scope", "per_iteration") or "per_iteration")
        self._shared_topk_idx: torch.Tensor | None = None
        self.fallback_mask_prob = float(getattr(config, "fallback_mask_prob", 0.0))
        self.runtime_miss_policy = "fallback_diagnostic"
        self.runtime_on_demand_loader = None
        self.runtime_touch_expert = None
        self.last_on_demand_experts: list[int] = []
        self.last_fallback_experts: list[int] = []
        self.last_on_demand_weight_mass = 0.0
        self.last_fallback_weight_mass = 0.0

    def reset_iteration_routing(self) -> None:
        self.iteration_router_probs = []
        self.iteration_router_probs_grad = []
        self._iteration_aux_raw = []
        self._shared_topk_idx = None

    def _record_aux(self, aux_raw: torch.Tensor) -> None:
        self._iteration_aux_raw.append(aux_raw)
        # Mean over loop iterations keeps the scale of a single dispatch.
        self.aux_loss_raw = torch.stack(self._iteration_aux_raw).mean()
        self.aux_loss = self.aux_loss_raw * self.router_aux_loss_coef

    def _shared_expert_output(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Average output of all shared experts."""
        out = sum(e(x_flat) for e in self.shared_experts)
        return out / len(self.shared_experts)

    def forward(
        self,
        x: torch.Tensor,
        available_expert_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, H]
            available_expert_mask: [num_experts] bool or float — which experts
                are in VRAM. None during training (all experts available).

        When available_expert_mask is provided the forward pass is written
        as pure tensor arithmetic (no Python if/else per expert) so that
        torch.compile can trace a single static graph without graph breaks.
        """
        B, S, H = x.shape
        x_flat = x.view(-1, H)                                   # [N, H]
        # Router softmax in FP32: an [N, E] island (E is small), mirrors MLX.
        scores = F.softmax(self.gate(x_flat).float(), dim=-1)    # [N, E] fp32
        scores_bse = scores.view(B, S, self.num_experts)
        self.last_on_demand_experts = []
        self.last_fallback_experts = []
        self.last_on_demand_weight_mass = 0.0
        self.last_fallback_weight_mass = 0.0
        # Live (with-grad) view drives the temporal / lookahead losses.
        self.last_router_probs_grad = scores_bse
        # Detached view for read-only consumers (logging, cluster layout,
        # inference-engine prefetch). Same tensor data, no autograd edge.
        self.last_router_probs = scores_bse.detach()
        self.iteration_router_probs.append(self.last_router_probs)
        self.iteration_router_probs_grad.append(scores_bse)

        shared_idx = self._shared_topk_idx
        if (self.router_scope == "shared" and shared_idx is not None
                and shared_idx.shape[0] == scores.shape[0] and len(self.iteration_router_probs) > 1):
            topk_idx = shared_idx
            topk_weight = scores.gather(-1, topk_idx)
        else:
            topk_weight, topk_idx = torch.topk(
                scores, k=self.num_experts_per_tok, dim=-1, sorted=False
            )
            if self.router_scope == "shared":
                self._shared_topk_idx = topk_idx
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight.to(x_flat.dtype)       # one [N, K] cast; expert sums stay in compute dtype

        y = torch.zeros_like(x_flat)

        training_fallback_mask = None
        if (
            available_expert_mask is None
            and self.training
            and self.fallback_mask_prob > 0.0
            and len(self.shared_experts) > 0
        ):
            # Simulate SSD/RAM cache misses during training so the shared
            # fallback sees the same distribution it must cover at inference.
            keep = torch.rand(self.num_experts, device=x_flat.device) >= self.fallback_mask_prob
            if not torch.any(keep):
                keep[torch.randint(0, self.num_experts, (1,), device=x_flat.device)] = True
            training_fallback_mask = keep

        if available_expert_mask is None and training_fallback_mask is None:
            # --- Standard path: routed experts + always-on shared residual ---
            # Sorted dispatch: one argsort + one host read per layer instead
            # of an ``any()`` sync per expert.  Each token runs only its
            # top-k experts; rows are contiguous per expert.
            dead_param_sum = None
            flat_idx = topk_idx.reshape(-1)
            order = torch.argsort(flat_idx, stable=True)
            counts = torch.bincount(flat_idx, minlength=self.num_experts).tolist()
            rows = order // self.num_experts_per_tok
            row_weight = topk_weight.reshape(-1)[order].unsqueeze(-1)
            offset = 0
            for i, expert in enumerate(self.experts):
                count = counts[i]
                if count:
                    token_idx = rows[offset:offset + count]
                    weight = row_weight[offset:offset + count]
                    y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
                    offset += count
                elif self.training:
                    # Dead-expert handling: keep the autograd edge alive
                    # WITHOUT touching y[0,0] (the previous trick poisoned
                    # the BOS position every step). We accumulate a
                    # 0·sum(params) scalar that gets routed into aux_loss
                    # below, which the trainer always backprops on.
                    contrib = sum(p.sum() for p in expert.parameters())
                    dead_param_sum = (
                        contrib if dead_param_sum is None else dead_param_sum + contrib
                    )

            # Always-on shared expert (mirrors the inference branch). The
            # previous code only ran shared_experts under the inference
            # mask path, so during training they received zero gradient
            # — and at inference, the cache-miss fallback fired untrained
            # weights, which is the dominant root cause of "loss looks
            # fine but generations are gibberish."
            if len(self.shared_experts) > 0:
                y = y + self._shared_expert_output(x_flat)

            if self.training:
                # Raw, un-scaled load balance (model exposes raw; trainer scales).
                load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
                aux_raw = (load * scores.mean(0)).sum() * self.num_experts
                if dead_param_sum is not None:
                    aux_raw = aux_raw + 0.0 * dead_param_sum
                # ``aux_loss`` (pre-scaled) stays for legacy readers;
                # chronos_loss_term prefers the iteration-mean ``aux_loss_raw``.
                self._record_aux(aux_raw)
            else:
                self.aux_loss_raw = scores.new_zeros(())
                self.aux_loss = scores.new_zeros(())

        else:
            # --- Masked/offload path: real lazy experts + shared fallback ---
            # We intentionally branch at the Python level here: when an
            # expert is cold we must not execute it at all, because the
            # live module may have been replaced by a placeholder after
            # expert offload. A multiply-by-zero path would still call the
            # unloaded expert and defeat lazy loading entirely.
            if available_expert_mask is None:
                avail = training_fallback_mask.to(dtype=x_flat.dtype, device=x_flat.device)
            else:
                avail = available_expert_mask.to(dtype=x_flat.dtype, device=x_flat.device)  # [E]
            shared_out_full = (
                self._shared_expert_output(x_flat)
                if len(self.shared_experts) > 0 else None
            )
            dead_param_sum = None
            # Sorted dispatch with ONE host read per call: selection counts,
            # routed mass and availability travel together.  The previous
            # per-expert ``mask.nonzero()`` was a device sync per selected
            # expert (E x virtual layers per forward; dominant on MPS/CUDA).
            # Rows per expert stay in ascending token order (stable sort), so
            # values and the index_add_ order are unchanged.
            flat_idx = topk_idx.reshape(-1)
            order = torch.argsort(flat_idx, stable=True)
            sorted_rows = order // self.num_experts_per_tok
            sorted_weight = topk_weight.reshape(-1)[order].unsqueeze(-1)
            mass = torch.zeros(self.num_experts, dtype=torch.float32, device=x_flat.device).index_add_(
                0, flat_idx, topk_weight.detach().reshape(-1).float())
            bins = torch.bincount(flat_idx, minlength=self.num_experts)
            host = torch.stack([bins.to(torch.float32), mass, (avail > 0).to(torch.float32)]).tolist()
            # FP32 counts are exact below 2**24 assignments; beyond that read
            # them as integers (the queue is already drained: no extra stall).
            counts = [int(c) for c in host[0]] if flat_idx.numel() < 2 ** 24 else bins.tolist()
            mass_host = host[1]
            avail_host = [v > 0 for v in host[2]]
            offset = 0

            for i, expert in enumerate(self.experts):
                if not counts[i]:
                    if self.training:
                        # Paged training may keep a cold expert's parameters
                        # on CPU while this layer executes on CUDA/MPS.  A
                        # cross-device 0*parameter edge is invalid and the
                        # cold page intentionally receives no update anyway.
                        live_params = [p for p in expert.parameters() if p.device == x_flat.device]
                        if live_params:
                            contrib = sum(p.sum() for p in live_params)
                            dead_param_sum = (
                                contrib if dead_param_sum is None else dead_param_sum + contrib
                            )
                    continue
                token_idx = sorted_rows[offset:offset + counts[i]]    # [T] ascending
                weight = sorted_weight[offset:offset + counts[i]]      # [T, 1]
                offset += counts[i]
                selected_mass = float(mass_host[i])
                expert_live = not isinstance(self.experts[i], LazyExpertPlaceholder)
                expert_available = bool(avail_host[i])
                inference_stale_mask = (
                    available_expert_mask is not None
                    and not self.training
                    and expert_live
                )
                if expert_live and (expert_available or inference_stale_mask):
                    touch = getattr(self, "runtime_touch_expert", None)
                    if touch is not None and not self.training:
                        touch(int(i))
                    if not expert_available and available_expert_mask is not None and not self.training:
                        self.last_on_demand_experts.append(int(i))
                        self.last_on_demand_weight_mass += selected_mass
                    blended = _run_expert(self.experts[i], x_flat[token_idx])    # [T, H]
                else:
                    loaded = False
                    loader = getattr(self, "runtime_on_demand_loader", None)
                    if (
                        available_expert_mask is not None
                        and not self.training
                        and getattr(self, "runtime_miss_policy", "fallback_diagnostic") in {"on_demand", "sync_on_demand"}
                        and loader is not None
                    ):
                        result = loader(int(i))
                        if isinstance(result, dict):
                            loaded = bool(result.get("ok", False))
                        else:
                            loaded = bool(result)

                    if loaded and not isinstance(self.experts[i], LazyExpertPlaceholder):
                        self.last_on_demand_experts.append(int(i))
                        self.last_on_demand_weight_mass += selected_mass
                        blended = _run_expert(self.experts[i], x_flat[token_idx])
                    else:
                        if (not self.training and
                                self.runtime_miss_policy == "sync_on_demand"):
                            raise RuntimeError(f"strict offload could not load expert {i}")
                        self.last_fallback_experts.append(int(i))
                        self.last_fallback_weight_mass += selected_mass
                        blended = (
                            shared_out_full[token_idx]
                            if shared_out_full is not None
                            else x_flat.new_zeros((token_idx.numel(), H))
                        )
                y.index_add_(0, token_idx, blended * weight)

            # The shared expert is a residual component, not only the cold
            # fallback. This makes available_mask=all_true equivalent to the
            # normal path and keeps SSD/DRAM inference faithful to training.
            if shared_out_full is not None:
                y = y + shared_out_full

            if self.training:
                load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
                aux_raw = (load * scores.mean(0)).sum() * self.num_experts
                if dead_param_sum is not None:
                    aux_raw = aux_raw + 0.0 * dead_param_sum
                self._record_aux(aux_raw)
            else:
                self.aux_loss_raw = scores.new_zeros(())
                self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(B, S, H)
