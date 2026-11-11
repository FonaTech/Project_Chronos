"""Recurrent-depth primitives used by the Chronos v2 kernel.

The recurrent controller is deliberately separate from attention/MoE.  It
owns loop conditioning, contractive step sizes, ACT weights, and convergence
statistics while the shared Chronos blocks remain ordinary traceable modules.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from chronos.model.precision_ops import RMSNorm


def recommended_dt_init(depth, horizon=2.0, cap=0.5):
    """Step-size init for a looped block: dt = horizon / depth, at most ``cap``.

    One loop iteration is an explicit Euler step x <- x + dt * f(x) of dx/dtau = f(x), where f is the
    block's residual update, so R iterations integrate over the horizon T = R * dt. Near the fixed
    direction f(x) ~ lambda * x each iteration multiplies the residual by (1 + dt * lambda) and the loop
    by (1 + dt * lambda)^(R-1), about exp(T * lambda): constructive interference sets the total gain through T,
    and depth only refines the integration. Holding T fixed therefore keeps the gain while depth grows.

    Measured at 15M (30 min per arm, block order, one seed per arm, noise floor about 0.005-0.008):
    - depth 8, dt 0.25 (T = 2) beat dt 0.5 (T = 4) by 0.018-0.079 in every block of steps after step 10;
    - depth 8, dt 0.25 beat the clamp default whose dt collapsed to 0.05 (T = 0.4) only by 0.002-0.015;
    - depth 8, sigmoid from dt 0.0625 ended at dt 0.033 (T = 0.26) and was worse than dt 0.25 by 0.085-0.333
      after step 100 (within noise before);
    - depth 4 was flat for dt 0.2-0.5 (T = 0.8-2) and worse at dt 1.0 or 0.05;
    - at equal step depth 8 (dt 0.25) led depth 4 (dt 0.5) by 0.041-0.049 in steps 100-600 and was level
      (+0.002, +0.008) afterwards;
    - depth 8, sigmoid from dt 0.25 (range 0.1-1) moved to dt 0.145 (T = 1.16) and ended 0.020-0.023 below
      the fixed dt 0.25 arm after step 600 (validation CE 4.761 against 4.776), so the optimiser prefers a
      smaller T than 2.
    The block rescales its own output when dt changes: the per-iteration residual gain g stayed near 1.07 for
    dt 0.15-0.5 (effective lambda = (g - 1) / dt fell from 0.48 to 0.13), and the total log-gain
    (R - 1) * ln g was 0.45-0.51 in all three. Only kappa = dt * lambda and that total are physical; dt alone
    is a gauge. So T = 2 is a safe starting point inside a flat region, not a fitted optimum, and ``cap`` is
    the largest dt that was measured; depths 2 and 3 are extrapolation.
    Use it with recurrent_dt_param="sigmoid", so the optimizer can still move dt either way."""
    depth = max(1, int(depth))
    return min(float(cap), float(horizon) / depth)


def _dt_raw_init(param, dt_init, dt_min, dt_max):
    """Raw stored value so step 0 gives dt == dt_init under the chosen parameterisation.
    clamp: store dt_init directly.  sigmoid: store the logit that inverts dt_min + (dt_max-dt_min)*sigmoid(raw)."""
    import math
    if param != "sigmoid":
        return float(dt_init)
    span = max(float(dt_max) - float(dt_min), 1e-9)
    frac = min(max((float(dt_init) - float(dt_min)) / span, 1e-6), 1 - 1e-6)
    return float(math.log(frac / (1.0 - frac)))


@dataclass
class RecurrentDepthState:
    halting_probs: torch.Tensor
    ponder_cost: torch.Tensor
    depth_used: torch.Tensor
    stability_loss: torch.Tensor
    distill_loss: torch.Tensor
    contraction: torch.Tensor = None
    exited_early: bool = False


class RecurrentController(nn.Module):
    """Adaptive-compute controller shared by all recurrent groups."""

    def __init__(self, config):
        super().__init__()
        hidden = int(config.hidden_size)
        depth = int(config.recurrent_depth)
        groups = int(config.recurrent_groups)
        self.depth = depth
        self.groups = groups
        self.min_depth = int(config.min_recurrent_depth)
        self.threshold = float(config.halting_threshold)
        self.epsilon = float(config.halting_epsilon)
        self.dt_min = float(config.recurrent_dt_min)
        self.dt_max = float(config.recurrent_dt_max)
        self.dt_param = str(getattr(config, "recurrent_dt_param", "clamp"))

        self.loop_embeddings = nn.Parameter(torch.zeros(depth, hidden))
        self.group_embeddings = nn.Parameter(torch.zeros(groups, hidden))
        self.halting_norm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.halting_head = nn.Linear(hidden, 1)
        # A depth-one migration must initially behave exactly like v1.
        nn.init.zeros_(self.halting_head.weight)
        nn.init.constant_(self.halting_head.bias, -1.0)
        self.step_sizes = nn.Parameter(
            torch.full((groups,), _dt_raw_init(self.dt_param, config.recurrent_dt_init, self.dt_min, self.dt_max))
        )
        self.training_progress = 0.0

    def set_training_progress(self, step: int, total_steps: int) -> None:
        self.training_progress = min(1.0, max(0.0, float(step) / max(int(total_steps), 1)))

    def adaptive_active(self, config, training: bool) -> bool:
        if not bool(config.adaptive_halting) or config.recurrent_strategy == "fixed":
            return False
        if not training:
            return True
        schedule = str(getattr(config, "depth_schedule", "fixed_then_adaptive")).lower()
        if schedule in {"fixed", "constant", "off"}:
            return False
        if schedule in {"adaptive", "act", "on"}:
            return True
        warmup = float(getattr(config, "depth_schedule_warmup_fraction", 0.25))
        return self.training_progress >= warmup

    def execution_depth(self, config, training: bool) -> int:
        """Choose one depth bucket for the complete batch/sequence.

        Batch-wide buckets keep KV and compiler shapes deterministic.  Jitter
        is deliberately disabled by default and never applies at inference.
        The curriculum increases its ceiling linearly after warmup; the
        default zero jitter retains the legacy maximum-depth training graph.
        """
        ceiling = int(config.recurrent_depth if training else config.max_inference_depth)
        if not training or str(getattr(config, "recurrent_strategy", "fixed")) == "fixed":
            return ceiling
        jitter = min(1.0, max(0.0, float(getattr(config, "recurrent_depth_jitter", 0.0))))
        if jitter == 0.0:
            return ceiling
        warmup = min(0.99, max(0.0, float(config.depth_schedule_warmup_fraction)))
        progress = max(0.0, (self.training_progress - warmup) / (1.0 - warmup))
        curriculum_max = self.min_depth + round((ceiling - self.min_depth) * progress)
        lower = max(self.min_depth, curriculum_max - round((curriculum_max - self.min_depth) * jitter))
        # CPU scalar draw does not synchronize the device and follows torch's
        # seeded RNG, making resumed/seeded runs reproducible.
        return int(torch.randint(lower, curriculum_max + 1, ()).item())

    def condition(self, hidden: torch.Tensor, step: int, group: int) -> torch.Tensor:
        # Reuse the last learned depth embedding for conservative test-time
        # depth extrapolation. The transition itself is weight-tied.
        depth_idx = min(int(step), self.loop_embeddings.shape[0] - 1)
        return hidden + self.loop_embeddings[depth_idx][None, None, :] + self.group_embeddings[group][None, None, :]

    def step_size(self, group: int) -> torch.Tensor:
        if self.dt_param == "sigmoid":
            return self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(self.step_sizes[group])
        return self.step_sizes[group].clamp(self.dt_min, self.dt_max)

    def contract(self, before: torch.Tensor, after: torch.Tensor, group: int) -> torch.Tensor:
        dt = self.step_size(group).to(dtype=after.dtype)
        return before + dt * (after - before)

    def halt_probability(self, hidden: torch.Tensor) -> torch.Tensor:
        # FP32 sigmoid avoids saturation in BF16/FP16 routing paths.
        logits = self.halting_head(self.halting_norm(hidden)).squeeze(-1).float()
        return torch.sigmoid(logits)


def recurrent_auxiliary_state(
    states: list[torch.Tensor],
    halt_probs: list[torch.Tensor],
    *,
    min_depth: int,
    threshold: float,
) -> tuple[torch.Tensor, RecurrentDepthState]:
    """ACT-style state mixture plus differentiable recurrence diagnostics."""
    if not states:
        raise ValueError("recurrent_auxiliary_state requires at least one state")
    if len(states) == 1:
        p = halt_probs[0].new_ones(halt_probs[0].shape)
        zero = states[0].new_zeros(())
        depth = p
        return states[0], RecurrentDepthState(
            halting_probs=p.unsqueeze(-1),
            ponder_cost=zero,
            depth_used=depth,
            stability_loss=zero,
            distill_loss=zero,
            contraction=zero,
        )

    # ACT uses an accumulated halting mass.  ``threshold`` is allowed to be
    # zero/one from a UI or checkpoint, but the numerical target is always
    # kept in [epsilon, 1] so neither setting can produce NaNs or a zero-mass
    # mixture.  The final pass receives the exact remainder and therefore
    # always makes the total mixture weight equal to one.
    epsilon = 1e-6
    target = min(1.0, max(epsilon, float(threshold)))
    # Draft forwards (depth override) may stop before ``min_depth``; the last
    # executed state must still receive the remaining mass, never zero.
    min_depth = min(int(min_depth), len(states))
    cumulative = halt_probs[0].new_zeros(halt_probs[0].shape)
    halted = torch.zeros_like(cumulative, dtype=torch.bool)
    mixed = torch.zeros_like(states[0])
    depth_used = torch.zeros_like(cumulative)
    accumulated = []
    for idx, (state, prob) in enumerate(zip(states, halt_probs)):
        step = idx + 1
        if step < min_depth:
            weight = torch.zeros_like(prob)
        elif idx == len(states) - 1:
            weight = torch.where(halted, torch.zeros_like(prob), 1.0 - cumulative)
        else:
            active = ~halted
            clipped = prob.float().clamp(0.0, 1.0).to(prob.dtype)
            new_halt = active & ((cumulative + clipped) >= target)
            weight = torch.where(new_halt, 1.0 - cumulative, clipped)
            weight = torch.where(active, weight, torch.zeros_like(weight))
            halted = halted | new_halt
        weight = weight.clamp_min(0.0)
        cumulative = (cumulative + weight).clamp(0.0, 1.0)
        mixed = mixed + state * weight.to(state.dtype).unsqueeze(-1)
        depth_used = depth_used + weight * float(step)
        accumulated.append(cumulative)

    # These states share transition weights. Detaching the final state makes
    # it a moving target and can REWARD expansion: for s_i = i*w, the old
    # semi-gradient of mean((s_i - stop_grad(s_D))**2) points toward larger w.
    # Use a symmetric depth-consistency objective, differentiating both ends.
    # The legacy distill_loss/config name is retained for checkpoint compatibility.
    final = states[-1].float()
    distill = torch.stack([
        (s.float() - final).pow(2).mean() for s in states[:-1]
    ]).mean()
    if len(states) >= 3:
        deltas = [states[i] - states[i - 1] for i in range(1, len(states))]
        stability = torch.stack([
            (deltas[i] - deltas[i - 1]).float().pow(2).mean()
            for i in range(1, len(deltas))
        ]).mean()
    else:
        stability = states[0].new_zeros(())
    # ACT remainder already integrates expected depth. Subtract one so a
    # mandatory first pass has zero penalty and depth=1 is v1-equivalent.
    ponder = (depth_used.float() - 1.0).clamp_min(0.0).mean()
    # Public state reports accumulated halt mass rather than raw logits.  This
    # is the quantity used by ACT/recurrent-depth monitors and makes the
    # contract explicit: the last recurrent pass is always fully halted.
    reported_halts = list(accumulated)
    reported_halts[-1] = torch.ones_like(reported_halts[-1])
    # Relative step norm; this does not establish a contractive mapping.
    contraction = torch.stack([
        (states[i] - states[i - 1]).float().pow(2).mean().clamp_min(1e-12).sqrt()
        / states[i - 1].float().pow(2).mean().clamp_min(1e-12).sqrt()
        for i in range(1, len(states))
    ]).mean() if len(states) > 1 else states[0].new_zeros(())
    return mixed, RecurrentDepthState(
        halting_probs=torch.stack(reported_halts, dim=-1),
        ponder_cost=ponder,
        depth_used=depth_used,
        stability_loss=stability,
        distill_loss=distill,
        contraction=contraction,
    )
