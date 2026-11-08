"""MLX execution of the Chronos recurrent controller and ACT mixture."""
import random
import mlx.core as mx
import mlx.nn as nn


class RecurrentControllerMLX(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, depth, groups = config.hidden_size, config.recurrent_depth, config.recurrent_groups
        self.loop_embeddings = mx.zeros((depth, h))
        self.group_embeddings = mx.zeros((groups, h))
        self.step_sizes = mx.full((groups,), config.recurrent_dt_init)
        self.halting_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.halting_head = nn.Linear(h, 1)
        self.halting_head.weight = mx.zeros((1, h))
        self.halting_head.bias = mx.full((1,), -1.0)
        self.training_progress = 0.0

    def set_training_progress(self, step, total):
        self.training_progress = min(1.0, max(0.0, step / max(total, 1)))

    def adaptive_active(self, cfg, training):
        if not cfg.adaptive_halting or cfg.recurrent_strategy == "fixed":
            return False
        if not training:
            return True
        if cfg.depth_schedule in {"fixed", "constant", "off"}:
            return False
        return (cfg.depth_schedule in {"adaptive", "act", "on"}
                or self.training_progress >= cfg.depth_schedule_warmup_fraction)

    def execution_depth(self, cfg, training):
        pinned = getattr(self, "_pinned_depth", None)
        if pinned is not None and training:
            return pinned          # sampled once per step by a compiled training step
        ceiling = cfg.recurrent_depth if training else cfg.max_inference_depth
        if not training or cfg.recurrent_strategy == "fixed" or cfg.recurrent_depth_jitter == 0:
            return ceiling
        warmup = min(.99, max(0.0, cfg.depth_schedule_warmup_fraction))
        progress = max(0.0, (self.training_progress - warmup) / (1-warmup))
        high = cfg.min_recurrent_depth + round((ceiling-cfg.min_recurrent_depth)*progress)
        low = max(cfg.min_recurrent_depth, high-round((high-cfg.min_recurrent_depth)*cfg.recurrent_depth_jitter))
        return random.randint(low, high)

    def condition(self, x, step, group):
        return x + self.loop_embeddings[min(step, self.loop_embeddings.shape[0]-1)][None, None, :] + self.group_embeddings[group][None, None, :]

    def contract(self, before, after, group, cfg):
        dt = mx.clip(self.step_sizes[group], cfg.recurrent_dt_min, cfg.recurrent_dt_max).astype(after.dtype)
        return before + dt*(after-before)

    def halt_probability(self, x):
        return mx.sigmoid(self.halting_head(self.halting_norm(x)).squeeze(-1).astype(mx.float32))


def auxiliary_state(states, probabilities, minimum, threshold):
    # A draft forward may run fewer iterations than ``min_recurrent_depth``;
    # the last executed state must still receive the remaining mass (Torch parity).
    minimum = min(int(minimum), len(states))
    target = min(1.0, max(1e-6, threshold))
    cumulative = mx.zeros_like(probabilities[0])
    halted = mx.zeros(cumulative.shape, dtype=mx.bool_)
    mixed, depth, accumulated = mx.zeros_like(states[0]), mx.zeros_like(cumulative), []
    for idx, (state, prob) in enumerate(zip(states, probabilities)):
        if idx + 1 < minimum:
            weight = mx.zeros_like(prob)
        elif idx == len(states)-1:
            weight = mx.where(halted, 0.0, 1-cumulative)
        else:
            clipped = mx.clip(prob, 0.0, 1.0)
            new_halt = ~halted & (cumulative+clipped >= target)
            weight = mx.where(~halted, mx.where(new_halt, 1-cumulative, clipped), 0.0)
            halted = halted | new_halt
        weight = mx.maximum(weight, 0)
        cumulative = mx.clip(cumulative+weight, 0.0, 1.0)
        mixed = mixed + state*weight[..., None].astype(state.dtype)
        depth = depth + weight*(idx+1)
        accumulated.append(cumulative)
    accumulated[-1] = mx.ones_like(cumulative)
    # Shared recurrent weights require gradients through BOTH endpoints;
    # a detached final state creates an expansion-rewarding semi-gradient.
    final = states[-1].astype(mx.float32)
    zero = mx.zeros(())
    distill = mx.stack([mx.mean((s.astype(mx.float32)-final)**2) for s in states[:-1]]).mean() if len(states)>1 else zero
    deltas = [states[i]-states[i-1] for i in range(1,len(states))]
    stability = mx.stack([mx.mean((deltas[i]-deltas[i-1]).astype(mx.float32)**2) for i in range(1,len(deltas))]).mean() if len(deltas)>1 else zero
    # Relative step norm is a diagnostic; it is not a Lipschitz guarantee.
    relative = mx.stack([mx.sqrt(mx.maximum(mx.mean(d.astype(mx.float32)**2), 1e-12)) / mx.sqrt(mx.maximum(mx.mean(s.astype(mx.float32)**2), 1e-12)) for d,s in zip(deltas, states[:-1])]).mean() if deltas else zero
    return mixed, {"halting_probs": mx.stack(accumulated, axis=-1), "depth_used": depth,
                   "ponder_loss": mx.maximum(depth-1, 0).mean(), "distill_loss": distill,
                   "stability_loss": stability, "contraction": relative}
