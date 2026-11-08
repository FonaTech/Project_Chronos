"""Chronos native kernel primitives (replaces the former minimind dependency).

Before 0.3.2 Chronos imported its config base, RoPE tables, SwiGLU FFN and a
dense GQA attention from a minimind checkout that ``chronos.deps`` cloned
from an unpinned ``master`` at import time and pushed onto ``sys.path``.
This module owns those pieces.  Field names, defaults, parameter names and
RoPE numerics are kept identical, so every existing checkpoint and config
sidecar loads unchanged (``tests/test_native_kernel.py`` checks this).

What changed by design:

* RoPE tables are FP32 and bit-identical to the old ones, but one table per
  ``(dim, base, scaling, device)`` is shared by every layer (``rope_tables``)
  instead of each attention layer holding its own copy.
* ``SwiGLU`` uses ``F.silu`` directly (no ``ACT2FN`` lookup per call) and keeps
  the ``gate_proj/up_proj/down_proj`` names expert pages are stored under.
* Logging is a sink registry (``add_log_sink``), so callers capture trainer
  output without monkey-patching a module attribute.
"""
from __future__ import annotations

import math
import os
import random
import threading
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

# ── Config base ───────────────────────────────────────────────────────────

YARN_DEFAULT = {
    "beta_fast": 32, "beta_slow": 1, "factor": 16,
    "original_max_position_embeddings": 2048, "attention_factor": 1.0, "type": "yarn",
}


class ChronosBaseConfig(PretrainedConfig):
    """Dense-transformer fields every Chronos model shares.

    Names and defaults match the pre-0.3.2 base so ``config.json`` sidecars
    written by any earlier release resolve to the same values.
    """

    model_type = "chronos_base"

    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 8, use_moe: bool = False, **kwargs):
        super().__init__(**kwargs)
        g = kwargs.get
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = g("dropout", 0.0)
        self.vocab_size = g("vocab_size", 6400)
        self.bos_token_id = g("bos_token_id", 1)
        self.eos_token_id = g("eos_token_id", 2)
        self.flash_attn = g("flash_attn", True)
        self.num_attention_heads = g("num_attention_heads", 8)
        self.num_key_value_heads = g("num_key_value_heads", 4)
        self.head_dim = g("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = g("hidden_act", "silu")
        self.intermediate_size = g("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = g("max_position_embeddings", 32768)
        self.rms_norm_eps = g("rms_norm_eps", 1e-6)
        self.rope_theta = g("rope_theta", 1e6)
        self.tie_word_embeddings = g("tie_word_embeddings", True)
        self.inference_rope_scaling = g("inference_rope_scaling", False)
        # YaRN is switched on by ``inference_rope_scaling``; transformers 5
        # validates an explicit ``rope_scaling`` kwarg against its own schema.
        self.rope_scaling = dict(YARN_DEFAULT) if self.inference_rope_scaling else None
        # MoE fields (ignored when use_moe is False).
        self.num_experts = g("num_experts", 4)
        self.num_experts_per_tok = g("num_experts_per_tok", 1)
        self.moe_intermediate_size = g("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = g("norm_topk_prob", True)
        self.router_aux_loss_coef = g("router_aux_loss_coef", 5e-4)


# ── RoPE (FP32, rotate-half layout, optional YaRN) ────────────────────────

def _inv_freq(dim: int, base: float, scaling: dict | None, end: int) -> tuple[torch.Tensor, float]:
    cpu = torch.device("cpu")     # explicit: callers may run under ``with torch.device('meta')``
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=cpu)[: dim // 2].float() / dim))
    attn_factor = 1.0
    if scaling is not None:
        # YaRN: f'(i) = f(i) * ((1 - g) + g / s), g a linear ramp over the band.
        orig_max = scaling.get("original_max_position_embeddings", 2048)
        factor = scaling.get("factor", 16)
        beta_fast, beta_slow = scaling.get("beta_fast", 32.0), scaling.get("beta_slow", 1.0)
        attn_factor = scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=cpu).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    return freqs, attn_factor


def precompute_freqs_cis(dim: int, end: int = 32 * 1024, rope_base: float = 1e6,
                         rope_scaling: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 duplicated cos/sin tables ``[end, dim]`` (rotate-half layout), on CPU.

    Same signature and bit-identical output as the pre-0.3.2 helper, so the
    MLX mirror (``chronos.mlx.attention._rope_freqs``) and every trained
    checkpoint keep their RoPE.
    """
    freqs, attn_factor = _inv_freq(int(dim), float(rope_base), rope_scaling, int(end))
    angles = torch.outer(torch.arange(int(end), device=freqs.device), freqs).float()
    # One thread: torch splits cos/sin across threads above a size threshold,
    # and the vector/scalar split points then differ by one ulp from a
    # shorter table (measured on torch 2.10, 18 threads).  Single-threaded,
    # every row depends on its angle only, so a cached table's prefix equals
    # a freshly computed shorter one bit for bit.  Built once per process.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        c, s = torch.cos(angles), torch.sin(angles)
    finally:
        torch.set_num_threads(threads)
    cos = torch.cat([c, c], dim=-1) * attn_factor
    sin = torch.cat([s, s], dim=-1) * attn_factor
    return cos, sin


_ROPE_LOCK = threading.Lock()
_ROPE_TABLES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _scaling_key(scaling: dict | None):
    return None if not scaling else tuple(sorted((str(k), str(v)) for k, v in scaling.items()))


def rope_tables(dim: int, length: int, base: float, scaling, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared FP32 cos/sin tables with at least ``length`` rows on ``device``.

    One table per ``(dim, base, scaling, device)`` serves every layer and
    grows geometrically.  YaRN rescales the frequencies only when the table
    is longer than ``original_max_position_embeddings``, so scaled tables are
    keyed by that regime as well, and growth never crosses the boundary.  Rows
    are therefore bit-identical to ``precompute_freqs_cis(dim, length)``.
    """
    dim, length = int(dim), int(length)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    cap = None
    regime = None
    if scaling:
        orig = scaling.get("original_max_position_embeddings", 2048)
        regime = length / orig > 1.0
        cap = None if regime else int(orig)        # unscaled regime stops at orig rows
    key = (dim, float(base), _scaling_key(scaling), str(dev), regime)
    with _ROPE_LOCK:
        hit = _ROPE_TABLES.get(key)
        if hit is not None and hit[0].shape[0] >= length:
            return hit
        rows = max(length, 2 * (hit[0].shape[0] if hit is not None else 0))
        if cap is not None:
            rows = max(length, min(rows, cap))
        cos, sin = precompute_freqs_cis(dim, rows, base, scaling)
        out = (cos.to(device=dev, dtype=torch.float32), sin.to(device=dev, dtype=torch.float32))
        _ROPE_TABLES[key] = out
        return out


def clear_rope_tables() -> None:
    with _ROPE_LOCK:
        _ROPE_TABLES.clear()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Rotate q/k with tables broadcast on ``unsqueeze_dim``; one rounding each."""
    c, s = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return (q * c + rotate_half(q) * s).to(q.dtype), (k * c + rotate_half(k) * s).to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, T, nKV, d] -> [B, T, nKV * n_rep, d] (only for kernels without native GQA)."""
    if n_rep == 1:
        return x
    b, t, h, d = x.shape
    return x[:, :, :, None, :].expand(b, t, h, n_rep, d).reshape(b, t, h * n_rep, d)


# ── SwiGLU feed-forward ───────────────────────────────────────────────────

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": F.silu, "swish": F.silu, "gelu": F.gelu, "relu": F.relu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
}


def activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name in _ACTIVATIONS:
        return _ACTIVATIONS[name]
    from transformers.activations import ACT2FN      # rare activations only
    return ACT2FN[name]


class FeedForward(nn.Module):
    """Gated FFN ``down(act(gate(x)) * up(x))``.

    Parameter names are the on-disk contract (dense FFN, routed and shared
    experts, ``.ctsr`` expert pages), so they must not change.
    """

    def __init__(self, config, intermediate_size: int | None = None):
        super().__init__()
        inter = int(intermediate_size or config.intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.act_fn = activation(getattr(config, "hidden_act", "silu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


SwiGLU = FeedForward


# ── Training utilities ────────────────────────────────────────────────────

def is_main_process() -> bool:
    import torch.distributed as dist
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


_LOG_SINKS: list[Callable[[str], None]] = []


def Logger(content) -> None:  # noqa: N802  (historic trainer API name)
    """Print on the main process and forward the line to registered sinks."""
    if not is_main_process():
        return
    print(content)
    for sink in list(_LOG_SINKS):
        try:
            sink(str(content))
        except Exception:
            pass


@contextmanager
def add_log_sink(sink: Callable[[str], None]):
    """Receive every ``Logger`` line while the context is active."""
    _LOG_SINKS.append(sink)
    try:
        yield sink
    finally:
        try:
            _LOG_SINKS.remove(sink)
        except ValueError:
            pass


def setup_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices); deterministic cuDNN."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _unwrap(module):
    module = getattr(module, "module", module)          # DDP
    return getattr(module, "_orig_mod", module)         # torch.compile


def save_resume_state(path: str, *, model, optimizer=None, epoch: int = 0, step: int = 0, **extra) -> str:
    """Atomically write a resume file (weights in their dtype + optimizer + counters).

    The release checkpoint itself is written by
    ``chronos.model.checkpoint.save_state_dict_with_config``; this file only
    carries what is needed to continue training.
    """
    import torch.distributed as dist
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    state = {
        "model": {k: v.detach().cpu() for k, v in _unwrap(model).state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch), "step": int(step),
        "world_size": dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
    }
    for key, value in extra.items():
        if value is not None:
            state[key] = _unwrap(value).state_dict() if hasattr(value, "state_dict") else value
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_resume_state(path: str):
    """Load a resume file; ``step`` is rescaled when the world size changed."""
    import torch.distributed as dist
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    saved = int(data.get("world_size", 1) or 1)
    current = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if saved != current:
        data["step"] = int(data.get("step", 0)) * saved // current
    return data
