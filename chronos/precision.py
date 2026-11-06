"""One precision policy for every Chronos training and inference chain.

A run picks a single *compute dtype* when it is built.  Weights, activations,
KV cache and SSD/RAM expert pages all live in that dtype, so the hot path
never converts tensors.  Precision is protected only where rounding would
cost accuracy, and each of those places is either a fused kernel that
accumulates in FP32 internally (no tensor cast) or a deliberately small cast:

  site                         how FP32 is obtained               cast size
  ---------------------------  ---------------------------------  ---------------
  attention softmax            fused SDPA (MLX / torch)           none
  RMSNorm statistics           mx.fast.rms_norm / fused torch     none
  RoPE                         fused rope / tables in compute dt  none
  router softmax               softmax(precise=True) (MLX)        none
  LM head + cross-entropy      upcast hidden [N,H] and W [V,H]    H/V of logits
  ACT halting / aux losses     [B,S,G,D] scalars                  tiny
  optimizer                    FP32 masters + moments             one per update
  gradient accumulation        FP32 running sum                   one per micro

Everything else stays in the compute dtype end to end.  The policy is also
the single place that knows device limits: MLX's CPU backend implements
GatherMM for float32 only (mlx 0.32.4), so CPU MLX runs compute in float32
for the whole model instead of upcasting every expert matmul.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

COMPUTE_DTYPES = ("float32", "bfloat16", "float16")
_ALIASES = {"fp32": "float32", "f32": "float32", "float": "float32",
            "bf16": "bfloat16", "fp16": "float16", "half": "float16", "f16": "float16"}

# Fused kernels (attention softmax, RMSNorm, RoPE, precise router softmax)
# that keep FP32 internally without materializing FP32 tensors.
FUSED_FP32_SITES = ("attention_softmax", "rms_norm", "rope", "router_softmax")
# The only explicit casts left in the chains, with their size bound.
BOUNDARY_CASTS = {
    "lm_head_loss": "hidden [N,H] + head [V,H] -> fp32; logits are born fp32",
    "halting_and_aux_losses": "[B,S,G,D] / [N,E] statistics -> fp32",
    "optimizer_update": "grad -> fp32 master once per update",
    "grad_accumulation": "fp32 running sum across micro-batches",
}


def normalize_dtype_name(name, default: str = "auto") -> str:
    """'bf16' / 'torch.bfloat16' / mx.bfloat16 -> 'bfloat16'; '' -> default."""
    if name is None or name == "":
        return default
    text = str(name).strip().lower().replace("torch.", "").replace("mlx.core.", "")
    text = _ALIASES.get(text, text)
    if text not in COMPUTE_DTYPES + ("auto",):
        raise ValueError(f"unsupported compute dtype {name!r}; expected one of {COMPUTE_DTYPES} or 'auto'")
    return text


@dataclass(frozen=True)
class PrecisionPolicy:
    backend: str                      # "mlx" | "torch"
    device: str                       # "gpu" | "cpu" | "mps" | "cuda"
    compute: str                      # weights, activations, KV cache, expert pages
    master: str = "float32"           # optimizer master weights + moments
    accumulate: str = "float32"       # gradient accumulation
    requested: str = "auto"
    reason: str = ""
    fused_fp32: tuple = field(default=FUSED_FP32_SITES)

    @property
    def kv_cache(self) -> str:
        return self.compute

    @property
    def expert_storage(self) -> str:
        return self.compute

    @property
    def low_precision(self) -> bool:
        return self.compute != "float32"

    def as_dict(self) -> dict:
        out = asdict(self)
        out.update(kv_cache=self.kv_cache, expert_storage=self.expert_storage,
                   boundary_casts=dict(BOUNDARY_CASTS), fused_fp32=list(self.fused_fp32))
        return out

    # ---- backend dtype objects -------------------------------------------
    def mx_dtype(self, which: str = "compute"):
        import mlx.core as mx
        return getattr(mx, getattr(self, which))

    def torch_dtype(self, which: str = "compute"):
        import torch
        return getattr(torch, getattr(self, which))


def resolve_mlx(requested="auto", device: str | None = None) -> PrecisionPolicy:
    """Pick the MLX compute dtype once for a whole run.

    GPU: BF16 by default (FP32 masters keep small updates).  CPU: FP32,
    because MLX CPU GatherMM (sparse expert dispatch) is FP32-only; a
    model-wide FP32 run is exact and avoids per-matmul upcasts.
    """
    import mlx.core as mx
    req = normalize_dtype_name(requested)
    dev = device or ("cpu" if mx.default_device().type == mx.DeviceType.cpu else "gpu")
    if dev == "cpu":
        reason = ("MLX CPU GatherMM supports float32 only; whole model runs in float32"
                  if req not in ("auto", "float32") else "MLX CPU backend has no fast low-precision GEMM")
        return PrecisionPolicy("mlx", "cpu", "float32", requested=req, reason=reason)
    if req == "auto":
        return PrecisionPolicy("mlx", dev, "bfloat16", requested=req, reason="GPU default: BF16 compute, FP32 masters")
    return PrecisionPolicy("mlx", dev, req, requested=req, reason="requested")


def resolve_torch(requested="auto", device: str = "cpu") -> PrecisionPolicy:
    """Pick the torch compute dtype once for a whole run or served model."""
    import torch
    req = normalize_dtype_name(requested)
    dev = str(device).split(":")[0]
    if req != "auto":
        if dev == "cuda" and req == "bfloat16" and not torch.cuda.is_bf16_supported():
            return PrecisionPolicy("torch", dev, "float16", requested=req, reason="GPU lacks BF16; using FP16")
        return PrecisionPolicy("torch", dev, req, requested=req, reason="requested")
    if dev == "cuda":
        dt = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return PrecisionPolicy("torch", dev, dt, requested=req, reason="CUDA default")
    if dev == "mps":
        # Measured on M5 Max / torch 2.10 (Chronos 50M, native dtype, no
        # autocast): BF16 is slower than FP32 for training fwd+bwd (0.749 vs
        # 0.619 s/step) and cached decode (49.1 vs 37.3 ms/token).  BF16
        # stays available on request (halves weight/KV memory).
        return PrecisionPolicy("torch", dev, "float32", requested=req,
                               reason="MPS default: FP32 measured faster than BF16")
    return PrecisionPolicy("torch", dev, "float32", requested=req, reason="CPU default: float32 is the fast CPU GEMM")


def place_torch_model(model, device: str, requested="auto") -> tuple[object, PrecisionPolicy]:
    """Move a torch Chronos model to ``device`` in the policy compute dtype.

    One conversion at load time (weights + any placeholders); afterwards the
    forward pass runs in that dtype with no per-op casts.  Call this before
    an ``ExpertStore`` is built so SSD/RAM expert pages adopt the same dtype.
    """
    from chronos.model.moe_chronos import LazyExpertPlaceholder
    policy = resolve_torch(requested, device)
    dtype = policy.torch_dtype()
    model = model.to(device=device, dtype=dtype)
    for module in model.modules():
        if isinstance(module, LazyExpertPlaceholder):
            module.dtype = dtype
    model._chronos_precision = policy
    return model, policy


def torch_model_dtype(model):
    """Compute dtype of a torch Chronos model (its embedding weight)."""
    inner = getattr(model, "model", model)
    emb = getattr(inner, "embed_tokens", None)
    if emb is not None:
        return emb.weight.dtype
    return next(model.parameters()).dtype


# ---- exact MLX <-> torch tensor bridges ------------------------------------
# numpy has no bfloat16, so a BF16 tensor crosses as its uint16 bit pattern
# (exact, no FP32 detour and no second rounding).  FP16/FP32 cross natively.

def mx_to_torch(arr, dtype: str | None = None):
    """mx.array -> CPU torch tensor, keeping (or converting once to) ``dtype``."""
    import mlx.core as mx
    import numpy as np
    import torch
    if dtype is not None:
        target = getattr(mx, normalize_dtype_name(dtype, "float32"))
        if arr.dtype != target:
            arr = arr.astype(target)
    if arr.dtype == mx.bfloat16:
        bits = arr.view(mx.uint16)
        mx.eval(bits)
        return torch.from_numpy(np.array(bits, copy=True)).view(torch.bfloat16)
    mx.eval(arr)
    return torch.from_numpy(np.array(arr, copy=True))


def torch_to_mx(tensor, dtype: str | None = None):
    """torch tensor -> mx.array, keeping (or converting once to) ``dtype``."""
    import mlx.core as mx
    import torch
    t = tensor.detach().cpu()
    if dtype is not None:
        target = getattr(torch, normalize_dtype_name(dtype, "float32"))
        if t.dtype != target:
            t = t.to(target)
    if t.dtype == torch.bfloat16:
        return mx.array(t.contiguous().view(torch.uint16).numpy()).view(mx.bfloat16)
    return mx.array(t.contiguous().numpy())
