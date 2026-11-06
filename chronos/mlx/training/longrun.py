"""Time-budgeted, resumable MLX training runs (pretrain → SFT → evaluation).

Designed for multi-day benchmark runs on Apple silicon:

* The wall-clock budget is split across phases; after a short throughput
  probe every phase's optimizer-step count is derived from the *measured*
  tokens/s, so the LR schedule (warmup + cosine) ends when the budget does.
* Every ``checkpoint_every_s`` the full state (BF16 weights, FP32 masters and
  moments, data-stream RNG, phase/step) is written atomically; ``resume``
  continues bit-for-bit from the last snapshot.
* Metrics go to ``metrics.jsonl`` (one JSON object per log interval) —
  losses by component, recurrent depth/halting, routing balance/entropy,
  lookahead recall, grad norm, LR, tokens/s, memory, plus periodic
  validation CE and BLEU/chrF.  The UI monitor tails this file.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


@dataclass
class PhaseSpec:
    name: str                       # "pretrain" | "sft"
    budget_fraction: float
    peak_lr: float
    tokens_per_step: int            # optimizer-step token budget (micro × accum)
    micro_tokens: int = 8192        # tokens per forward/backward
    seq_len: int = 512              # pretrain window
    warmup_fraction: float = 0.02
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0


@dataclass
class RunSpec:
    run_dir: str
    shards_dir: str
    tokenizer_dir: str
    model_config: dict
    phases: list[PhaseSpec]
    budget_hours: float = 72.0
    eval_reserve_fraction: float = 0.04
    log_every_steps: int = 20
    validate_every_s: float = 3600.0
    checkpoint_every_s: float = 1800.0
    bleu_samples: int = 200          # periodic FLORES devtest subset during SFT
    final_eval_samples: int = 1012   # full FLORES+ devtest at the end (0 = skip)
    seed: int = 1234
    dtype: str = "bfloat16"
    # "time": warmup + cosine over the phase's wall-clock budget, so the LR
    # reaches its floor exactly at the deadline even if throughput varies
    # (shared GPU, thermal throttling).  "step": classic step schedule.
    schedule: str = "time"
    # Telemetry (monitor pages): per-tensor weight/grad/update statistics every
    # ``tensor_stats_every_s`` (0 = off) and per-virtual-layer routing.
    tensor_stats_every_s: float = 1800.0
    # Greedy FLORES+ dev probe at each pretrain validation (``<|en|>…<|zh|>``
    # format, both directions); samples go to ``samples.jsonl``.  0 = off.
    probe_samples: int = 16
    # Start from these weights instead of a fresh initialization (a bundle's
    # ``model/*.pth``).  The checkpoint's topology and vocabulary must match
    # ``model_config``; optimizer state starts fresh.  Ignored on resume.
    init_checkpoint: str | None = None
    # Exported checkpoints are ``<run_dir>/<export_prefix>_<phase>[.best].pth``.
    export_prefix: str = "chronos_enzh"
    extra: dict = field(default_factory=dict)


def lr_at(step: int, total: int, peak: float, warmup_fraction: float, min_ratio: float) -> float:
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def lr_at_fraction(progress: float, peak: float, warmup_fraction: float, min_ratio: float,
                   first_steps: int) -> float:
    """Warmup + cosine as a function of elapsed budget fraction in [0, 1]."""
    progress = min(1.0, max(0.0, progress))
    wf = max(1e-6, warmup_fraction)
    if progress < wf:
        return peak * max(progress / wf, 1e-3 * min(first_steps + 1, 10))
    t = (progress - wf) / max(1e-6, 1 - wf)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def _atomic_write(path: str, write) -> None:
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def _snapshot_arrays(tree) -> dict:
    return {k: np.array(v.astype(mx.float32)) if v.dtype == mx.bfloat16 else np.array(v)
            for k, v in tree_flatten(tree)}


def _snapshot_dtypes(tree) -> dict:
    return {k: str(v.dtype) for k, v in tree_flatten(tree)}


def _restore_arrays(flat: dict, dtypes: dict):
    items = []
    for k, v in flat.items():
        arr = mx.array(v)
        if dtypes.get(k) == "mlx.core.bfloat16":
            arr = arr.astype(mx.bfloat16)
        items.append((k, arr))
    return tree_unflatten(items)


def _optimizer_snapshot(optimizer) -> dict:
    flat = {k: v for k, v in tree_flatten(optimizer.state) if isinstance(v, mx.array)}
    return {"arrays": {k: np.array(v) for k, v in flat.items()},
            "dtypes": {k: str(v.dtype) for k, v in flat.items()}}


def _optimizer_restore(optimizer, model, snap: dict) -> None:
    optimizer.init(model.trainable_parameters())       # recreates python-side decay flags
    current = dict(tree_flatten(optimizer.state))
    for k, v in snap["arrays"].items():
        if k in current:
            current[k] = mx.array(v)
    optimizer.state = tree_unflatten(list(current.items()))


def routing_metrics(router: mx.array | None, top_k: int) -> dict:
    """Load balance and confidence over every dispatch ``[B, S, V, E]``."""
    if router is None:
        return {}
    E = router.shape[-1]
    flat = router.reshape(-1, router.shape[2], E).astype(mx.float32)        # [N, V, E]
    top = mx.argpartition(-flat, kth=top_k - 1, axis=-1)[..., :top_k]
    load = (top[..., None] == mx.arange(E)).astype(mx.float32).sum(-2).mean(0) / top_k   # [V, E]
    importance = flat.mean(0)
    entropy = -(flat * mx.log(mx.maximum(flat, 1e-9))).sum(-1).mean()
    cv = (load.std(-1) / mx.maximum(load.mean(-1), 1e-9)).mean()
    dead = (load.max(0) < (0.05 / E)).sum()
    mx.eval(load, importance, entropy, cv, dead)
    return {"route_entropy": float(entropy.item()), "route_entropy_max": float(math.log(E)),
            "load_cv": float(cv.item()), "dead_experts": int(dead.item()),
            "max_expert_load": float(load.max().item() * E)}


def recurrent_metrics(state: dict | None) -> dict:
    if not state:
        return {}
    out = {}
    for key in ("ponder_loss", "distill_loss", "stability_loss", "contraction", "early_exit_ratio"):
        if key in state:
            out["rec_" + key] = float(mx.array(state[key]).item())
    if "depth_used" in state:
        d = state["depth_used"].astype(mx.float32)
        out["rec_depth_mean"] = float(d.mean().item())
        out["rec_depth_per_group"] = [round(float(v), 3) for v in np.array(d.mean(axis=(0, 1))).tolist()]
    if "halting_probs" in state:
        h = state["halting_probs"].astype(mx.float32)          # [B, S, G, D] accumulated mass
        out["rec_halt_mass_by_step"] = [round(float(v), 3) for v in np.array(h.mean(axis=(0, 1, 2))).tolist()]
    return out


class PretrainSource:
    def __init__(self, shards_dir, seq_len, seed):
        from chronos.data.corpus.shards import PackedTokenStream
        self.stream = PackedTokenStream(os.path.join(shards_dir, "pretrain", "train"), seq_len, seed=seed)
        self.valid = PackedTokenStream(os.path.join(shards_dir, "pretrain", "valid"), seq_len, seed=seed + 1)

    def batch(self, micro_tokens: int):
        rows = max(1, micro_tokens // self.stream.seq_len)
        arr = self.stream.batch(rows)
        return arr[:, :-1], arr[:, 1:]                         # inputs, next-token labels

    def valid_batches(self, n: int, micro_tokens: int):
        rng_state = self.valid.rng.bit_generator.state
        self.valid.rng = np.random.default_rng(0)               # fixed validation windows
        rows = max(1, micro_tokens // self.valid.seq_len)
        out = [self.valid.batch(rows) for _ in range(n)]
        self.valid.rng.bit_generator.state = rng_state
        return [(a[:, :-1], a[:, 1:]) for a in out]

    def rng_state(self):
        return self.stream.rng.bit_generator.state

    def set_rng_state(self, state):
        self.stream.rng.bit_generator.state = state


class SFTSource:
    def __init__(self, shards_dir, seed):
        from chronos.data.corpus.shards import SFTRecords
        self.records = SFTRecords(os.path.join(shards_dir, "sft", "train"), seed=seed)
        self.valid = SFTRecords(os.path.join(shards_dir, "sft", "valid"), seed=seed + 1)

    @staticmethod
    def _shift(ids, labels):
        return ids[:, :-1], labels[:, 1:]                      # labels already -100 outside replies

    def batch(self, micro_tokens: int):
        return self._shift(*self.records.batch(micro_tokens))

    def valid_batches(self, n: int, micro_tokens: int):
        state = self.valid.rng.bit_generator.state
        self.valid.rng = np.random.default_rng(0)
        out = [self._shift(*self.valid.batch(micro_tokens)) for _ in range(n)]
        self.valid.rng.bit_generator.state = state
        return out

    def rng_state(self):
        return self.records.rng.bit_generator.state

    def set_rng_state(self, state):
        self.records.rng.bit_generator.state = state


def check_init_checkpoint(path: str, config) -> None:
    """Fail fast when ``init_checkpoint`` does not fit ``model_config``.

    ``_load_mlx_from_checkpoint`` silently starts from random weights when the
    file is missing.  A mismatched topology or vocabulary would instead fail
    deep inside ``load_state_dict``, so both are checked here.
    """
    from chronos.model.checkpoint import SNIFFABLE_TOPOLOGY_FIELDS, UNSNIFFABLE_REQUIRED_FIELDS, read_checkpoint_config
    if not os.path.isfile(path):
        raise FileNotFoundError(f"init_checkpoint not found: {path}")
    saved, _ = read_checkpoint_config(path)
    if not saved:
        return                                   # no sidecar: load_state_dict_controlled still checks shapes
    keys = SNIFFABLE_TOPOLOGY_FIELDS + UNSNIFFABLE_REQUIRED_FIELDS + (
        "recurrent_depth", "prelude_layers", "coda_layers", "recurrent_attention_mode", "rope_layout")
    diff = {k: (saved[k], getattr(config, k, None)) for k in keys
            if k in saved and saved[k] is not None and saved[k] != getattr(config, k, None)}
    if diff:
        raise ValueError(f"init_checkpoint {path} does not match model_config: "
                         + ", ".join(f"{k}: checkpoint {a!r} vs spec {b!r}" for k, (a, b) in diff.items()))


class LongRun:
    def __init__(self, spec: RunSpec, log=print):
        from chronos.model.config import ChronosConfig
        from chronos.mlx.model import ChronosMLXModel
        from chronos.mlx.training.optim import MixedPrecisionAdamW
        from chronos.mlx.training.trainer import _load_mlx_from_checkpoint
        self.spec, self.log = spec, log
        os.makedirs(spec.run_dir, exist_ok=True)
        self.config = ChronosConfig(**spec.model_config)
        import torch
        torch.manual_seed(spec.seed)        # reproducible initialization (built in Torch, then converted)
        mx.random.seed(spec.seed)
        init = spec.init_checkpoint
        if init:
            init = os.path.abspath(os.path.expanduser(init))
            check_init_checkpoint(init, self.config)
        self.model = _load_mlx_from_checkpoint(self.config, init, dtype_name=spec.dtype)
        self.model.train()
        self.optimizer = MixedPrecisionAdamW(learning_rate=0.0, weight_decay=spec.phases[0].weight_decay)
        self.optimizer.init(self.model.trainable_parameters())
        self.state = {"phase_index": 0, "phase_step": 0, "phase_total": None, "elapsed_s": 0.0,
                      "wall_s": 0.0,
                      "tokens_seen": 0, "phase_tokens_per_s": {}, "rng": {}, "best_valid": {}}
        self.metrics_path = os.path.join(spec.run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(spec.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(os.path.join(spec.run_dir, "run_spec.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(spec), fh, ensure_ascii=False, indent=2)
        from chronos.mlx.training import telemetry
        self.telemetry = telemetry
        self.heartbeat = telemetry.Heartbeat(spec.run_dir)
        self.precision = getattr(self.model, "_chronos_precision", None)
        self.runtime = telemetry.runtime_info(self.model, spec, self.precision)
        try:
            telemetry._atomic_json(os.path.join(spec.run_dir, "runtime.json"), self.runtime)
        except OSError:
            pass
        self.parameters_path = os.path.join(spec.run_dir, "parameters.jsonl")
        self.heartbeat.beat("starting", precision=self.runtime["precision"].get("compute"),
                            code=self.runtime["code"]["sha256"])

    # ---- persistence -------------------------------------------------------
    def save_state(self, tag="latest"):
        if getattr(self, "_run_start", None) is not None:
            self.state["wall_s"] = time.monotonic() - self._run_start
        source = getattr(self, "_active_source", None)
        if source is not None:
            phase = self.spec.phases[min(self.state["phase_index"], len(self.spec.phases) - 1)]
            self.state["rng"][phase.name] = source.rng_state()

        path = os.path.join(self.ckpt_dir, f"{tag}.state.pkl")
        params = self.model.parameters()
        payload = {"state": self.state, "params": _snapshot_arrays(params), "dtypes": _snapshot_dtypes(params),
                   "optimizer": _optimizer_snapshot(self.optimizer)}
        _atomic_write(path, lambda tmp: pickle.dump(payload, open(tmp, "wb"), protocol=5))
        return path

    def try_resume(self) -> bool:
        path = os.path.join(self.ckpt_dir, "latest.state.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # written by this run; not untrusted input
        self.model.update(_restore_arrays(payload["params"], payload["dtypes"]))
        _optimizer_restore(self.optimizer, self.model, payload["optimizer"])
        self.state = payload["state"]
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.log(f"[run] resumed phase={self.state['phase_index']} step={self.state['phase_step']} "
                 f"elapsed={self.state['elapsed_s'] / 3600:.2f}h")
        return True

    def export_checkpoint(self, name: str, stage: str):
        from chronos.mlx.training.trainer import _save_mlx_checkpoint
        out = os.path.join(self.spec.run_dir, f"{name}.pth")
        _save_mlx_checkpoint(self.model, self.config, out, stage=stage)
        cfg_path = out.replace(".pth", ".config.json")
        if os.path.exists(cfg_path):  # record the run's own tokenizer, not minimind's
            from chronos.model.checkpoint import tokenizer_metadata
            meta = json.load(open(cfg_path, encoding="utf-8"))
            meta["tokenizer"] = tokenizer_metadata(os.path.abspath(self.spec.tokenizer_dir))
            json.dump(meta, open(cfg_path, "w", encoding="utf-8"), indent=2, sort_keys=True)
        return out

    def _emit(self, record: dict):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _beat(self, state: str | None = None, **fields):
        st = self.state
        phases = self.spec.phases
        pi = min(int(st.get("phase_index", 0)), len(phases) - 1)
        run_start = getattr(self, "_run_start", None)
        self.heartbeat.beat(
            state, phase=phases[pi].name if st.get("phase_index", 0) < len(phases) else "done",
            phase_index=int(st.get("phase_index", 0)), step=int(st.get("phase_step", 0)), total=st.get("phase_total"),
            phase_wall_s=st.get("phase_wall_s"), phase_budget_s=st.get("phase_budget_s"),
            run_wall_s=(time.monotonic() - run_start) if run_start is not None else st.get("wall_s"),
            tokens_seen=st.get("tokens_seen"), tokens_processed=st.get("tokens_processed"),
            rollbacks=int(st.get("rollbacks", 0)), lr_scale=float(st.get("lr_scale", 1.0)),
            best_valid=st.get("best_valid"), **fields)

    def _tensor_telemetry(self, phase, step: int, grads, before, accum: int, lr: float, norm: float):
        """One ``parameters.jsonl`` row: per-tensor weight RMS/|max|, pre-clip
        gradient RMS, relative update, per-group norms, expert grid, routing."""
        tel = self.telemetry
        t0 = time.monotonic()
        params = dict(tree_flatten(self.model.parameters()))
        stats = tel.tensor_stats(params, dict(tree_flatten(grads)) if grads is not None else None,
                                 dict(tree_flatten(before)) if before is not None else None,
                                 grad_scale=1.0 / max(1, accum))
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": phase.name, "step": step, "lr": lr,
               "grad_norm": norm, **stats, "experts": tel.expert_table(stats["tensors"]),
               "routing": tel.routing_snapshot(self.model)}
        row["telemetry_s"] = round(time.monotonic() - t0, 3)
        with open(self.parameters_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    # ---- training ----------------------------------------------------------
    def _loss_fn(self, model, ids, labels):
        from chronos.mlx.training.trainer import _chronos_regularized_loss, draft_exit_kl, draft_exit_logits
        draft = draft_exit_logits(model, ids, model.config)       # None unless draft_exit_kl_coef > 0
        logits, lookahead, _ = model(ids, logits_dtype=mx.float32)
        lg = logits.reshape(-1, logits.shape[-1])
        lb = labels.reshape(-1)
        mask = lb != -100
        ce = nn.losses.cross_entropy(lg, mx.where(mask, lb, 0), reduction="none")
        ce = (ce * mask).sum() / mx.maximum(mask.sum(), 1)
        # Labels here are pre-aligned with the logits (no shift).
        draft_kl = None if draft is None else draft_exit_kl(draft, logits, labels != -100)
        return _chronos_regularized_loss(model, ce, lookahead, model.config, metrics={"ce": ce, "primary": ce},
                                         draft_kl=draft_kl)

    def _source(self, phase: PhaseSpec):
        if phase.name == "pretrain":
            return PretrainSource(self.spec.shards_dir, phase.seq_len, self.spec.seed)
        return SFTSource(self.spec.shards_dir, self.spec.seed)

    def _micro_step(self, grad_fn, ids, labels):
        loss, grads = grad_fn(self.model, mx.array(ids, dtype=mx.int32), mx.array(labels, dtype=mx.int32))
        return loss, grads

    def _apply(self, grads, n_micro, phase, lr):
        from chronos.mlx.training.trainer import _scale_grads
        grads = _scale_grads(grads, 1.0 / n_micro)
        grads, norm = optim.clip_grad_norm(grads, phase.grad_clip)
        self.optimizer.learning_rate = lr
        self.optimizer.weight_decay = phase.weight_decay
        self.optimizer.update(self.model, grads)
        return norm

    def _validate(self, source, phase, n_batches=8) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for ids, labels in source.valid_batches(n_batches, phase.micro_tokens):
            logits, _, _ = self.model(mx.array(ids, dtype=mx.int32), logits_dtype=mx.float32)
            lb = mx.array(labels, dtype=mx.int32).reshape(-1)
            mask = lb != -100
            ce = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         mx.where(mask, lb, 0), reduction="none")
            s, n = (ce * mask).sum(), mask.sum()
            mx.eval(s, n)
            total, count = total + float(s.item()), count + int(n.item())
        self.model.train()
        return total / max(count, 1)

    def run_phase(self, pi: int, deadline_s: float, max_steps: int | None = None):
        """Train phase ``pi`` until its step total, its deadline, or ``max_steps``
        more optimizer updates (used by tests and staged runs)."""
        from chronos.mlx.training.trainer import _add_grads, _collect_router_probs, _tree_all_finite
        phase = self.spec.phases[pi]
        source = self._source(phase)
        if self.state["rng"].get(phase.name):
            source.set_rng_state(self.state["rng"][phase.name])
        accum = max(1, phase.tokens_per_step // phase.micro_tokens)
        grad_fn = nn.value_and_grad(self.model, self._loss_fn)
        controller = self.model.recurrent_controller
        top_k = int(self.config.num_experts_per_tok)
        t_phase = time.monotonic()
        last_ckpt = last_val = time.monotonic()
        window_tokens, window_processed, window_t = 0, 0, time.monotonic()
        self._active_source = source
        stop_at = None if max_steps is None else self.state["phase_step"] + int(max_steps)
        if self.state["phase_step"] == 0 and not self.state.get("phase_budget_s"):
            self.state["phase_budget_s"] = max(1.0, deadline_s - time.monotonic())
            self.state["phase_wall_s"] = 0.0
        last_wall = time.monotonic()
        last_tensors = -math.inf                    # first update after (re)start is sampled
        tensor_every = float(getattr(self.spec, "tensor_stats_every_s", 0.0) or 0.0)
        self._emit({"phase": phase.name, "event": "phase_start", "step": self.state["phase_step"],
                    "deadline_h": round((deadline_s - time.monotonic()) / 3600, 3)})
        self._beat("running")
        while True:
            step, total = self.state["phase_step"], self.state["phase_total"]
            if total is not None and step >= total and self.spec.schedule != "time":
                break
            if stop_at is not None and step >= stop_at:
                self.state["rng"][phase.name] = source.rng_state()
                return
            if time.monotonic() >= deadline_s:
                self.log(f"[run] phase {phase.name} reached its wall-clock budget at step {step}")
                break
            now_wall = time.monotonic()
            self.state["phase_wall_s"] = self.state.get("phase_wall_s", 0.0) + (now_wall - last_wall)
            last_wall = now_wall
            time_based = self.spec.schedule == "time" and self.state.get("phase_budget_s")
            progress = (self.state["phase_wall_s"] / self.state["phase_budget_s"]) if time_based else \
                step / max(total or 1_000_000, 1)
            controller.set_training_progress(int(progress * 1_000_000), 1_000_000)
            # MLX's global RNG (cache-miss simulation, dropout) cannot be
            # snapshotted, so make each step's randomness a pure function of
            # (seed, phase, step, rollbacks): resume is then bit-exact.
            rb = int(self.state.get("rollbacks", 0))
            mx.random.seed((self.spec.seed * 1_000_003 + pi * 100_003 + step + rb * 7_919) % (2 ** 31))
            if time_based:
                lr = lr_at_fraction(progress, phase.peak_lr, phase.warmup_fraction, phase.min_lr_ratio, step)
            else:
                lr = lr_at(step, total or 1_000_000, phase.peak_lr, phase.warmup_fraction, phase.min_lr_ratio)
            lr *= float(self.state.get("lr_scale", 1.0))
            t0 = time.monotonic()
            acc, tokens, processed, loss_sum = None, 0, 0, mx.zeros(())
            for _ in range(accum):
                ids, labels = source.batch(phase.micro_tokens)
                loss, grads = self._micro_step(grad_fn, ids, labels)
                acc = grads if acc is None else _add_grads(acc, grads)
                loss_sum = loss_sum + loss
                tokens += int((labels != -100).sum())      # supervised tokens
                processed += int(ids.size)                  # forward/backward tokens
                mx.eval(acc, loss_sum)          # bound the lazy graph to one micro-batch
            # Tensor telemetry keeps references to the pre-update weights (MLX
            # arrays are immutable: no copy) and the FP32 accumulated grads.
            want_tensors = tensor_every > 0 and time.monotonic() - last_tensors >= tensor_every
            before = self.model.parameters() if want_tensors else None
            norm = self._apply(acc, accum, phase, lr)
            mx.eval(self.model.parameters(), self.optimizer.state, norm)
            dt = time.monotonic() - t0
            if not math.isfinite(float(norm.item())) or not math.isfinite(float(loss_sum.item())):
                # Restore the last snapshot but keep wall-clock progress, halve
                # the LR and change the data/RNG stream so the failing step is
                # not replayed verbatim.
                keep = {k: self.state.get(k) for k in ("phase_wall_s", "phase_budget_s")}
                rollbacks = int(self.state.get("rollbacks", 0)) + 1
                lr_scale = max(0.125, float(self.state.get("lr_scale", 1.0)) * 0.5)
                self.log(f"[run] non-finite step {step}; rollback #{rollbacks}, lr_scale={lr_scale}")
                if not self.try_resume():
                    raise FloatingPointError("non-finite loss before first snapshot")
                self.state.update(keep, rollbacks=rollbacks, lr_scale=lr_scale)
                if self.state["rng"].get(phase.name):
                    source.set_rng_state(self.state["rng"][phase.name])
                source.batch(phase.micro_tokens)          # skip ahead in the data stream
                if rollbacks > 8:
                    raise FloatingPointError("too many non-finite rollbacks")
                self._emit({"phase": phase.name, "event": "rollback", "step": step, "rollbacks": rollbacks,
                            "lr_scale": lr_scale})
                self._beat("running", last_rollback=time.time())
                last_wall = time.monotonic()
                continue
            self.state["phase_step"] = step + 1
            if want_tensors:
                try:
                    self._tensor_telemetry(phase, step + 1, acc, before, accum, lr, float(norm.item()))
                except Exception as exc:  # noqa: BLE001 - telemetry must never stop training
                    self.log(f"[telemetry] tensor stats skipped: {exc!r}")
                last_tensors = time.monotonic()
            before = None
            self.state["tokens_seen"] += tokens
            self.state["tokens_processed"] = self.state.get("tokens_processed", 0) + processed
            self.state["elapsed_s"] += dt
            window_tokens += tokens
            window_processed += processed
            refresh_eta = self.spec.schedule == "time" and (step + 1) % 50 == 0
            if (total is None and step + 1 >= 6) or refresh_eta:
                # Throughput probe (first time) or ETA refresh (time schedule).
                tps = window_tokens / max(time.monotonic() - window_t, 1e-6)
                remaining = max(0.0, deadline_s - time.monotonic())
                self.state["phase_total"] = step + 1 + int(remaining * tps / max(tokens, 1) * 0.97)
                if phase.name not in self.state["phase_tokens_per_s"]:
                    self.log(f"[run] {phase.name}: {tps:,.0f} tok/s → ~{self.state['phase_total']} steps "
                             f"(~{self.state['phase_total'] * tokens / 1e9:.2f}B tokens)")
                self.state["phase_tokens_per_s"][phase.name] = tps
            if (step + 1) % self.spec.log_every_steps == 0 or step < 3:
                comps = {k: round(float(v.item()), 5) for k, v in getattr(self.model, "_last_loss_components", {}).items()}
                rec = {"phase": phase.name, "step": step + 1, "total": self.state["phase_total"],
                       "lr": lr, "grad_norm": float(norm.item()), "loss": float(loss_sum.item()) / accum,
                       "tokens_per_s": window_tokens / max(time.monotonic() - window_t, 1e-6),
                       "processed_tokens_per_s": window_processed / max(time.monotonic() - window_t, 1e-6),
                       "tokens_processed": self.state["tokens_processed"],
                       "tokens_seen": self.state["tokens_seen"], "step_s": dt,
                       "peak_mem_gb": mx.get_peak_memory() / 1e9, "active_mem_gb": mx.get_active_memory() / 1e9,
                       "cache_mem_gb": mx.get_cache_memory() / 1e9,
                       "phase_wall_s": self.state.get("phase_wall_s"), "phase_budget_s": self.state.get("phase_budget_s"),
                       "lr_scale": float(self.state.get("lr_scale", 1.0)), "rollbacks": int(self.state.get("rollbacks", 0)),
                       **comps, **recurrent_metrics(getattr(self.model, "last_recurrent_state", None)),
                       **routing_metrics(_collect_router_probs(self.model), top_k)}
                self._emit(rec)
                self.log(f"[{phase.name}] {step + 1}/{self.state['phase_total']} loss={rec['loss']:.4f} "
                         f"ce={comps.get('primary_loss', 0):.4f} gn={rec['grad_norm']:.2f} lr={lr:.2e} "
                         f"{rec['tokens_per_s']:,.0f} tok/s depth={rec.get('rec_depth_mean', 0):.2f}")
                window_tokens, window_processed, window_t = 0, 0, time.monotonic()
                self._beat("running", last={k: rec.get(k) for k in (
                    "loss", "primary_loss", "grad_norm", "lr", "tokens_per_s", "step_s", "peak_mem_gb",
                    "active_mem_gb", "rec_depth_mean", "load_cv")})
            now = time.monotonic()
            if now - last_val >= self.spec.validate_every_s:
                self._beat("evaluating")
                self._periodic_eval(source, phase)
                last_val = time.monotonic()
                self._beat("running")
            if now - last_ckpt >= self.spec.checkpoint_every_s:
                self._beat("checkpointing")
                self.state["rng"][phase.name] = source.rng_state()
                self.save_state()
                last_ckpt = time.monotonic()
                self._beat("running", last_checkpoint=time.time())
        self.state["rng"][phase.name] = source.rng_state()
        self._beat("evaluating")
        self._periodic_eval(source, phase)
        self.export_checkpoint(f"{self.spec.export_prefix}_{phase.name}", stage=phase.name)
        self._emit({"phase": phase.name, "event": "phase_end", "step": self.state["phase_step"],
                    "tokens_seen": self.state["tokens_seen"]})

    def _periodic_eval(self, source, phase):
        ce = self._validate(source, phase)
        rec = {"phase": phase.name, "step": self.state["phase_step"], "event": "validation", "valid_ce": ce,
               "valid_ppl": math.exp(min(ce, 30))}
        if phase.name == "sft" and self.spec.bleu_samples:
            rec.update(self.translation_eval(self.spec.bleu_samples))
        elif phase.name == "pretrain" and getattr(self.spec, "probe_samples", 0):
            rec.update(self._pretrain_probe(int(self.spec.probe_samples)))
        best = self.state["best_valid"].get(phase.name)
        if best is None or ce < best:
            self.state["best_valid"][phase.name] = ce
            self.export_checkpoint(f"{self.spec.export_prefix}_{phase.name}.best", stage=phase.name)
        self._emit(rec)
        self.log(f"[eval] {json.dumps(rec, ensure_ascii=False)}")

    def _pretrain_probe(self, n: int) -> dict:
        from chronos.mlx.training.inspect_checkpoint import translation_probe
        self.model.eval()
        try:
            probe = translation_probe(asdict(self.spec), self.model, "pretrain", n)
        except Exception as exc:  # noqa: BLE001 - a probe failure must not stop training
            self.log(f"[eval] pretrain probe skipped: {exc!r}")
            return {}
        finally:
            self.model.train()
        out = {}
        with open(os.path.join(self.spec.run_dir, "samples.jsonl"), "a", encoding="utf-8") as fh:
            for direction in ("en_zh", "zh_en"):
                res = probe.get(direction)
                if not isinstance(res, dict):
                    continue
                out[f"probe_{direction}_bleu"], out[f"probe_{direction}_chrf"] = res["bleu"], res["chrf"]
                for sample in res.get("samples", [])[:3]:
                    fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": "pretrain",
                                         "step": self.state["phase_step"], "direction": direction,
                                         "set": probe.get("probe_set"), **sample}, ensure_ascii=False) + "\n")
        if out:
            out["probe_n"], out["probe_s"] = n, probe.get("probe_s")
        return out

    # ---- translation quality (greedy, KV cache, MLX GPU) ----------------------
    def _tokenizer(self):
        if not hasattr(self, "_tok"):
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(os.path.join(self.spec.tokenizer_dir, "tokenizer.json"))
        return self._tok

    def translate(self, sources: list[str], max_new_tokens: int = 160) -> list[str]:
        from chronos.data.corpus.shards import SFT_PROMPT_EN_ZH
        tok = self._tokenizer()
        end = tok.token_to_id("<|im_end|>")
        self.model.eval()
        out = []
        for src in sources:
            prompt = f"<|im_start|>user\n{SFT_PROMPT_EN_ZH}{src}<|im_end|>\n<|im_start|>assistant\n"
            ids = tok.encode(prompt, add_special_tokens=False).ids
            src_len = len(tok.encode(src, add_special_tokens=False).ids)
            limit = min(max_new_tokens, int(1.6 * src_len) + 16)   # zh output ≲ 1.6× en tokens
            logits, _, cache = self.model(mx.array([ids], dtype=mx.int32))
            gen = []
            for _ in range(limit):
                nxt = int(mx.argmax(logits[0, -1]).item())
                if nxt == end:
                    break
                gen.append(nxt)
                logits, _, cache = self.model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
            out.append(tok.decode(gen, skip_special_tokens=True))
        self.model.train()
        return out

    def translation_eval(self, n: int) -> dict:
        import sacrebleu
        from chronos.data.corpus.evalsets import load_eval_sets
        raw = self.spec.extra.get("raw_root")
        if not raw:
            return {}
        pairs = load_eval_sets(raw, ["flores_devtest"]).get("flores_devtest", [])[:n]
        if not pairs:
            return {}
        t0 = time.monotonic()
        hyps = self.translate([en for en, _zh in pairs])
        refs = [zh for _en, zh in pairs]
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        with open(os.path.join(self.spec.run_dir, "samples.jsonl"), "a", encoding="utf-8") as fh:
            for (en, zh), hyp in list(zip(pairs, hyps))[:5]:
                fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": self.spec.phases[
                    min(self.state["phase_index"], len(self.spec.phases) - 1)].name, "step": self.state["phase_step"],
                    "direction": "en_zh", "set": "flores_devtest", "src": en, "ref": zh, "hyp": hyp},
                    ensure_ascii=False) + "\n")
        return {"flores_devtest_n": len(pairs), "bleu_zh": round(bleu.score, 2), "chrf": round(chrf.score, 2),
                "eval_s": round(time.monotonic() - t0, 1)}

    # ---- whole run -----------------------------------------------------------
    def run(self):
        # Pin every lazily-used module now so source edits during a multi-day
        # run cannot change the code that evaluates it.
        import sacrebleu  # noqa: F401
        import tokenizers  # noqa: F401
        import chronos.data.corpus.evalsets  # noqa: F401
        import chronos.data.corpus.shards  # noqa: F401
        import chronos.mlx.training.inspect_checkpoint  # noqa: F401
        try:
            return self._run()
        except KeyboardInterrupt:
            self._beat("stopped", error="interrupted")
            raise
        except BaseException as exc:
            self._beat("failed", error=f"{type(exc).__name__}: {exc}"[:800])
            raise

    def _run(self):
        resumed = self.try_resume()
        if resumed:
            self._emit({"phase": self.spec.phases[min(self.state["phase_index"], len(self.spec.phases) - 1)].name,
                        "event": "resume", "step": self.state["phase_step"],
                        "wall_h": round(self.state.get("wall_s", 0.0) / 3600, 3), "code": self.runtime["code"]["sha256"]})
        budget_s = self.spec.budget_hours * 3600 * (1 - self.spec.eval_reserve_fraction)
        # Wall time already spent (including eval/checkpoint I/O) before a resume.
        start = time.monotonic() - (self.state.get("wall_s", 0.0) if resumed else 0.0)
        self._run_start = start
        phase_end = start
        for pi, phase in enumerate(self.spec.phases):
            phase_end += budget_s * phase.budget_fraction
            if pi < self.state["phase_index"]:
                continue
            self.log(f"[run] phase {phase.name}: deadline in {(phase_end - time.monotonic()) / 3600:.2f}h")
            self.run_phase(pi, phase_end)
            self.state.update(phase_index=pi + 1, phase_step=0, phase_total=None,
                              phase_budget_s=None, phase_wall_s=0.0, lr_scale=1.0)
            self.save_state()
        self._beat("evaluating")
        final = {"event": "final", "tokens_seen": self.state["tokens_seen"],
                 "hours": round((time.monotonic() - start) / 3600, 2)}
        if self.spec.final_eval_samples:
            final.update(self.translation_eval(self.spec.final_eval_samples))
        self._emit(final)
        self._beat("finished", final=final)
        self.log(f"[run] done: {json.dumps(final, ensure_ascii=False)}")
        return final


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Time-budgeted Chronos MLX training run")
    parser.add_argument("--spec", required=True, help="RunSpec JSON (see chronos/mlx/training/recipes)")
    parser.add_argument("--budget-hours", type=float, default=None)
    args = parser.parse_args(argv)
    raw = json.load(open(args.spec, encoding="utf-8"))
    raw["phases"] = [PhaseSpec(**p) for p in raw["phases"]]
    spec = RunSpec(**raw)
    if args.budget_hours is not None:
        spec.budget_hours = args.budget_hours
    LongRun(spec, log=lambda m: print(m, flush=True)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
