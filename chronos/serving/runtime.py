"""Chronos serving runtime: native kernel + SSD-lazy experts + self-speculation.

This is the engine behind the Ollama/OpenAI protocol layer.  It keeps every
Chronos property — recurrent depth, shared_first KV, SSD→RAM→device expert
offload with lookahead prefetch, and recurrent-depth self-speculative
decoding — so a client talking the Ollama protocol gets the same memory and
throughput behaviour as native use.
"""
from __future__ import annotations

import gc
import json
import os
import threading
import time
from dataclasses import dataclass, field

import torch


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return 0.0


@dataclass
class LoadedModel:
    name: str
    path: str
    config: object
    model: object
    tokenizer: object
    engine: object | None
    device: str
    lazy_experts: bool
    loaded_at: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


def _pick_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Inference-only options a server may override on a loaded checkpoint.
# None of them changes the trained weights; all are validated by ChronosConfig.
INFERENCE_OVERRIDE_KEYS = ("kv_cache_dtype", "kv_quant_group_size", "mla_decode", "mla_cache_rope",
                           "router_scope", "max_resident_experts", "ram_budget_mb")


def apply_inference_overrides(config, overrides: dict | None) -> dict:
    """Set validated inference options on ``config``; return what was applied."""
    from chronos.model.config import ChronosConfig
    applied = {k: v for k, v in (overrides or {}).items() if k in INFERENCE_OVERRIDE_KEYS and v is not None}
    unknown = set(overrides or {}) - set(INFERENCE_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"not an inference option: {sorted(unknown)}")
    if applied:
        for key in ("max_resident_experts", "ram_budget_mb"):
            if key in applied and float(applied[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        # Validate against the checkpoint's full config: a partial probe falls
        # back to class defaults (e.g. 4 KV heads) that may not fit the model.
        probe = {k: v for k, v in config.to_dict().items() if k not in applied}
        for key in ("transformers_version", "_name_or_path", "_commit_hash", "architectures", "model_type"):
            probe.pop(key, None)
        checked = ChronosConfig(**probe, **applied)  # raises on an invalid value, normalises aliases
        for key in applied:
            setattr(config, key, getattr(checked, key))
            applied[key] = getattr(checked, key)
    return applied


def _find_tokenizer_dir(ckpt_path: str, meta: dict) -> str:
    from chronos.model.checkpoint import resolve_checkpoint_tokenizer
    return resolve_checkpoint_tokenizer(ckpt_path, fallback=False)


def load_model(path: str, *, name: str | None = None, device: str | None = "auto",
               lazy_experts: bool = True, ram_expert_fraction: float = 0.5,
               expert_cache_root: str | None = None, dtype: str = "auto",
               inference_overrides: dict | None = None) -> LoadedModel:
    """Load a Chronos checkpoint for serving.

    With ``lazy_experts`` (default) the routed experts are sharded to an SSD
    cache and only ``ram_expert_fraction`` of them are ever resident; the
    rest stream in on demand, guided by the lookahead router.  ``dtype``
    picks the single compute dtype (``chronos.precision.resolve_torch``):
    weights, activations, KV cache and expert pages all use it.
    """
    from transformers import PreTrainedTokenizerFast
    from chronos.model.checkpoint import (
        chronos_config_from_checkpoint, load_checkpoint_state_dict, load_state_dict_controlled,
        read_checkpoint_config,
    )
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.precision import place_torch_model
    meta, _ = read_checkpoint_config(path)
    meta_full = {}
    cfg_path = os.path.splitext(path)[0] + ".config.json"
    if os.path.exists(cfg_path):
        meta_full = json.load(open(cfg_path, encoding="utf-8"))
    config, _sources = chronos_config_from_checkpoint(path)
    applied = apply_inference_overrides(config, inference_overrides)
    tok_dir = _find_tokenizer_dir(path, meta_full)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    device = _pick_device(device)
    weights = load_checkpoint_state_dict(path, map_location="cpu")
    model = ChronosForCausalLM(config)
    engine = None
    if lazy_experts and getattr(config, "use_moe", True):
        from chronos.runtime.inference_engine import ChronosInferenceEngine
        from ui.tabs.inference_tab import (
            _apply_offload_expert_budget, _checkpoint_expert_cache_dir,
            _replace_torch_experts_with_placeholders,
        )
        base = {k: v for k, v in weights.items() if ".mlp.experts." not in k}
        load_state_dict_controlled(model, base, allow_missing_substrings=(".mlp.experts.",))
        _replace_torch_experts_with_placeholders(model, config)
        model, precision = place_torch_model(model, device, dtype)
        model = model.eval()
        cache_dir = expert_cache_root or _checkpoint_expert_cache_dir(path, config)
        engine = ChronosInferenceEngine(model, config, ssd_dir=cache_dir)
        _apply_offload_expert_budget(engine, config, ram_expert_fraction)
        engine.setup_from_state_dict(weights, warm_expert_ids=[])
    else:
        load_state_dict_controlled(model, weights)
        model, precision = place_torch_model(model, device, dtype)
        model = model.eval()
    del weights
    gc.collect()
    return LoadedModel(name=name or os.path.splitext(os.path.basename(path))[0], path=os.path.abspath(path),
                       config=config, model=model, tokenizer=tokenizer, engine=engine, device=device,
                       lazy_experts=engine is not None,
                       meta={"tokenizer_dir": tok_dir, "precision": precision.as_dict(),
                             "inference_overrides": applied, **meta_full})


class ChronosRuntime:
    """Thread-safe model registry + generation for the protocol server."""

    def __init__(self, models_dir: str | None = None, *, device: str = "auto", lazy_experts: bool = True,
                 ram_expert_fraction: float = 0.5, keep_alive_s: float = 300.0, dtype: str = "auto",
                 inference_overrides: dict | None = None):
        self.models_dir = models_dir
        self.inference_overrides = dict(inference_overrides or {})
        self.dtype = dtype
        self.device, self.lazy_experts = device, lazy_experts
        self.ram_expert_fraction = ram_expert_fraction
        self.keep_alive_s = keep_alive_s
        self.loaded: dict[str, LoadedModel] = {}
        self.last_used: dict[str, float] = {}
        self.lock = threading.RLock()

    # ---- registry ------------------------------------------------------------
    def available(self) -> dict[str, str]:
        """name → checkpoint path for every ``*.pth`` with a config sidecar."""
        out = {}
        if self.models_dir and os.path.isdir(self.models_dir):
            for root, _dirs, files in os.walk(self.models_dir):
                for f in files:
                    if f.endswith(".pth") and os.path.exists(os.path.join(root, f[:-4] + ".config.json")):
                        out[f[:-4]] = os.path.join(root, f)
        for name, lm in self.loaded.items():
            out.setdefault(name, lm.path)
        return out

    def get(self, name: str) -> LoadedModel:
        with self.lock:
            name = name.split(":")[0]
            if name not in self.loaded:
                path = self.available().get(name)
                if path is None:
                    raise KeyError(f"model '{name}' not found")
                self.loaded[name] = load_model(path, name=name, device=self.device,
                                               lazy_experts=self.lazy_experts,
                                               ram_expert_fraction=self.ram_expert_fraction,
                                               dtype=self.dtype,
                                               inference_overrides=self.inference_overrides)
            self.last_used[name] = time.time()
            return self.loaded[name]

    def unload(self, name: str) -> bool:
        with self.lock:
            lm = self.loaded.pop(name, None)
            if lm and lm.engine is not None:
                try:
                    lm.engine.teardown()
                except Exception:
                    pass
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return lm is not None

    # ---- generation ------------------------------------------------------------
    def _forward_fn(self, lm: LoadedModel, telemetry: dict):
        engine = lm.engine
        if engine is None:
            def fwd(ids, cache, depth):
                return lm.model(ids, past_key_values=cache, use_cache=True, recurrent_depth=depth)
            return fwd
        cm = engine.cache_manager
        state = {"lookahead": None}

        def fwd(ids, cache, depth):
            # Lookahead-guided residency: prefetch next experts to RAM, promote
            # the predicted set, then run with the current availability masks.
            if state["lookahead"] is not None:
                predicted = engine._predict_expert_ids(state["lookahead"], all_future=True,
                                                       capacity=int(cm.expert_store.ram_capacity))
                cm.prefetch_experts_to_ram(predicted)
                ready = engine._promotion_ready_ids(predicted[: int(cm.expert_store.vram_capacity)])
                if ready:
                    cm.ensure_resident(ready)
            masks = cm.availability_masks_all_layers()
            out = lm.model(ids.to(lm.device), past_key_values=cache, use_cache=True,
                           recurrent_depth=depth, available_expert_masks=masks)
            state["lookahead"] = out.lookahead_probs
            rt = engine._consume_moe_runtime_stats()
            telemetry["fallback_weight_mass"] += float(rt["fallback_executed_weight_mass"])
            telemetry["on_demand_loads"] += len(rt["on_demand_experts"])
            telemetry["forwards"] += 1
            return out
        return fwd

    def encode_chat(self, lm: LoadedModel, messages: list[dict]) -> list[int]:
        text = lm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return lm.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode_prompt(self, lm: LoadedModel, prompt: str, system: str | None = None, raw: bool = False):
        if raw:
            return lm.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.encode_chat(lm, msgs)

    def generate_stream(self, name: str, input_ids: list[int], *, max_new_tokens: int = 256,
                        temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, seed: int | None = None,
                        speculative: bool = True, draft_depth: int = 1, gamma: int = 4,
                        miss_policy: str = "on_demand", stop: list[str] | None = None):
        """Yield ``{"text": delta}`` chunks, then a final ``{"done": True, ...stats}``."""
        from chronos.runtime.speculative import speculative_generate, speculative_supported
        lm = self.get(name)
        tok = lm.tokenizer
        eos = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"))
               if isinstance(i, int) and i >= 0]
        telemetry = {"fallback_weight_mass": 0.0, "on_demand_loads": 0, "forwards": 0}
        import queue
        chunks: queue.Queue = queue.Queue()
        cancel = threading.Event()
        result = {}
        ids = torch.tensor([input_ids], device=lm.device)
        do_sample = float(temperature) > 0
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
        depth = int(getattr(lm.config, "max_inference_depth", None) or lm.config.recurrent_depth)
        use_spec = bool(speculative) and speculative_supported(lm.config) and int(draft_depth) < depth

        def worker():
            with self.lock:   # one generation at a time per process (single GPU)
                rss0 = _rss_mb()
                if lm.engine is not None:
                    lm.engine._runtime_stats = {"on_demand_loads": 0, "on_demand_load_time_s": 0.0,
                                                "sync_ssd_loads": 0, "async_cold_miss_prefetches": 0,
                                                "resident_vram_hits": 0, "resident_ram_hits": 0,
                                                "on_demand_loaded_experts": [], "on_demand_promoted_experts": []}
                    lm.engine._install_moe_runtime_hooks(lm.engine._normalize_miss_policy(miss_policy))
                generated: list[int] = []
                emitted_text = ""
                t0 = time.perf_counter()

                def on_token(t):
                    nonlocal emitted_text
                    if t in eos:
                        return
                    generated.append(t)
                    text = tok.decode(generated, skip_special_tokens=True)
                    if text.endswith("�"):
                        return                       # wait for the rest of a multi-byte char
                    delta, emitted_text = text[len(emitted_text):], text
                    if delta:
                        chunks.put({"text": delta})
                try:
                    _, stats = speculative_generate(
                        lm.model, ids, max_new_tokens=max_new_tokens,
                        draft_depth=draft_depth if use_spec else depth, gamma=gamma,
                        do_sample=do_sample, temperature=max(float(temperature), 1e-6),
                        top_k=int(top_k or 0), top_p=float(top_p or 1.0), eos_token_id=eos,
                        generator=gen, on_token=on_token, forward_fn=self._forward_fn(lm, telemetry),
                        should_stop=cancel.is_set)
                    result.update(stats.as_dict())
                except Exception as exc:                 # surfaced to the client as an error chunk
                    result["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    if lm.engine is not None:
                        lm.engine._clear_moe_runtime_hooks()
                    elapsed = time.perf_counter() - t0
                    store = getattr(getattr(lm.engine, "cache_manager", None), "expert_store", None)
                    result.update({
                        "prompt_tokens": len(input_ids), "completion_tokens": len(generated),
                        "eval_duration_ns": int(elapsed * 1e9), "speculative": use_spec,
                        "draft_depth": draft_depth if use_spec else depth, "full_depth": depth,
                        "lazy_experts": lm.lazy_experts, "rss_mb": round(_rss_mb(), 1),
                        "rss_delta_mb": round(_rss_mb() - rss0, 1), **telemetry,
                        **({"expert_store": store.stats()} if store is not None else {}),
                    })
                    chunks.put(None)

        threading.Thread(target=worker, daemon=True).start()
        stop = [s for s in (stop or []) if s]
        produced = ""
        finished = False            # True once the worker's ``None`` sentinel was consumed
        try:
            while True:
                item = chunks.get()
                if item is None:
                    finished = True
                    break
                text = item["text"]
                if stop:
                    produced += text
                    hits = [produced.find(s) for s in stop if s in produced]
                    if hits:
                        text = text[: max(0, len(text) - (len(produced) - min(hits)))]
                        if text:
                            yield {"text": text}
                        result["done_reason"] = "stop"
                        cancel.set()
                        break
                yield {"text": text}
        finally:
            cancel.set()                  # client went away or stop string hit
            while not finished:           # drain until the worker has released the model
                finished = chunks.get() is None
        result.setdefault("done_reason", "length" if result.get("completion_tokens", 0) >= max_new_tokens else "stop")
        yield {"done": True, **result}
