"""Ollama- and OpenAI-compatible HTTP server for Chronos models.

The stock ``ollama`` CLI and any Ollama/OpenAI client can talk to the native
Chronos runtime by pointing at this server::

    chronos serve --models-dir runs/my_run --port 11435
    OLLAMA_HOST=127.0.0.1:11435 ollama run chronos_sft "Hello world"

The kernel is never converted to another architecture: recurrent depth,
SSD-lazy experts with lookahead prefetch, compressed KV and self-speculative
decoding all stay active behind the protocol.

Security: binds 127.0.0.1 and has no authentication by default.  Binding a
non-loopback address requires ``--allow-remote`` *and* ``--api-key``.
``--monitor-runs-dir`` mounts the read-only training monitor at ``/monitor/``;
it reuses the same key through its own login page (session cookie).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone

try:  # module level: FastAPI resolves ``Request`` annotations by name
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
except ImportError:  # pragma: no cover - serving extra not installed
    FastAPI = HTTPException = Request = JSONResponse = PlainTextResponse = StreamingResponse = None

OLLAMA_COMPAT_VERSION = "0.40.14"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(path: str) -> str:
    st = os.stat(path)
    return hashlib.sha256(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()


def _options(body: dict) -> dict:
    opts = dict(body.get("options") or {})
    return {
        "max_new_tokens": int(opts.get("num_predict", body.get("max_tokens", 256)) or 256),
        "temperature": float(opts.get("temperature", body.get("temperature", 0.0)) or 0.0),
        "top_k": int(opts.get("top_k", body.get("top_k", 0)) or 0),
        "top_p": float(opts.get("top_p", body.get("top_p", 1.0)) or 1.0),
        "seed": opts.get("seed", body.get("seed")),
        "stop": opts.get("stop", body.get("stop")) or [],
        "speculative": bool(opts.get("chronos_speculative", True)),
        "draft_depth": int(opts.get("chronos_draft_depth", 1)),
        "gamma": int(opts.get("chronos_gamma", 4)),
        "miss_policy": str(opts.get("chronos_miss_policy", "on_demand")),
    }


def _details(lm_or_cfg) -> dict:
    cfg = getattr(lm_or_cfg, "config", lm_or_cfg)
    return {"parent_model": "", "format": "chronos", "family": "chronos", "families": ["chronos"],
            "parameter_size": f"{getattr(cfg, '_total_params_m', 0) or 0:.1f}M" if hasattr(cfg, "_total_params_m") else "",
            "quantization_level": "BF16"}


def _final_ollama(stats: dict, model: str, total_t0: float, load_s: float) -> dict:
    return {
        "model": model, "created_at": _now(), "done": True,
        "done_reason": stats.get("done_reason", "stop"),
        "total_duration": int((time.perf_counter() - total_t0) * 1e9),
        "load_duration": int(load_s * 1e9),
        "prompt_eval_count": stats.get("prompt_tokens", 0), "prompt_eval_duration": 0,
        "eval_count": stats.get("completion_tokens", 0),
        "eval_duration": stats.get("eval_duration_ns", 0),
        "chronos": {k: v for k, v in stats.items() if k not in ("done", "done_reason")},
    }


def create_app(runtime, *, api_key: str | None = None, monitor_runs_dir: str | None = None):
    if FastAPI is None:
        raise ImportError("chronos serve needs fastapi and uvicorn (pip install 'Project_Chronos[serve]')")
    app = FastAPI(title="Chronos Ollama-compatible server", version=OLLAMA_COMPAT_VERSION)
    expected = f"Bearer {api_key}".encode("utf-8") if api_key else b""

    @app.middleware("http")
    async def auth(request: Request, call_next):
        path = request.url.path
        # /monitor has its own login (cookie or Bearer) with the same key.
        exempt = path in ("/", "/api/version") or (monitor_runs_dir and (path == "/monitor" or path.startswith("/monitor/")))
        if api_key and not exempt:
            given = request.headers.get("authorization", "").encode("utf-8")
            if not hmac.compare_digest(given, expected):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    if monitor_runs_dir:
        from fastapi.responses import RedirectResponse
        from chronos.monitor.server import create_monitor_app

        @app.get("/monitor", include_in_schema=False)
        def monitor_redirect():
            return RedirectResponse("/monitor/")

        app.mount("/monitor", create_monitor_app(monitor_runs_dir, api_key=api_key))

    def _resolve(name: str):
        try:
            t0 = time.perf_counter()
            lm = runtime.get(name)
            return lm, time.perf_counter() - t0
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"model '{name}' not found, try pulling it first") from exc

    def _model_entry(name, path):
        st = os.stat(path)
        return {"name": f"{name}:latest", "model": f"{name}:latest",
                "modified_at": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                "size": st.st_size, "digest": _digest(path),
                "details": {"parent_model": "", "format": "chronos", "family": "chronos",
                            "families": ["chronos"], "parameter_size": "", "quantization_level": "BF16"}}

    @app.get("/")
    def root():
        return PlainTextResponse("Ollama is running")

    @app.head("/")
    def root_head():
        return PlainTextResponse("")

    @app.get("/api/version")
    def version():
        return {"version": OLLAMA_COMPAT_VERSION}

    @app.get("/api/tags")
    def tags():
        return {"models": [_model_entry(n, p) for n, p in sorted(runtime.available().items())]}

    @app.post("/api/show")
    async def show(request: Request):
        body = await request.json()
        name = (body.get("model") or body.get("name") or "").split(":")[0]
        path = runtime.available().get(name)
        if path is None:
            raise HTTPException(status_code=404, detail=f"model '{name}' not found")
        meta = json.load(open(os.path.splitext(path)[0] + ".config.json", encoding="utf-8"))
        cfg = meta.get("config", {})
        info = {"general.architecture": "chronos", "chronos.recurrent_depth": cfg.get("recurrent_depth"),
                "chronos.recurrent_attention_mode": cfg.get("recurrent_attention_mode"),
                "chronos.num_experts": cfg.get("num_experts"), "chronos.experts_per_token": cfg.get("num_experts_per_tok"),
                "chronos.kv_latent_dim": cfg.get("kv_latent_dim"), "chronos.context_length": cfg.get("max_position_embeddings"),
                "chronos.hidden_size": cfg.get("hidden_size"), "chronos.vocab_size": cfg.get("vocab_size")}
        return {"modelfile": f"FROM {path}\n", "parameters": "temperature 0\n", "template": "{{ .Prompt }}",
                "details": _model_entry(name, path)["details"], "model_info": info,
                "capabilities": ["completion"], "modified_at": _model_entry(name, path)["modified_at"]}

    @app.get("/api/ps")
    def ps():
        out = []
        for name, lm in list(runtime.loaded.items()):
            store = getattr(getattr(lm.engine, "cache_manager", None), "expert_store", None)
            st = store.stats() if store is not None else {}
            entry = _model_entry(name, lm.path)
            entry.update(size_vram=int(st.get("vram_bytes", 0) or 0),
                         expires_at=datetime.fromtimestamp(runtime.last_used.get(name, time.time()) + runtime.keep_alive_s,
                                                           timezone.utc).isoformat().replace("+00:00", "Z"),
                         chronos={"device": lm.device, "lazy_experts": lm.lazy_experts,
                                  **{k: lm.meta[k] for k in ("paging", "paging_resolved", "num_pages")
                                     if k in (lm.meta or {})},
                                  "expert_store": st})
            out.append(entry)
        return {"models": out}

    @app.post("/api/pull")
    async def pull(request: Request):
        body = await request.json()
        name = (body.get("model") or body.get("name") or "").split(":")[0]
        if name not in runtime.available():
            return JSONResponse({"error": f"model '{name}' is not in --models-dir; Chronos serves local checkpoints"},
                                status_code=404)
        return {"status": "success"}

    def _ndjson(gen):
        for item in gen:
            yield json.dumps(item, ensure_ascii=False) + "\n"

    def _run(body, input_ids, lm, load_s, shape):
        opts = _options(body)
        t0 = time.perf_counter()
        name = lm.name
        def events():
            for item in runtime.generate_stream(name, input_ids, **opts):
                if item.get("done"):
                    final = _final_ollama(item, body.get("model", name), t0, load_s)
                    if "error" in item:
                        final["error"] = item["error"]
                    yield shape(None, final)
                else:
                    yield shape(item["text"], None)
        return events

    def _respond(body, events, collect):
        if body.get("stream", True):
            return StreamingResponse(_ndjson(events()), media_type="application/x-ndjson")
        return JSONResponse(collect(list(events())))

    def _unload_if_requested(body, name) -> bool:
        keep = body.get("keep_alive")
        if keep in (0, "0", "0s", "0m") and not body.get("prompt") and not body.get("messages"):
            runtime.unload(name)
            return True
        return False

    @app.post("/api/generate")
    async def generate(request: Request):
        body = await request.json()
        name = (body.get("model") or "").split(":")[0]
        if _unload_if_requested(body, name):
            return {"model": body.get("model"), "created_at": _now(), "response": "", "done": True,
                    "done_reason": "unload"}
        lm, load_s = _resolve(name)
        if not body.get("prompt"):     # Ollama "load model" request
            return {"model": body.get("model"), "created_at": _now(), "response": "", "done": True,
                    "done_reason": "load"}
        ids = runtime.encode_prompt(lm, body["prompt"], body.get("system"), bool(body.get("raw")))

        def shape(text, final):
            if final is not None:
                return {**final, "response": ""}
            return {"model": body.get("model"), "created_at": _now(), "response": text, "done": False}

        def collect(items):
            return {**items[-1], "response": "".join(i.get("response", "") for i in items[:-1])}
        return _respond(body, _run(body, ids, lm, load_s, shape), collect)

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        name = (body.get("model") or "").split(":")[0]
        messages = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in body.get("messages") or []]
        if _unload_if_requested(body, name):
            return {"model": body.get("model"), "created_at": _now(),
                    "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "unload"}
        lm, load_s = _resolve(name)
        if not messages:
            return {"model": body.get("model"), "created_at": _now(),
                    "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "load"}
        ids = runtime.encode_chat(lm, messages)

        def shape(text, final):
            if final is not None:
                return {**final, "message": {"role": "assistant", "content": ""}}
            return {"model": body.get("model"), "created_at": _now(),
                    "message": {"role": "assistant", "content": text}, "done": False}

        def collect(items):
            text = "".join(i["message"]["content"] for i in items[:-1])
            return {**items[-1], "message": {"role": "assistant", "content": text}}
        return _respond(body, _run(body, ids, lm, load_s, shape), collect)

    @app.get("/v1/models")
    def openai_models():
        return {"object": "list", "data": [{"id": n, "object": "model", "created": int(os.stat(p).st_mtime),
                                            "owned_by": "chronos"} for n, p in sorted(runtime.available().items())]}

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        import uuid
        body = await request.json()
        name = (body.get("model") or "").split(":")[0]
        lm, load_s = _resolve(name)
        messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in body.get("messages") or []]
        ids = runtime.encode_chat(lm, messages)
        opts = _options({**body, "options": {"num_predict": body.get("max_tokens") or body.get("max_completion_tokens") or 256,
                                             "temperature": body.get("temperature", 0.0), "top_p": body.get("top_p", 1.0),
                                             "seed": body.get("seed"), "stop": body.get("stop"),
                                             **(body.get("chronos") or {})}})
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        if body.get("stream"):
            def sse():
                for item in runtime.generate_stream(name, ids, **opts):
                    if item.get("done"):
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                                 "choices": [{"index": 0, "delta": {}, "finish_reason": item.get("done_reason", "stop")}],
                                 "chronos": item}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
                    else:
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                                 "choices": [{"index": 0, "delta": {"content": item["text"]}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        items = list(runtime.generate_stream(name, ids, **opts))
        final = items[-1]
        return {"id": cid, "object": "chat.completion", "created": created, "model": name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(i["text"] for i in items[:-1])},
                             "finish_reason": final.get("done_reason", "stop")}],
                "usage": {"prompt_tokens": final.get("prompt_tokens", 0), "completion_tokens": final.get("completion_tokens", 0),
                          "total_tokens": final.get("prompt_tokens", 0) + final.get("completion_tokens", 0)},
                "chronos": final}

    return app


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="chronos serve", description=__doc__.split("\n\n")[0])
    parser.add_argument("--models-dir", required=True, help="directory searched for <name>.pth + <name>.config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435, help="11434 would collide with a running Ollama")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="compute dtype: auto|float32|bfloat16|float16 "
                        "(auto: CUDA bf16/fp16, MPS fp32, CPU fp32)")
    parser.add_argument("--no-lazy-experts", action="store_true", help="keep every expert resident (no SSD offload)")
    parser.add_argument("--ram-expert-fraction", type=float, default=0.5)
    parser.add_argument("--paging", choices=["auto", "cluster", "expert"], default="auto",
                        help="expert page layout; auto pages by training cluster when the checkpoint "
                             "was trained with cluster locality, per expert otherwise")
    # Inference-only options (default: whatever the checkpoint config says).
    parser.add_argument("--kv-cache-dtype", choices=["auto", "int8"], default=None,
                        help="int8: per-token int8 KV cache with FP32 group scales")
    parser.add_argument("--mla-decode", choices=["auto", "absorbed", "expanded"], default=None)
    parser.add_argument("--mla-cache-rope", action="store_true", default=None,
                        help="also cache post-RoPE rope keys (memory for throughput)")
    parser.add_argument("--router-scope", choices=["per_iteration", "shared"], default=None,
                        help="only change this for checkpoints trained with the same scope")
    parser.add_argument("--max-resident-experts", type=int, default=None,
                        help="hard cap on fast-tier expert pages")
    parser.add_argument("--ram-budget-mb", type=float, default=None, help="hard cap on RAM expert pages (MiB)")
    parser.add_argument("--api-key", default=os.environ.get("CHRONOS_API_KEY"))
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--monitor-runs-dir", default=None,
                        help="also serve the read-only training monitor for this runs directory at /monitor/")
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        if not (args.allow_remote and args.api_key):
            parser.error("binding a non-loopback host requires --allow-remote and --api-key")
    import uvicorn
    from chronos.serving.runtime import ChronosRuntime
    overrides = {"kv_cache_dtype": args.kv_cache_dtype, "mla_decode": args.mla_decode,
                 "mla_cache_rope": args.mla_cache_rope, "router_scope": args.router_scope,
                 "max_resident_experts": args.max_resident_experts, "ram_budget_mb": args.ram_budget_mb}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    runtime = ChronosRuntime(args.models_dir, device=args.device, lazy_experts=not args.no_lazy_experts,
                             ram_expert_fraction=args.ram_expert_fraction, dtype=args.dtype,
                             inference_overrides=overrides, paging=args.paging)
    print(f"[chronos serve] http://{args.host}:{args.port}  models={sorted(runtime.available())}  "
          f"auth={'on' if args.api_key else 'off (loopback only)'}  overrides={overrides or 'none'}", flush=True)
    if args.monitor_runs_dir:
        print(f"[chronos serve] monitor: http://{args.host}:{args.port}/monitor/  "
              f"runs={os.path.abspath(args.monitor_runs_dir)}", flush=True)
    app = create_app(runtime, api_key=args.api_key, monitor_runs_dir=args.monitor_runs_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
