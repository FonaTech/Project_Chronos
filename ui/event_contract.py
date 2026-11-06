"""Keep browser event IDs tied to the Gradio app that produced the page.

Gradio's numeric fn_index changes when events are inserted. A tab surviving a
server restart can otherwise send an old timer's empty payload to a new button
handler. Reject that request before it enters the queue; new pages reload when
the server instance changes. API clients without the browser header still work.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid


HEADER = "x-chronos-ui-contract"
CONTRACT_PATH = "/chronos-ui/contract"


class UIEventContract:
    def __init__(self):
        self.token = uuid.uuid4().hex
        self.blocks = None

    def head(self):
        script = Path(__file__).with_name("event_contract.js").read_text(encoding="utf-8")
        return "<script>" + script.replace("__CHRONOS_CONTRACT_TOKEN__", json.dumps(self.token)) + "</script>"


class UIEventContractMiddleware:
    def __init__(self, app, contract):
        self.app = app
        self.contract = contract

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "").rstrip("/")
        method = scope["method"]

        async def respond(status, payload):
            body = json.dumps(payload).encode()
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"), (b"cache-control", b"no-store"),
                (b"content-length", str(len(body)).encode()),
            ]})
            await send({"type": "http.response.body", "body": body})

        if method == "GET" and path.endswith(CONTRACT_PATH):
            return await respond(200, {"token": self.contract.token})

        queued_event = method == "POST" and path.endswith("/queue/join")
        direct_event = method == "POST" and any(
            segment in path for segment in ("/run/", "/api/", "/call/")
        )
        client_token = dict(scope.get("headers", [])).get(HEADER.encode(), b"").decode()
        reason = None
        if (queued_event or direct_event) and client_token and client_token != self.contract.token:
            reason = "server_restarted"

        if queued_event and reason is None:
            chunks = []
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = None  # Let FastAPI produce its normal validation error.
            if isinstance(payload, dict) and not payload.get("simple_format"):
                blocks = self.contract.blocks
                functions = blocks.fns
                # Respect per-session functions if Gradio rendered them dynamically.
                session = blocks.state_holder.session_data.get(payload.get("session_hash"))
                if session is not None:
                    functions = session.blocks_config.fns
                index = payload.get("fn_index")
                fn = functions.get(index) if isinstance(index, int) else None
                if fn is None:
                    reason = "unknown_event"
                elif isinstance(payload.get("data"), list) and len(payload["data"]) != len(fn.inputs):
                    reason = "input_count_mismatch"

            delivered = False
            original_receive = receive

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await original_receive()

            receive = replay

        if reason:
            outdated = client_token != self.contract.token
            return await respond(409 if outdated else 422, {
                "code": "chronos_ui_outdated" if outdated else "chronos_invalid_event",
                "reason": reason,
                "error": "页面与当前服务的事件接口不一致，请刷新页面。 / Reload the page to synchronize UI events.",
                "detail": "No event was executed. Refresh the Chronos page and retry.",
            })

        async def no_cached_page(message):
            if method == "GET" and message["type"] == "http.response.start":
                headers = message.get("headers", [])
                content_type = dict(headers).get(b"content-type", b"")
                if b"text/html" in content_type or path.endswith("/config"):
                    message = dict(message, headers=[(k, v) for k, v in headers if k.lower() != b"cache-control"]
                                   + [(b"cache-control", b"no-store")])
            await send(message)

        return await self.app(scope, receive, no_cached_page)


def event_contract_app_kwargs(blocks):
    """Pass through Blocks.launch(app_kwargs=...) so Gradio keeps its lifespan."""
    contract = getattr(blocks, "chronos_event_contract", None)
    if contract is None:
        return {}
    from starlette.middleware import Middleware
    return {"middleware": [Middleware(UIEventContractMiddleware, contract=contract)]}
