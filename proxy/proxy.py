"""proxy.py — Translating reverse proxy: Anthropic Messages API → OpenAI Chat Completions API.

Handles both streaming (SSE) and non-streaming translation between the two
API formats so that the ``claude`` CLI (which speaks Anthropic Messages API)
can talk to an SGLang backend (which speaks OpenAI Chat Completions API).
"""

import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI(title="LLM Metrics Proxy")

BACKEND_URL = "http://localhost:8000"
METRICS_PATH = Path("metrics.jsonl")
# Always use the actual SGLang model name
SGLANG_MODEL = os.environ.get("MODEL_PATH", "zai-org/GLM-4.5-Air-FP8")


# ---------------------------------------------------------------------------
# Request translation: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def _anthropic_to_openai(body: dict) -> dict:
    """Translate Anthropic Messages API request → OpenAI Chat Completions request."""
    messages = []

    if "system" in body:
        system_text = body["system"]
        if isinstance(system_text, list):
            system_text = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in system_text
            )
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        text_parts.append(str(block.get("content", "")))
                    else:
                        text_parts.append(json.dumps(block))
                else:
                    text_parts.append(str(block))
            content = "\n".join(text_parts)

        messages.append({"role": role, "content": content})

    openai_req = {
        "model": SGLANG_MODEL,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 0.7),
        "stream": body.get("stream", False),
    }

    # When streaming, ask SGLang to include usage info in the final chunk
    # so we can report real token counts in the Anthropic message_delta event.
    if openai_req["stream"]:
        openai_req["stream_options"] = {"include_usage": True}

    if "top_p" in body:
        openai_req["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        openai_req["stop"] = body["stop_sequences"]

    return openai_req


# ---------------------------------------------------------------------------
# Response translation: OpenAI → Anthropic  (non-streaming)
# ---------------------------------------------------------------------------

def _openai_to_anthropic(data: dict, model: str = "default") -> dict:
    """Translate OpenAI Chat Completions response → Anthropic Messages API response."""
    content = []
    stop_reason = "end_turn"

    if "choices" in data and data["choices"]:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        # Handle content being None, a string, or missing entirely
        text = msg.get("content") or ""
        if text:
            content.append({"type": "text", "text": str(text)})

        fr = choice.get("finish_reason") or ""
        if fr == "length":
            stop_reason = "max_tokens"
        elif fr == "stop":
            stop_reason = "end_turn"

    usage = data.get("usage") or {}
    return {
        "id": data.get("id", f"msg_{int(time.time())}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event_type: str, data: dict) -> str:
    """Format a single Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Endpoint handler — dispatcher
# ---------------------------------------------------------------------------

async def _handle_messages(request: Request) -> Response:
    body_bytes = await request.body()
    t0 = time.perf_counter()

    try:
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        return Response(content=b'{"error": "invalid JSON"}', status_code=400)

    model = body.get("model", "default")
    is_stream = body.get("stream", False)
    openai_req = _anthropic_to_openai(body)

    print(f"  [Proxy] Inbound {'Stream' if is_stream else 'Static'} request to {model}")

    if is_stream:
        return await _handle_stream(openai_req, model, t0, body_bytes, body)
    else:
        return await _handle_non_stream(openai_req, model, t0, body_bytes, body)


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------

async def _handle_non_stream(
    openai_req: dict, model: str, t0: float, req_body: bytes, orig_body: dict
) -> Response:
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{BACKEND_URL}/v1/chat/completions",
            json=openai_req,
        )

    latency = time.perf_counter() - t0

    try:
        openai_data = resp.json()
        print(f"  [Proxy] Raw SGLang Response: {resp.content.decode()[:500]}...")
        anthropic_resp = _openai_to_anthropic(openai_data, model)
        text = ""
        if anthropic_resp.get("content"):
            text = anthropic_resp["content"][0].get("text", "")
        print(f"  [Proxy] Response: {text[:60]}... ({latency:.2f}s)")
        resp_bytes = json.dumps(anthropic_resp).encode()
    except Exception as e:
        print(f"  [Proxy] Translation Error: {e}")
        resp_bytes = resp.content

    _log_metric("/v1/messages", latency, req_body, resp_bytes, openai_resp=resp.content)

    return Response(
        content=resp_bytes,
        status_code=resp.status_code,
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Streaming path  (OpenAI SSE → Anthropic SSE)
# ---------------------------------------------------------------------------

async def _handle_stream(
    openai_req: dict, model: str, t0: float, req_body: bytes, orig_body: dict
) -> StreamingResponse:
    async def generate():
        full_text = ""
        sent_header = False       # True once message_start has been emitted
        stream_finished = False   # True once finish_reason received
        msg_id = f"msg_{int(time.time())}"
        input_tokens = 0
        output_tokens = 0

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/v1/chat/completions",
                json=openai_req,
            ) as resp:
                if resp.status_code != 200:
                    print(f"  [Proxy] SGLang Error: {resp.status_code}")
                    yield _sse_event("error", {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Backend error {resp.status_code}"},
                    })
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    data_str = line.split("data:", 1)[1].strip()
                    if not data_str or data_str == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Grab the message id from the first chunk
                    if "id" in chunk:
                        msg_id = chunk["id"]

                    # Capture usage from the final usage-only chunk
                    # (sent when stream_options.include_usage is true)
                    if "usage" in chunk and chunk["usage"]:
                        input_tokens = chunk["usage"].get("prompt_tokens", input_tokens)
                        output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

                    # Skip chunks without choices (e.g. the usage-only tail chunk)
                    if not chunk.get("choices"):
                        continue

                    choice = chunk["choices"][0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    finish = choice.get("finish_reason")

                    # --- Emit Anthropic preamble on first real chunk ----------
                    if not sent_header:
                        yield _sse_event("message_start", {
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": SGLANG_MODEL,
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {
                                    "input_tokens": input_tokens,
                                    "output_tokens": 0,
                                },
                            },
                        })
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        })
                        yield _sse_event("ping", {"type": "ping"})
                        sent_header = True

                    # --- Content delta ----------------------------------------
                    if text:
                        print(text, end="", flush=True)
                        full_text += text
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        })

                    # --- Finish -----------------------------------------------
                    if finish:
                        stream_finished = True
                        stop = "max_tokens" if finish == "length" else "end_turn"
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": 0,
                        })
                        yield _sse_event("message_delta", {
                            "type": "message_delta",
                            "delta": {"stop_reason": stop, "stop_sequence": None},
                            "usage": {"output_tokens": output_tokens},
                        })
                        yield _sse_event("message_stop", {"type": "message_stop"})

        # If we opened the stream but never got a finish_reason, close cleanly
        # so the Claude CLI doesn't hang waiting for message_stop.
        if sent_header and not stream_finished:
            yield _sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            })
            yield _sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })
            yield _sse_event("message_stop", {"type": "message_stop"})

        print(f"\n  [Proxy] Stream complete. {len(full_text)} chars, {output_tokens} output tokens")
        latency = time.perf_counter() - t0
        _log_metric(
            "/v1/messages", latency, req_body,
            json.dumps({"full_text_length": len(full_text), "output_tokens": output_tokens}).encode(),
            stream=True, full_text=full_text,
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _forward_generic(request: Request) -> Response:
    body = await request.body()
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=60) as client:
        backend_resp = await client.request(
            method=request.method,
            url=f"{BACKEND_URL}{request.url.path}",
            headers=dict(request.headers),
            content=body,
            timeout=60,
        )

    latency = time.perf_counter() - t0
    _log_metric(request.url.path, latency, body, backend_resp.content)

    return Response(
        content=backend_resp.content,
        status_code=backend_resp.status_code,
        headers=dict(backend_resp.headers),
    )


def _log_metric(
    path: str, latency: float, req_body: bytes, resp_body: bytes,
    openai_resp: bytes = b"", stream: bool = False, full_text: str = "",
) -> None:
    record: dict = {
        "ts": time.time(),
        "path": path,
        "latency_s": round(latency, 4),
        "req_bytes": len(req_body),
        "resp_bytes": len(resp_body),
        "stream": stream,
    }

    raw = openai_resp or resp_body
    try:
        data = json.loads(raw)
        if "usage" in data:
            usage = data["usage"]
            record["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        if "model" in data: record["model"] = data["model"]
        if "id" in data: record["request_id"] = data["id"]
    except: pass

    if stream and full_text: record["completion_text_length"] = len(full_text)

    with METRICS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/v1/messages")
@app.post("/v1/v1/messages")
async def messages_endpoint(request: Request) -> Response:
    return await _handle_messages(request)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request) -> Response:
    return await _forward_generic(request)
