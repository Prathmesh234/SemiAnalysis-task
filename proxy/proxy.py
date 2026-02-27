"""proxy.py — FastAPI reverse proxy on :8001 that forwards to :8000 and logs comprehensive metrics to JSONL."""

import json
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response

app = FastAPI(title="LLM Metrics Proxy")

BACKEND_URL = "http://localhost:8000"
METRICS_PATH = Path("metrics.jsonl")


async def _forward(request: Request) -> Response:
    """Forward an incoming request to the backend and log latency + token counts."""
    body = await request.body()
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        backend_resp = await client.request(
            method=request.method,
            url=f"{BACKEND_URL}{request.url.path}",
            headers=dict(request.headers),
            content=body,
            timeout=300,
        )

    latency = time.perf_counter() - t0
    _log_metric(request.url.path, latency, body, backend_resp.content)

    return Response(
        content=backend_resp.content,
        status_code=backend_resp.status_code,
        headers=dict(backend_resp.headers),
    )


def _log_metric(path: str, latency: float, req_body: bytes, resp_body: bytes) -> None:
    """Append a single JSON line with comprehensive request metadata and timing."""
    record: dict = {
        "ts": time.time(),
        "path": path,
        "latency_s": round(latency, 4),
        "req_bytes": len(req_body),
        "resp_bytes": len(resp_body),
    }

    # Parse token usage from OpenAI-compatible response
    try:
        data = json.loads(resp_body)

        if "usage" in data:
            usage = data["usage"]
            record["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            # SGLang may include prompt_tokens_details with cached_tokens
            if isinstance(usage.get("prompt_tokens_details"), dict):
                record["usage"]["prompt_tokens_details"] = usage["prompt_tokens_details"]

            # SGLang may include completion_tokens_details
            if isinstance(usage.get("completion_tokens_details"), dict):
                record["usage"]["completion_tokens_details"] = usage["completion_tokens_details"]

        # Capture model name
        if "model" in data:
            record["model"] = data["model"]

        # Capture request ID
        if "id" in data:
            record["request_id"] = data["id"]

        # Capture finish reason
        if "choices" in data and data["choices"]:
            finish_reason = data["choices"][0].get("finish_reason")
            if finish_reason:
                record["finish_reason"] = finish_reason

    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass

    # Parse request body for input metadata
    try:
        req_data = json.loads(req_body)
        if "messages" in req_data:
            record["num_messages"] = len(req_data["messages"])
        if "max_tokens" in req_data:
            record["requested_max_tokens"] = req_data["max_tokens"]
        if "temperature" in req_data:
            record["temperature"] = req_data["temperature"]
        if "stream" in req_data:
            record["stream"] = req_data["stream"]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    with METRICS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request) -> Response:
    """Catch-all route that proxies every request to the backend."""
    return await _forward(request)
