"""proxy.py — FastAPI reverse proxy on :8001 that forwards to :8000 and logs metrics to JSONL."""

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
    """Append a single JSON line with request metadata and timing."""
    record = {
        "ts": time.time(),
        "path": path,
        "latency_s": round(latency, 4),
        "req_bytes": len(req_body),
        "resp_bytes": len(resp_body),
    }
    # Best-effort parse token usage from OpenAI-compatible response
    try:
        data = json.loads(resp_body)
        if "usage" in data:
            record["usage"] = data["usage"]
    except (json.JSONDecodeError, KeyError):
        pass

    with METRICS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request) -> Response:
    """Catch-all route that proxies every request to the backend."""
    return await _forward(request)
