"""collector.py — Read proxy JSONL log and compute aggregated stats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AggregatedMetrics:
    """Summary statistics over all logged requests."""

    total_requests: int = 0
    total_latency_s: float = 0.0
    avg_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


def collect(metrics_path: Path = Path("metrics.jsonl")) -> AggregatedMetrics:
    """Parse the JSONL log written by the proxy and return aggregated stats."""
    records: list[dict] = []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return AggregatedMetrics()

    latencies = sorted(r["latency_s"] for r in records)
    n = len(latencies)

    prompt_tokens = sum(
        r.get("usage", {}).get("prompt_tokens", 0) for r in records
    )
    completion_tokens = sum(
        r.get("usage", {}).get("completion_tokens", 0) for r in records
    )

    return AggregatedMetrics(
        total_requests=n,
        total_latency_s=sum(latencies),
        avg_latency_s=sum(latencies) / n,
        p50_latency_s=latencies[n // 2],
        p99_latency_s=latencies[int(n * 0.99)],
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
    )
