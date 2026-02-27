"""collector.py — Comprehensive metrics collection from SGLang Prometheus, nvidia-smi, and proxy JSONL.

Scrapes every metric domain:
  1) Proxy JSONL            — per-request latency, token counts, bytes
  2) SGLang /metrics        — Prometheus counters/gauges/histograms from all workers
  3) nvidia-smi             — GPU util%, memory, temperature, power, SM clock
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ProxyMetrics:
    """Per-request stats parsed from the proxy JSONL log."""

    total_requests: int = 0
    total_latency_s: float = 0.0
    avg_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p90_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    min_latency_s: float = 0.0
    max_latency_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    avg_tokens_per_request: float = 0.0
    total_req_bytes: int = 0
    total_resp_bytes: int = 0


@dataclass
class LatencyMetrics:
    """Latency breakdown from SGLang Prometheus histograms."""

    ttft_avg_s: float = 0.0          # Time to First Token
    ttft_p50_s: float = 0.0
    ttft_p95_s: float = 0.0
    ttft_p99_s: float = 0.0
    itl_avg_s: float = 0.0           # Inter-Token Latency
    itl_p50_s: float = 0.0
    itl_p95_s: float = 0.0
    itl_p99_s: float = 0.0
    e2e_avg_s: float = 0.0           # End-to-End request latency
    e2e_p50_s: float = 0.0
    e2e_p95_s: float = 0.0
    e2e_p99_s: float = 0.0
    queue_time_avg_s: float = 0.0    # Queue wait time
    queue_time_p99_s: float = 0.0


@dataclass
class ThroughputMetrics:
    """Token throughput from SGLang Prometheus."""

    gen_throughput_tok_s: float = 0.0         # Current generation tok/s (gauge)
    total_prompt_tokens: int = 0              # Counter: total prefill tokens
    total_generation_tokens: int = 0          # Counter: total decode tokens
    total_requests: int = 0                   # Counter: total requests processed
    total_aborted_requests: int = 0           # Counter: aborted requests
    # Derived
    effective_throughput_tok_s: float = 0.0   # Computed from counters if timestamps available


@dataclass
class GPUMetrics:
    """Per-GPU hardware metrics from nvidia-smi."""

    gpu_id: int = 0
    name: str = ""
    utilization_pct: float = 0.0          # SM utilization %
    memory_used_mib: float = 0.0          # MiB used
    memory_total_mib: float = 0.0         # MiB total
    memory_utilization_pct: float = 0.0   # Memory bandwidth utilization %
    temperature_c: float = 0.0            # GPU temp °C
    power_draw_w: float = 0.0             # Current power draw W
    power_limit_w: float = 0.0            # Power limit W
    sm_clock_mhz: int = 0                 # Current SM clock
    mem_clock_mhz: int = 0                # Current memory clock
    pcie_tx_mb_s: float = 0.0             # PCIe TX throughput MB/s
    pcie_rx_mb_s: float = 0.0             # PCIe RX throughput MB/s
    ecc_errors_total: int = 0             # ECC errors (volatile)


@dataclass
class CacheMetrics:
    """KV cache and prefix cache metrics from SGLang Prometheus."""

    cache_hit_rate: float = 0.0           # Prefix cache hit rate (0-1)
    cached_tokens_total: int = 0          # Tokens served from cache
    evicted_tokens_total: int = 0         # Tokens evicted from GPU
    load_back_tokens_total: int = 0       # Tokens loaded back from CPU
    eviction_duration_avg_s: float = 0.0
    load_back_duration_avg_s: float = 0.0
    token_usage_ratio: float = 0.0        # KV cache utilization (0-1)
    num_used_tokens: int = 0              # Absolute token count in KV cache
    max_total_tokens: int = 0             # Maximum KV cache capacity


@dataclass
class SchedulerMetrics:
    """Scheduler state from SGLang Prometheus gauges."""

    num_running_reqs: int = 0
    num_queue_reqs: int = 0
    num_retracted_total: int = 0          # Preempted requests
    num_paused_reqs: int = 0
    gpu_execution_seconds_total: float = 0.0  # Total GPU busy time


@dataclass
class SpeculativeMetrics:
    """Speculative decoding (MTP/EAGLE) metrics."""

    accept_length_avg: float = 0.0        # Average tokens accepted per step
    accept_rate: float = 0.0              # Ratio of accepted/total draft tokens
    enabled: bool = False


@dataclass
class DisaggMetrics:
    """Disaggregated (2P2D) KV transfer metrics."""

    kv_transfer_speed_gb_s: float = 0.0   # KV cache transfer speed
    kv_transfer_latency_ms: float = 0.0   # Transfer latency
    num_prefill_prealloc_queue: int = 0    # Requests in pre-alloc queue


@dataclass
class AggregatedMetrics:
    """Complete benchmark metrics across all domains."""

    timestamp: float = 0.0
    collection_duration_s: float = 0.0

    # Domain-specific sub-metrics
    proxy: ProxyMetrics = field(default_factory=ProxyMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    gpu: list[GPUMetrics] = field(default_factory=list)
    cache: CacheMetrics = field(default_factory=CacheMetrics)
    scheduler: SchedulerMetrics = field(default_factory=SchedulerMetrics)
    speculative: SpeculativeMetrics = field(default_factory=SpeculativeMetrics)
    disagg: DisaggMetrics = field(default_factory=DisaggMetrics)

    # Raw Prometheus samples (for custom analysis)
    raw_prometheus: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Legacy compat — flat accessors used by report.py
    @property
    def total_requests(self) -> int:
        return self.proxy.total_requests

    @property
    def total_latency_s(self) -> float:
        return self.proxy.total_latency_s

    @property
    def avg_latency_s(self) -> float:
        return self.proxy.avg_latency_s

    @property
    def p50_latency_s(self) -> float:
        return self.proxy.p50_latency_s

    @property
    def p99_latency_s(self) -> float:
        return self.proxy.p99_latency_s

    @property
    def total_prompt_tokens(self) -> int:
        return self.proxy.total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self.proxy.total_completion_tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) Proxy JSONL collection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _collect_proxy(metrics_path: Path) -> ProxyMetrics:
    """Parse the JSONL log written by the proxy and return detailed stats."""
    if not metrics_path.exists():
        return ProxyMetrics()

    records: list[dict] = []
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return ProxyMetrics()

    latencies = sorted(r.get("latency_s", 0) for r in records)
    n = len(latencies)

    prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in records)
    completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in records)
    cached_tokens = sum(
        r.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        for r in records
        if isinstance(r.get("usage", {}).get("prompt_tokens_details"), dict)
    )
    total_req_bytes = sum(r.get("req_bytes", 0) for r in records)
    total_resp_bytes = sum(r.get("resp_bytes", 0) for r in records)

    def _percentile(sorted_vals: list[float], pct: float) -> float:
        idx = min(int(len(sorted_vals) * pct), len(sorted_vals) - 1)
        return sorted_vals[idx]

    return ProxyMetrics(
        total_requests=n,
        total_latency_s=sum(latencies),
        avg_latency_s=sum(latencies) / n,
        p50_latency_s=_percentile(latencies, 0.50),
        p90_latency_s=_percentile(latencies, 0.90),
        p95_latency_s=_percentile(latencies, 0.95),
        p99_latency_s=_percentile(latencies, 0.99),
        min_latency_s=latencies[0],
        max_latency_s=latencies[-1],
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_cached_tokens=cached_tokens,
        avg_tokens_per_request=(prompt_tokens + completion_tokens) / n,
        total_req_bytes=total_req_bytes,
        total_resp_bytes=total_resp_bytes,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2) SGLang Prometheus scraping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _scrape_prometheus(url: str, timeout: float = 5.0) -> dict[str, list[dict]]:
    """Scrape a Prometheus /metrics endpoint, parse text format into a dict.

    Returns: {metric_name: [{labels: {...}, value: float}, ...]}
    """
    try:
        with urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except (URLError, OSError, TimeoutError):
        return {}

    metrics: dict[str, list[dict]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Parse: metric_name{label1="val1",label2="val2"} value
        # or:    metric_name value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?(.*?)\}?\s+([\d.eE+\-]+|NaN|Inf|\+Inf|-Inf)$', line)
        if not match:
            # Try without labels
            parts = line.split()
            if len(parts) == 2:
                name, val_str = parts
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                metrics.setdefault(name, []).append({"labels": {}, "value": val})
            continue

        name = match.group(1)
        labels_str = match.group(2)
        val_str = match.group(3)

        try:
            val = float(val_str)
        except ValueError:
            continue

        labels = {}
        if labels_str:
            for lm in re.finditer(r'(\w+)="([^"]*)"', labels_str):
                labels[lm.group(1)] = lm.group(2)

        metrics.setdefault(name, []).append({"labels": labels, "value": val})

    return metrics


def _get_scalar(prom: dict, name: str, default: float = 0.0) -> float:
    """Get a single scalar value from parsed Prometheus metrics."""
    # Try both sglang: and sglang_ prefix formats
    for prefix_name in [name, name.replace(":", "_")]:
        entries = prom.get(prefix_name, [])
        if entries:
            return entries[0]["value"]
    return default


def _get_histogram_percentile(prom: dict, name: str, percentile: float) -> float:
    """Estimate a percentile from Prometheus histogram bucket data."""
    bucket_name = f"{name}_bucket"
    for bn in [bucket_name, bucket_name.replace(":", "_")]:
        entries = prom.get(bn, [])
        if not entries:
            continue

        # Sort by le (upper bound)
        buckets = []
        for e in entries:
            le = e["labels"].get("le", "+Inf")
            if le == "+Inf":
                le_val = float("inf")
            else:
                try:
                    le_val = float(le)
                except ValueError:
                    continue
            buckets.append((le_val, e["value"]))

        buckets.sort(key=lambda x: x[0])
        if not buckets:
            return 0.0

        total = buckets[-1][1]  # +Inf bucket = total count
        if total == 0:
            return 0.0

        target = percentile * total
        prev_bound = 0.0
        prev_count = 0.0
        for bound, count in buckets:
            if count >= target and bound != float("inf"):
                # Linear interpolation within bucket
                bucket_fraction = (target - prev_count) / max(count - prev_count, 1)
                return prev_bound + bucket_fraction * (bound - prev_bound)
            prev_bound = bound
            prev_count = count

        # Fallback to last finite bucket
        for bound, _ in reversed(buckets):
            if bound != float("inf"):
                return bound
        return 0.0

    return 0.0


def _get_histogram_avg(prom: dict, name: str) -> float:
    """Compute average from histogram _sum / _count."""
    for prefix in [name, name.replace(":", "_")]:
        sum_entries = prom.get(f"{prefix}_sum", [])
        count_entries = prom.get(f"{prefix}_count", [])
        if sum_entries and count_entries:
            total_sum = sum(e["value"] for e in sum_entries)
            total_count = sum(e["value"] for e in count_entries)
            if total_count > 0:
                return total_sum / total_count
    return 0.0


def _collect_sglang_prometheus(
    worker_urls: list[str],
) -> tuple[LatencyMetrics, ThroughputMetrics, CacheMetrics, SchedulerMetrics,
           SpeculativeMetrics, DisaggMetrics, dict]:
    """Scrape all SGLang workers and aggregate metrics."""

    # Merge all worker metrics
    merged: dict[str, list[dict]] = {}
    for url in worker_urls:
        prom = _scrape_prometheus(f"{url}/metrics")
        for k, v in prom.items():
            merged.setdefault(k, []).extend(v)

    if not merged:
        return (LatencyMetrics(), ThroughputMetrics(), CacheMetrics(),
                SchedulerMetrics(), SpeculativeMetrics(), DisaggMetrics(), {})

    # ── Latency ─────────────────────────────────────────────
    latency = LatencyMetrics(
        ttft_avg_s=_get_histogram_avg(merged, "sglang:time_to_first_token_seconds"),
        ttft_p50_s=_get_histogram_percentile(merged, "sglang:time_to_first_token_seconds", 0.50),
        ttft_p95_s=_get_histogram_percentile(merged, "sglang:time_to_first_token_seconds", 0.95),
        ttft_p99_s=_get_histogram_percentile(merged, "sglang:time_to_first_token_seconds", 0.99),
        itl_avg_s=_get_histogram_avg(merged, "sglang:inter_token_latency_seconds"),
        itl_p50_s=_get_histogram_percentile(merged, "sglang:inter_token_latency_seconds", 0.50),
        itl_p95_s=_get_histogram_percentile(merged, "sglang:inter_token_latency_seconds", 0.95),
        itl_p99_s=_get_histogram_percentile(merged, "sglang:inter_token_latency_seconds", 0.99),
        e2e_avg_s=_get_histogram_avg(merged, "sglang:e2e_request_latency_seconds"),
        e2e_p50_s=_get_histogram_percentile(merged, "sglang:e2e_request_latency_seconds", 0.50),
        e2e_p95_s=_get_histogram_percentile(merged, "sglang:e2e_request_latency_seconds", 0.95),
        e2e_p99_s=_get_histogram_percentile(merged, "sglang:e2e_request_latency_seconds", 0.99),
        queue_time_avg_s=_get_histogram_avg(merged, "sglang:queue_time_seconds"),
        queue_time_p99_s=_get_histogram_percentile(merged, "sglang:queue_time_seconds", 0.99),
    )

    # ── Throughput ──────────────────────────────────────────
    throughput = ThroughputMetrics(
        gen_throughput_tok_s=_get_scalar(merged, "sglang:gen_throughput"),
        total_prompt_tokens=int(_get_scalar(merged, "sglang:prompt_tokens_total")),
        total_generation_tokens=int(_get_scalar(merged, "sglang:generation_tokens_total")),
        total_requests=int(_get_scalar(merged, "sglang:num_requests_total")),
        total_aborted_requests=int(_get_scalar(merged, "sglang:num_aborted_requests_total")),
    )

    # ── Cache ───────────────────────────────────────────────
    cache = CacheMetrics(
        cache_hit_rate=_get_scalar(merged, "sglang:cache_hit_rate"),
        cached_tokens_total=int(_get_scalar(merged, "sglang:cached_tokens_total")),
        evicted_tokens_total=int(_get_scalar(merged, "sglang:evicted_tokens_total")),
        load_back_tokens_total=int(_get_scalar(merged, "sglang:load_back_tokens_total")),
        eviction_duration_avg_s=_get_histogram_avg(merged, "sglang:eviction_duration_seconds"),
        load_back_duration_avg_s=_get_histogram_avg(merged, "sglang:load_back_duration_seconds"),
        token_usage_ratio=_get_scalar(merged, "sglang:token_usage"),
        num_used_tokens=int(_get_scalar(merged, "sglang:num_used_tokens")),
        max_total_tokens=int(_get_scalar(merged, "sglang:max_total_num_tokens")),
    )

    # ── Scheduler ───────────────────────────────────────────
    scheduler = SchedulerMetrics(
        num_running_reqs=int(_get_scalar(merged, "sglang:num_running_reqs")),
        num_queue_reqs=int(_get_scalar(merged, "sglang:num_queue_reqs")),
        num_retracted_total=int(_get_scalar(merged, "sglang:num_retracted_requests_total")),
        num_paused_reqs=int(_get_scalar(merged, "sglang:num_paused_reqs")),
        gpu_execution_seconds_total=_get_scalar(merged, "sglang:gpu_execution_seconds_total"),
    )

    # ── Speculative Decoding ────────────────────────────────
    accept_len = _get_scalar(merged, "sglang:spec_accept_length")
    accept_rate = _get_scalar(merged, "sglang:spec_accept_rate")
    speculative = SpeculativeMetrics(
        accept_length_avg=accept_len,
        accept_rate=accept_rate,
        enabled=(accept_len > 0 or accept_rate > 0),
    )

    # ── Disaggregated (2P2D) KV Transfer ───────────────────
    disagg = DisaggMetrics(
        kv_transfer_speed_gb_s=_get_scalar(merged, "sglang:kv_transfer_speed_gb_s"),
        kv_transfer_latency_ms=_get_scalar(merged, "sglang:kv_transfer_latency_ms"),
        num_prefill_prealloc_queue=int(_get_scalar(merged, "sglang:num_prefill_prealloc_queue_reqs")),
    )

    return latency, throughput, cache, scheduler, speculative, disagg, merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3) nvidia-smi GPU hardware metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_NVIDIA_SMI_FIELDS = (
    "index,name,utilization.gpu,memory.used,memory.total,"
    "utilization.memory,temperature.gpu,power.draw,power.limit,"
    "clocks.current.sm,clocks.current.memory,"
    "pcie.link.gen.current,pcie.link.width.current,"
    "ecc.errors.corrected.volatile.total"
)


def _collect_gpu_metrics() -> list[GPUMetrics]:
    """Query nvidia-smi for per-GPU hardware metrics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + _NVIDIA_SMI_FIELDS,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    gpus: list[GPUMetrics] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue

        def _safe_float(val: str, default: float = 0.0) -> float:
            try:
                return float(val) if val not in ("[N/A]", "N/A", "[Not Supported]") else default
            except (ValueError, TypeError):
                return default

        def _safe_int(val: str, default: int = 0) -> int:
            try:
                return int(float(val)) if val not in ("[N/A]", "N/A", "[Not Supported]") else default
            except (ValueError, TypeError):
                return default

        gpus.append(GPUMetrics(
            gpu_id=_safe_int(parts[0]),
            name=parts[1],
            utilization_pct=_safe_float(parts[2]),
            memory_used_mib=_safe_float(parts[3]),
            memory_total_mib=_safe_float(parts[4]),
            memory_utilization_pct=_safe_float(parts[5]),
            temperature_c=_safe_float(parts[6]),
            power_draw_w=_safe_float(parts[7]),
            power_limit_w=_safe_float(parts[8]),
            sm_clock_mhz=_safe_int(parts[9]),
            mem_clock_mhz=_safe_int(parts[10]),
            pcie_tx_mb_s=0.0,   # Requires dcgmi or separate query
            pcie_rx_mb_s=0.0,   # Requires dcgmi or separate query
            ecc_errors_total=_safe_int(parts[13]) if len(parts) > 13 else 0,
        ))

    return gpus


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main collection entrypoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# Default SGLang worker ports for 2P2D setup
DEFAULT_WORKER_URLS = [
    "http://127.0.0.1:30000",  # Prefill worker 1
    "http://127.0.0.1:30002",  # Prefill worker 2
    "http://127.0.0.1:30001",  # Decode worker 1
    "http://127.0.0.1:30003",  # Decode worker 2
]


def collect(
    metrics_path: Path = Path("metrics.jsonl"),
    worker_urls: list[str] | None = None,
    scrape_prometheus: bool = True,
    scrape_gpu: bool = True,
) -> AggregatedMetrics:
    """Collect metrics from all sources and return a comprehensive snapshot.

    Args:
        metrics_path: Path to the proxy JSONL log.
        worker_urls: List of SGLang worker base URLs (e.g. http://host:port).
                     Defaults to the standard 2P2D ports.
        scrape_prometheus: Whether to scrape SGLang /metrics endpoints.
        scrape_gpu: Whether to query nvidia-smi for GPU hardware metrics.
    """
    t0 = time.perf_counter()

    if worker_urls is None:
        # Build from env vars or use defaults
        worker_urls = _build_worker_urls()

    # 1) Proxy JSONL
    proxy = _collect_proxy(metrics_path)

    # 2) SGLang Prometheus
    if scrape_prometheus:
        latency, throughput, cache, scheduler, speculative, disagg, raw_prom = (
            _collect_sglang_prometheus(worker_urls)
        )
    else:
        latency = LatencyMetrics()
        throughput = ThroughputMetrics()
        cache = CacheMetrics()
        scheduler = SchedulerMetrics()
        speculative = SpeculativeMetrics()
        disagg = DisaggMetrics()
        raw_prom = {}

    # 3) GPU hardware
    gpu_list = _collect_gpu_metrics() if scrape_gpu else []

    elapsed = time.perf_counter() - t0

    return AggregatedMetrics(
        timestamp=time.time(),
        collection_duration_s=round(elapsed, 4),
        proxy=proxy,
        latency=latency,
        throughput=throughput,
        gpu=gpu_list,
        cache=cache,
        scheduler=scheduler,
        speculative=speculative,
        disagg=disagg,
        raw_prometheus=raw_prom,
    )


def _build_worker_urls() -> list[str]:
    """Build worker URLs from environment variables or defaults."""
    host = os.environ.get("HOST_IP", "127.0.0.1")
    p1 = os.environ.get("PREFILL_PORT_1", "30000")
    p2 = os.environ.get("PREFILL_PORT_2", "30002")
    d1 = os.environ.get("DECODE_PORT_1", "30001")
    d2 = os.environ.get("DECODE_PORT_2", "30003")
    return [
        f"http://{host}:{p1}",
        f"http://{host}:{p2}",
        f"http://{host}:{d1}",
        f"http://{host}:{d2}",
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Continuous collection (for real-time dashboards / logging)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def collect_continuous(
    output_path: Path = Path("metrics_timeseries.jsonl"),
    interval_s: float = 5.0,
    duration_s: float | None = None,
    **kwargs,
) -> None:
    """Collect metrics at regular intervals and append to a JSONL file.

    Useful for capturing time-series data during a benchmark run.

    Args:
        output_path: Where to append periodic snapshots.
        interval_s: Seconds between collections.
        duration_s: Total duration to collect. None = run forever until Ctrl+C.
    """
    import signal

    stop = False

    def _stop(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    start = time.time()
    print(f"📊 Continuous metrics collection started → {output_path}")
    print(f"   Interval: {interval_s}s | Duration: {duration_s or '∞'}s")

    while not stop:
        if duration_s and (time.time() - start) >= duration_s:
            break

        m = collect(**kwargs)
        snapshot = _metrics_to_dict(m)

        with output_path.open("a") as f:
            f.write(json.dumps(snapshot, default=str) + "\n")

        time.sleep(interval_s)

    print(f"✅ Collection stopped. {output_path} is ready.")


def _metrics_to_dict(m: AggregatedMetrics) -> dict:
    """Convert AggregatedMetrics to a flat-ish dict for serialization."""
    result: dict[str, Any] = {
        "timestamp": m.timestamp,
        "collection_duration_s": m.collection_duration_s,
    }

    # Proxy
    for f_name in ProxyMetrics.__dataclass_fields__:
        result[f"proxy_{f_name}"] = getattr(m.proxy, f_name)

    # Latency
    for f_name in LatencyMetrics.__dataclass_fields__:
        result[f"latency_{f_name}"] = getattr(m.latency, f_name)

    # Throughput
    for f_name in ThroughputMetrics.__dataclass_fields__:
        result[f"throughput_{f_name}"] = getattr(m.throughput, f_name)

    # Cache
    for f_name in CacheMetrics.__dataclass_fields__:
        result[f"cache_{f_name}"] = getattr(m.cache, f_name)

    # Scheduler
    for f_name in SchedulerMetrics.__dataclass_fields__:
        result[f"sched_{f_name}"] = getattr(m.scheduler, f_name)

    # Speculative
    for f_name in SpeculativeMetrics.__dataclass_fields__:
        result[f"spec_{f_name}"] = getattr(m.speculative, f_name)

    # Disaggregated
    for f_name in DisaggMetrics.__dataclass_fields__:
        result[f"disagg_{f_name}"] = getattr(m.disagg, f_name)

    # GPU (per-GPU)
    for gpu in m.gpu:
        gid = gpu.gpu_id
        for f_name in GPUMetrics.__dataclass_fields__:
            result[f"gpu{gid}_{f_name}"] = getattr(gpu, f_name)

    return result
