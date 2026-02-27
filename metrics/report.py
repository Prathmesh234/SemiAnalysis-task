"""report.py — Generate comprehensive markdown reports, Pareto plots, and GPU dashboards."""

from __future__ import annotations

from pathlib import Path

from metrics.collector import (
    AggregatedMetrics,
    GPUMetrics,
    collect,
)


def markdown_summary(metrics: AggregatedMetrics | None = None) -> str:
    """Return a full markdown report covering every metric domain."""
    if metrics is None:
        metrics = collect()

    sections: list[str] = []

    # ── Header ──────────────────────────────────────────────
    sections.append("# 📊 SGLang Benchmark Metrics Report\n")

    # ── 1) Proxy Request Summary ────────────────────────────
    p = metrics.proxy
    sections.append("## 1. Request Summary (Proxy)\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Total requests | {p.total_requests} |")
    sections.append(f"| Avg latency | {p.avg_latency_s:.3f}s |")
    sections.append(f"| P50 latency | {p.p50_latency_s:.3f}s |")
    sections.append(f"| P90 latency | {p.p90_latency_s:.3f}s |")
    sections.append(f"| P95 latency | {p.p95_latency_s:.3f}s |")
    sections.append(f"| P99 latency | {p.p99_latency_s:.3f}s |")
    sections.append(f"| Min latency | {p.min_latency_s:.3f}s |")
    sections.append(f"| Max latency | {p.max_latency_s:.3f}s |")
    sections.append(f"| Prompt tokens | {p.total_prompt_tokens:,} |")
    sections.append(f"| Completion tokens | {p.total_completion_tokens:,} |")
    sections.append(f"| Cached tokens | {p.total_cached_tokens:,} |")
    sections.append(f"| Avg tokens/request | {p.avg_tokens_per_request:.1f} |")
    sections.append(f"| Total request bytes | {p.total_req_bytes:,} |")
    sections.append(f"| Total response bytes | {p.total_resp_bytes:,} |")
    if p.total_latency_s > 0:
        overall_tok_s = p.total_completion_tokens / p.total_latency_s
        sections.append(f"| **Overall throughput** | **{overall_tok_s:.1f} tok/s** |")
    sections.append("")

    # ── 2) SGLang Latency (Prometheus) ──────────────────────
    lat = metrics.latency
    sections.append("## 2. Latency Breakdown (SGLang Prometheus)\n")
    sections.append("| Metric | Avg | P50 | P95 | P99 |")
    sections.append("|--------|-----|-----|-----|-----|")
    sections.append(
        f"| **TTFT** (Time to First Token) | {lat.ttft_avg_s*1000:.1f}ms "
        f"| {lat.ttft_p50_s*1000:.1f}ms | {lat.ttft_p95_s*1000:.1f}ms "
        f"| {lat.ttft_p99_s*1000:.1f}ms |"
    )
    sections.append(
        f"| **ITL** (Inter-Token Latency) | {lat.itl_avg_s*1000:.1f}ms "
        f"| {lat.itl_p50_s*1000:.1f}ms | {lat.itl_p95_s*1000:.1f}ms "
        f"| {lat.itl_p99_s*1000:.1f}ms |"
    )
    sections.append(
        f"| **E2E** (End-to-End) | {lat.e2e_avg_s:.3f}s "
        f"| {lat.e2e_p50_s:.3f}s | {lat.e2e_p95_s:.3f}s "
        f"| {lat.e2e_p99_s:.3f}s |"
    )
    sections.append(
        f"| **Queue time** | {lat.queue_time_avg_s*1000:.1f}ms "
        f"| — | — | {lat.queue_time_p99_s*1000:.1f}ms |"
    )
    # Derived: TPOT estimate
    if lat.itl_avg_s > 0:
        tpot_ms = lat.itl_avg_s * 1000
        sections.append(f"\n> **TPOT** (Time Per Output Token) ≈ **{tpot_ms:.1f}ms**")
    sections.append("")

    # ── 3) Throughput ───────────────────────────────────────
    tp = metrics.throughput
    sections.append("## 3. Throughput (SGLang Prometheus)\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Generation throughput | {tp.gen_throughput_tok_s:.1f} tok/s |")
    sections.append(f"| Total prompt tokens processed | {tp.total_prompt_tokens:,} |")
    sections.append(f"| Total generation tokens processed | {tp.total_generation_tokens:,} |")
    sections.append(f"| Total requests (server-side) | {tp.total_requests:,} |")
    sections.append(f"| Aborted requests | {tp.total_aborted_requests:,} |")
    sections.append("")

    # ── 4) GPU Hardware ─────────────────────────────────────
    if metrics.gpu:
        sections.append("## 4. GPU Hardware (nvidia-smi)\n")
        sections.append("| GPU | Name | Util% | Mem Used | Mem Total | Mem BW% | Temp°C | Power W | SM MHz | Mem MHz |")
        sections.append("|-----|------|-------|----------|-----------|---------|--------|---------|--------|---------|")
        for g in metrics.gpu:
            sections.append(
                f"| {g.gpu_id} | {g.name} | {g.utilization_pct:.0f}% "
                f"| {g.memory_used_mib:.0f} MiB | {g.memory_total_mib:.0f} MiB "
                f"| {g.memory_utilization_pct:.0f}% | {g.temperature_c:.0f}°C "
                f"| {g.power_draw_w:.0f}/{g.power_limit_w:.0f} "
                f"| {g.sm_clock_mhz} | {g.mem_clock_mhz} |"
            )

        # Averages
        if len(metrics.gpu) > 1:
            avg_util = sum(g.utilization_pct for g in metrics.gpu) / len(metrics.gpu)
            avg_mem_util = sum(g.memory_utilization_pct for g in metrics.gpu) / len(metrics.gpu)
            total_power = sum(g.power_draw_w for g in metrics.gpu)
            avg_temp = sum(g.temperature_c for g in metrics.gpu) / len(metrics.gpu)
            sections.append(f"\n**Averages:** Util={avg_util:.0f}% | Mem BW={avg_mem_util:.0f}% | "
                            f"Total Power={total_power:.0f}W | Avg Temp={avg_temp:.0f}°C")

        # Role-based analysis for 2P2D
        if len(metrics.gpu) >= 4:
            prefill_gpus = metrics.gpu[:2]
            decode_gpus = metrics.gpu[2:4]
            avg_pf_util = sum(g.utilization_pct for g in prefill_gpus) / 2
            avg_dc_util = sum(g.utilization_pct for g in decode_gpus) / 2
            avg_pf_mem = sum(g.memory_utilization_pct for g in prefill_gpus) / 2
            avg_dc_mem = sum(g.memory_utilization_pct for g in decode_gpus) / 2
            sections.append(f"\n**2P2D Role Analysis:**")
            sections.append(f"- Prefill GPUs (0,1): SM Util={avg_pf_util:.0f}% | Mem BW={avg_pf_mem:.0f}%")
            sections.append(f"- Decode GPUs (2,3):  SM Util={avg_dc_util:.0f}% | Mem BW={avg_dc_mem:.0f}%")
        sections.append("")

    # ── 5) KV Cache ─────────────────────────────────────────
    c = metrics.cache
    sections.append("## 5. KV Cache & Prefix Cache\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Cache hit rate | {c.cache_hit_rate*100:.1f}% |")
    sections.append(f"| Tokens served from cache | {c.cached_tokens_total:,} |")
    sections.append(f"| Tokens evicted (GPU → CPU) | {c.evicted_tokens_total:,} |")
    sections.append(f"| Tokens loaded back (CPU → GPU) | {c.load_back_tokens_total:,} |")
    sections.append(f"| Eviction duration (avg) | {c.eviction_duration_avg_s*1000:.1f}ms |")
    sections.append(f"| Load-back duration (avg) | {c.load_back_duration_avg_s*1000:.1f}ms |")
    sections.append(f"| KV cache utilization | {c.token_usage_ratio*100:.1f}% |")
    sections.append(f"| Tokens in KV cache | {c.num_used_tokens:,} |")
    sections.append(f"| Max KV cache capacity | {c.max_total_tokens:,} |")
    sections.append("")

    # ── 6) Scheduler ────────────────────────────────────────
    s = metrics.scheduler
    sections.append("## 6. Scheduler State\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Running requests | {s.num_running_reqs} |")
    sections.append(f"| Queued requests | {s.num_queue_reqs} |")
    sections.append(f"| Retracted (preempted) | {s.num_retracted_total} |")
    sections.append(f"| Paused requests | {s.num_paused_reqs} |")
    sections.append(f"| GPU execution time (total) | {s.gpu_execution_seconds_total:.2f}s |")
    sections.append("")

    # ── 7) Speculative Decoding ─────────────────────────────
    sp = metrics.speculative
    sections.append("## 7. Speculative Decoding (MTP/EAGLE)\n")
    if sp.enabled:
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.append(f"| Avg accepted length | {sp.accept_length_avg:.2f} tokens |")
        sections.append(f"| Accept rate | {sp.accept_rate*100:.1f}% |")
    else:
        sections.append("*Not active — no speculative decoding metrics detected.*")
    sections.append("")

    # ── 8) Disaggregated KV Transfer ────────────────────────
    d = metrics.disagg
    sections.append("## 8. Disaggregated KV Transfer (2P2D / Mooncake)\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| KV transfer speed | {d.kv_transfer_speed_gb_s:.2f} GB/s |")
    sections.append(f"| KV transfer latency | {d.kv_transfer_latency_ms:.2f}ms |")
    sections.append(f"| Prefill pre-alloc queue | {d.num_prefill_prealloc_queue} |")
    sections.append("")

    # ── Footer ──────────────────────────────────────────────
    sections.append(f"---\n*Collected in {metrics.collection_duration_s:.3f}s*")

    return "\n".join(sections)


def pareto_plot(output_path: Path = Path("pareto.png")) -> Path:
    """Generate a Pareto frontier plot of latency vs throughput and save to *output_path*.

    Requires matplotlib.
    """
    import json

    import matplotlib.pyplot as plt

    metrics = collect()
    if metrics.proxy.total_requests == 0:
        return output_path

    # Read per-request data for scatter plot
    metrics_path = Path("metrics.jsonl")
    per_req_latency = []
    per_req_throughput = []
    if metrics_path.exists():
        with metrics_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                lat = r.get("latency_s", 0)
                tokens = r.get("usage", {}).get("completion_tokens", 0)
                if lat > 0:
                    per_req_latency.append(lat)
                    per_req_throughput.append(tokens / lat if tokens else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    if per_req_latency:
        ax.scatter(per_req_latency, per_req_throughput, alpha=0.4, s=20, label="Per-request")

    # Highlight aggregate average
    p = metrics.proxy
    avg_throughput = p.total_requests / max(p.total_latency_s, 1e-6)
    ax.scatter(
        [p.avg_latency_s], [avg_throughput],
        marker="*", s=200, c="red", zorder=5, label="Average",
    )

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Latency vs Throughput — GLM-4.5-Air-FP8 2P2D")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def gpu_timeline_plot(
    timeseries_path: Path = Path("metrics_timeseries.jsonl"),
    output_path: Path = Path("gpu_timeline.png"),
) -> Path:
    """Plot GPU utilization, memory, temperature, and power over time.

    Requires metrics_timeseries.jsonl from `collect_continuous()`.
    """
    import json

    import matplotlib.pyplot as plt

    if not timeseries_path.exists():
        return output_path

    records = []
    with timeseries_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return output_path

    # Find how many GPUs
    num_gpus = 0
    for i in range(16):
        if f"gpu{i}_gpu_id" in records[0]:
            num_gpus = i + 1
        else:
            break

    if num_gpus == 0:
        return output_path

    t0 = records[0]["timestamp"]
    times = [(r["timestamp"] - t0) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for gpu_id in range(num_gpus):
        label = f"GPU {gpu_id}"

        # Utilization
        utils = [r.get(f"gpu{gpu_id}_utilization_pct", 0) for r in records]
        axes[0, 0].plot(times, utils, label=label, alpha=0.8)

        # Memory used
        mems = [r.get(f"gpu{gpu_id}_memory_used_mib", 0) for r in records]
        axes[0, 1].plot(times, mems, label=label, alpha=0.8)

        # Temperature
        temps = [r.get(f"gpu{gpu_id}_temperature_c", 0) for r in records]
        axes[1, 0].plot(times, temps, label=label, alpha=0.8)

        # Power
        powers = [r.get(f"gpu{gpu_id}_power_draw_w", 0) for r in records]
        axes[1, 1].plot(times, powers, label=label, alpha=0.8)

    axes[0, 0].set_ylabel("SM Util %")
    axes[0, 0].set_title("GPU Utilization")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_ylabel("Memory (MiB)")
    axes[0, 1].set_title("GPU Memory Used")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_ylabel("Temperature °C")
    axes[1, 0].set_title("GPU Temperature")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_ylabel("Power (W)")
    axes[1, 1].set_title("GPU Power Draw")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("GPU Hardware Timeline — GLM-4.5-Air-FP8 2P2D", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def throughput_timeline_plot(
    timeseries_path: Path = Path("metrics_timeseries.jsonl"),
    output_path: Path = Path("throughput_timeline.png"),
) -> Path:
    """Plot throughput, queue depth, and cache metrics over time."""
    import json

    import matplotlib.pyplot as plt

    if not timeseries_path.exists():
        return output_path

    records = []
    with timeseries_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return output_path

    t0 = records[0]["timestamp"]
    times = [(r["timestamp"] - t0) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # Throughput
    gen_tps = [r.get("throughput_gen_throughput_tok_s", 0) for r in records]
    axes[0, 0].plot(times, gen_tps, label="Gen tok/s", color="green", alpha=0.8)
    axes[0, 0].set_ylabel("Tokens/s")
    axes[0, 0].set_title("Generation Throughput")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Queue depth
    running = [r.get("sched_num_running_reqs", 0) for r in records]
    queued = [r.get("sched_num_queue_reqs", 0) for r in records]
    axes[0, 1].plot(times, running, label="Running", color="blue", alpha=0.8)
    axes[0, 1].plot(times, queued, label="Queued", color="orange", alpha=0.8)
    axes[0, 1].set_ylabel("Requests")
    axes[0, 1].set_title("Scheduler Queue")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Cache hit rate
    cache_hr = [r.get("cache_cache_hit_rate", 0) * 100 for r in records]
    axes[1, 0].plot(times, cache_hr, label="Cache Hit %", color="purple", alpha=0.8)
    axes[1, 0].set_ylabel("Hit Rate %")
    axes[1, 0].set_title("Prefix Cache Hit Rate")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # KV cache utilization
    kv_util = [r.get("cache_token_usage_ratio", 0) * 100 for r in records]
    axes[1, 1].plot(times, kv_util, label="KV Usage %", color="red", alpha=0.8)
    axes[1, 1].set_ylabel("Utilization %")
    axes[1, 1].set_title("KV Cache Utilization")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Throughput & Cache Timeline — GLM-4.5-Air-FP8 2P2D", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
