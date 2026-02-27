"""report.py — Generate Pareto plots and a markdown summary table from collected metrics."""

from __future__ import annotations

from pathlib import Path

from metrics.collector import AggregatedMetrics, collect


def markdown_summary(metrics: AggregatedMetrics | None = None) -> str:
    """Return a markdown table summarising the benchmark run."""
    if metrics is None:
        metrics = collect()

    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total requests | {metrics.total_requests} |",
        f"| Avg latency (s) | {metrics.avg_latency_s:.3f} |",
        f"| P50 latency (s) | {metrics.p50_latency_s:.3f} |",
        f"| P99 latency (s) | {metrics.p99_latency_s:.3f} |",
        f"| Prompt tokens | {metrics.total_prompt_tokens} |",
        f"| Completion tokens | {metrics.total_completion_tokens} |",
    ]
    return "\n".join(lines)


def pareto_plot(output_path: Path = Path("pareto.png")) -> Path:
    """Generate a Pareto frontier plot of latency vs throughput and save to *output_path*.

    Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    metrics = collect()

    # Stub: single-point plot; extend with per-concurrency sweeps
    fig, ax = plt.subplots()
    ax.scatter(
        [metrics.avg_latency_s],
        [metrics.total_requests / max(metrics.total_latency_s, 1e-6)],
        marker="o",
    )
    ax.set_xlabel("Avg Latency (s)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Latency vs Throughput")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
