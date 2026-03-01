"""run_benchmark.py — Single entrypoint for the SWE-bench benchmark harness.

Usage:
    python run_benchmark.py --domain swebench --concurrency 4 --n-tasks 50
    python run_benchmark.py --collect-metrics-only     # just scrape & print, no benchmark
    python run_benchmark.py --live-monitor              # continuous GPU/server monitoring
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from client.task_queue import TaskItem, run_queue
from client.turn_manager import run_multi_turn
from task_loaders.load_swebench import load_swebench_tasks
from metrics.collector import collect, collect_continuous
from metrics.report import (
    gpu_timeline_plot,
    markdown_summary,
    pareto_plot,
    throughput_timeline_plot,
)


async def _worker(task: TaskItem) -> str:
    """Process a single TaskItem through the multi-turn claude loop."""
    result = await run_multi_turn(task.prompt)
    return result.final_output


async def main(domain: str, concurrency: int, n_tasks: int | None) -> None:
    """Load tasks, run the benchmark queue, and print a report."""
    print(f"Loading tasks  domain={domain}  n_tasks={n_tasks}")
    tasks = load_swebench_tasks(n_tasks=n_tasks)

    print(f"Running {len(tasks)} tasks  concurrency={concurrency}")
    results = await run_queue(tasks, _worker, concurrency=concurrency)

    print(f"\nCompleted {len(results)} tasks")

    # Collect comprehensive metrics from all sources
    print("\n📊 Collecting metrics from proxy + SGLang Prometheus + nvidia-smi ...")
    metrics = collect()
    report = markdown_summary(metrics)
    print(report)

    # Save report
    report_path = Path("benchmark_report.md")
    report_path.write_text(report)
    print(f"\n📄 Report saved to {report_path}")

    # Generate plots
    try:
        pareto_plot("pareto.png")
        print("📈 Pareto plot saved to pareto.png")
    except Exception as e:
        print(f"⚠️  Pareto plot failed: {e}")

    # Try timeseries plots if continuous monitoring was running
    try:
        if Path("metrics_timeseries.jsonl").exists():
            gpu_timeline_plot()
            print("📈 GPU timeline saved to gpu_timeline.png")
            throughput_timeline_plot()
            print("📈 Throughput timeline saved to throughput_timeline.png")
    except Exception as e:
        print(f"⚠️  Timeline plots failed: {e}")


def cli() -> None:
    """Parse CLI args and launch the async main."""
    parser = argparse.ArgumentParser(description="SWE-bench benchmark runner")
    parser.add_argument("--domain", default="swebench", help="Task domain")
    parser.add_argument("--concurrency", type=int, default=4, help="Max parallel tasks")
    parser.add_argument("--n-tasks", type=int, default=None, help="Limit number of tasks")

    # Metrics-only modes
    parser.add_argument(
        "--collect-metrics-only", action="store_true",
        help="Just collect and print metrics, don't run benchmark",
    )
    parser.add_argument(
        "--live-monitor", action="store_true",
        help="Continuously collect metrics (GPU + server) every N seconds",
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=5.0,
        help="Interval in seconds for --live-monitor (default: 5)",
    )
    parser.add_argument(
        "--monitor-duration", type=float, default=None,
        help="Total duration for --live-monitor in seconds (default: unlimited)",
    )

    args = parser.parse_args()

    if args.collect_metrics_only:
        print("📊 Collecting metrics snapshot ...")
        metrics = collect()
        print(markdown_summary(metrics))
        return

    if args.live_monitor:
        collect_continuous(
            interval_s=args.monitor_interval,
            duration_s=args.monitor_duration,
        )
        return

    asyncio.run(main(args.domain, args.concurrency, args.n_tasks))


if __name__ == "__main__":
    cli()
