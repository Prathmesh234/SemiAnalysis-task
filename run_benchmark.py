"""run_benchmark.py — Single entrypoint for the SWE-bench benchmark harness.

Usage:
    python run_benchmark.py --domain swebench --concurrency 4 --n-tasks 50
"""

from __future__ import annotations

import argparse
import asyncio

from client.task_queue import TaskItem, run_queue
from client.turn_manager import run_multi_turn
from datasets.load_swebench import load_swebench_tasks
from metrics.collector import collect
from metrics.report import markdown_summary


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
    metrics = collect()
    print(markdown_summary(metrics))


def cli() -> None:
    """Parse CLI args and launch the async main."""
    parser = argparse.ArgumentParser(description="SWE-bench benchmark runner")
    parser.add_argument("--domain", default="swebench", help="Task domain")
    parser.add_argument("--concurrency", type=int, default=4, help="Max parallel tasks")
    parser.add_argument("--n-tasks", type=int, default=None, help="Limit number of tasks")
    args = parser.parse_args()

    asyncio.run(main(args.domain, args.concurrency, args.n_tasks))


if __name__ == "__main__":
    cli()
