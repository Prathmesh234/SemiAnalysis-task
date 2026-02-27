"""task_queue.py — asyncio.Queue that pops TaskItem dicts with semaphore-limited concurrency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class TaskItem:
    """A single coding task to be executed."""

    instance_id: str
    prompt: str
    repo: str = ""
    base_commit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


async def run_queue(
    tasks: list[TaskItem],
    worker: Callable[[TaskItem], Awaitable[Any]],
    concurrency: int = 4,
) -> list[Any]:
    """Feed *tasks* through an asyncio.Queue processed by *concurrency* workers.

    Returns a list of results in task order.
    """
    queue: asyncio.Queue[tuple[int, TaskItem]] = asyncio.Queue()
    results: dict[int, Any] = {}
    sem = asyncio.Semaphore(concurrency)

    for idx, task in enumerate(tasks):
        queue.put_nowait((idx, task))

    async def _worker() -> None:
        while not queue.empty():
            async with sem:
                try:
                    idx, task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results[idx] = await worker(task)
                queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    return [results[i] for i in sorted(results)]
