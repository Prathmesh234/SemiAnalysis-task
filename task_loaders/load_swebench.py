"""load_swebench.py — Return a list of TaskItem dataclasses from SWE-bench Verified."""

from __future__ import annotations

from datasets import load_dataset

from client.task_queue import TaskItem

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"


def load_swebench_tasks(split: str = "test", n_tasks: int | None = None) -> list[TaskItem]:
    """Load SWE-bench Verified and convert rows to TaskItem instances.

    Parameters
    ----------
    split:
        HuggingFace dataset split to use.
    n_tasks:
        If set, truncate to the first *n_tasks* items.
    """
    ds = load_dataset(DATASET_NAME, split=split)

    tasks: list[TaskItem] = []
    for row in ds:
        tasks.append(
            TaskItem(
                instance_id=row["instance_id"],
                prompt=row["problem_statement"],
                repo=row.get("repo", ""),
                base_commit=row.get("base_commit", ""),
                metadata={"hints": row.get("hints_text", "")},
            )
        )
        if n_tasks and len(tasks) >= n_tasks:
            break

    return tasks
