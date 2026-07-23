"""Pure task queue policy helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence


def queue_priority_for(vip: bool) -> int:
    return 0 if vip else 1


def build_batch_result(
    task_ids: Sequence[str],
    task_results: Iterable[bool],
    start_time: datetime,
    end_time: datetime,
) -> dict:
    completed = sum(1 for result in task_results if result)
    total = len(task_ids)
    return {
        "total": total,
        "completed": completed,
        "failed": total - completed,
        "processing_time": (end_time - start_time).total_seconds(),
    }


__all__ = [
    "build_batch_result",
    "queue_priority_for",
]
