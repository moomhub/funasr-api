"""Execution state helpers for the OFFLINE/SPK task queue."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, Tuple


TaskKey = Tuple[str, str]


class TaskExecutionRegistry:
    """Track queued and in-flight task keys without changing queue payloads."""

    def __init__(self):
        self._queued: set[TaskKey] = set()
        self._inflight: set[TaskKey] = set()

    def try_enqueue(self, key: TaskKey) -> bool:
        if self.is_tracked(key):
            return False
        self._queued.add(key)
        return True

    def try_begin_external(self, key: TaskKey) -> bool:
        if self.is_tracked(key):
            return False
        self._inflight.add(key)
        return True

    def begin_queued(self, key: TaskKey) -> None:
        self._queued.discard(key)
        self._inflight.add(key)

    def finish(self, key: TaskKey) -> None:
        self._inflight.discard(key)

    def is_tracked(self, key: TaskKey) -> bool:
        return key in self._queued or key in self._inflight

    def clear(self) -> None:
        self._queued.clear()
        self._inflight.clear()


class DelayedRetryScheduler:
    """Keep at most one delayed retry task for each task key."""

    def __init__(self):
        self._tasks: Dict[TaskKey, asyncio.Task] = {}

    def __bool__(self) -> bool:
        return bool(self._tasks)

    def schedule(
        self,
        key: TaskKey,
        coroutine_factory: Callable[[], Awaitable[None]],
        *,
        name: str,
    ) -> bool:
        if key in self._tasks:
            return False
        task = asyncio.create_task(coroutine_factory(), name=name)
        self._tasks[key] = task
        task.add_done_callback(lambda completed: self._discard(key, completed))
        return True

    def release_current(self, key: TaskKey) -> bool:
        current = asyncio.current_task()
        if current is None or self._tasks.get(key) is not current:
            return False
        self._tasks.pop(key, None)
        return True

    async def cancel_all(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _discard(self, key: TaskKey, completed: asyncio.Task) -> None:
        if self._tasks.get(key) is completed:
            self._tasks.pop(key, None)


__all__ = ["DelayedRetryScheduler", "TaskExecutionRegistry", "TaskKey"]
