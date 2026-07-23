"""Execution state helpers for the OFFLINE/SPK task queue."""

from __future__ import annotations

from typing import Tuple


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


__all__ = ["TaskExecutionRegistry", "TaskKey"]
