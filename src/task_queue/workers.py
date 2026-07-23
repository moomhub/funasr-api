"""Concurrent worker runtime for priority queued tasks."""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Awaitable, Callable

from src.task_queue.execution import (
    TaskExecutionRegistry,
)
from src.task_queue.policy import queue_priority_for


TaskProcessor = Callable[[str, str], Awaitable[bool]]


class QueueWorkerRuntime:
    """Own queue state, workers, and duplicate guards."""

    def __init__(
        self,
        *,
        process_task: TaskProcessor,
        logger: logging.Logger,
        shutdown_timeout_seconds: float = 30.0,
        worker_name_prefix: str = "offline-spk-queue-worker",
    ):
        self.process_task = process_task
        self.logger = logger
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.worker_name_prefix = worker_name_prefix

        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.registry = TaskExecutionRegistry()
        self.workers: list[asyncio.Task] = []
        self.running = False
        self._counter = itertools.count()

    def start(self, worker_count: int) -> None:
        if self.running:
            return
        self.running = True
        for index in range(max(1, worker_count)):
            worker = asyncio.create_task(
                self._worker_loop(),
                name=f"{self.worker_name_prefix}-{index + 1}",
            )
            self.workers.append(worker)

    async def stop(self) -> None:
        if not self.running and not self.workers:
            return

        self.running = False
        workers = list(self.workers)
        for _ in workers:
            self.queue.put_nowait(
                (-1, next(self._counter), "__stop__", "")
            )
        if workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*workers, return_exceptions=True),
                    timeout=self.shutdown_timeout_seconds,
                )
            except asyncio.TimeoutError:
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

        self.workers.clear()
        self.queue = asyncio.PriorityQueue()
        self.registry.clear()

    def enqueue(self, kind: str, task_id: str, vip: bool = False) -> bool:
        key = (kind, task_id)
        if not self.registry.try_enqueue(key):
            return False
        self.queue.put_nowait(
            (queue_priority_for(vip), next(self._counter), kind, task_id)
        )
        return True

    async def _worker_loop(self) -> None:
        while True:
            _priority, _order, kind, task_id = await self.queue.get()
            if kind == "__stop__":
                self.queue.task_done()
                return

            key = (kind, task_id)
            self.registry.begin_queued(key)
            try:
                await self.process_task(kind, task_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error(
                    "Queue task failed: kind=%s task_id=%s error_type=%s",
                    kind,
                    task_id,
                    type(exc).__name__,
                )
                self.logger.debug(
                    "Queue task failure details: kind=%s task_id=%s",
                    kind,
                    task_id,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            finally:
                self.registry.finish(key)
                self.queue.task_done()


__all__ = ["QueueWorkerRuntime", "TaskProcessor"]
