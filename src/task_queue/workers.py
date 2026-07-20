"""Concurrent worker runtime for priority queued tasks."""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Awaitable, Callable, Optional

from src.task_queue.execution import (
    DelayedRetryScheduler,
    TaskExecutionRegistry,
    TaskKey,
)
from src.task_queue.policy import queue_priority_for


TaskProcessor = Callable[[str, str], Awaitable[bool]]
RetryVipResolver = Callable[[str, str], Awaitable[Optional[bool]]]


class QueueWorkerRuntime:
    """Own queue state, workers, duplicate guards, and delayed retries."""

    def __init__(
        self,
        *,
        process_task: TaskProcessor,
        resolve_retry_vip: RetryVipResolver,
        logger: logging.Logger,
        retry_delay_seconds: float = 1.0,
        shutdown_timeout_seconds: float = 30.0,
        worker_name_prefix: str = "offline-spk-queue-worker",
    ):
        self.process_task = process_task
        self.resolve_retry_vip = resolve_retry_vip
        self.logger = logger
        self.retry_delay_seconds = retry_delay_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.worker_name_prefix = worker_name_prefix

        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.registry = TaskExecutionRegistry()
        self.retry_scheduler = DelayedRetryScheduler()
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
        if not self.running and not self.workers and not self.retry_scheduler:
            return

        self.running = False
        await self.retry_scheduler.cancel_all()
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
            should_retry = False
            try:
                should_retry = not await self.process_task(kind, task_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                should_retry = True
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

            if should_retry:
                self._schedule_retry(kind, task_id)

    def _schedule_retry(self, kind: str, task_id: str) -> bool:
        if not self.running:
            return False
        key = (kind, task_id)
        return self.retry_scheduler.schedule(
            key,
            lambda: self._delayed_retry(kind, task_id),
            name=f"{kind}-retry-{task_id}",
        )

    async def _delayed_retry(self, kind: str, task_id: str) -> None:
        await asyncio.sleep(self.retry_delay_seconds)
        if not self.running:
            return

        try:
            vip = await self.resolve_retry_vip(kind, task_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                "Queue retry lookup failed: kind=%s task_id=%s error_type=%s",
                kind,
                task_id,
                type(exc).__name__,
            )
            self.logger.debug(
                "Queue retry lookup failure details: kind=%s task_id=%s",
                kind,
                task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return

        if vip is None:
            return

        key: TaskKey = (kind, task_id)
        self.retry_scheduler.release_current(key)
        self.enqueue(kind, task_id, vip=vip)


__all__ = ["QueueWorkerRuntime", "RetryVipResolver", "TaskProcessor"]
