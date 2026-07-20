"""Immediate OFFLINE batch processing outside the worker queue."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, List

from src.application.task_results import OfflineBatchContext
from src.task_queue.execution import TaskExecutionRegistry
from src.task_queue.policy import build_batch_result

logger = logging.getLogger(__name__)


class ImmediateBatchProcessor:
    def __init__(
        self,
        *,
        enabled: bool,
        task_repository: Any,
        task_service: Any,
        result_handler: Any,
        registry: TaskExecutionRegistry,
    ):
        self.enabled = enabled
        self.task_repository = task_repository
        self.task_service = task_service
        self.result_handler = result_handler
        self.registry = registry

    async def process(self, task_ids: Iterable[str]) -> dict:
        if not self.enabled:
            raise RuntimeError("OFFLINE immediate batch processing is disabled")

        requested_ids = list(task_ids)
        start_time = datetime.now(timezone.utc)
        task_results: List[bool] = []
        for task_id in requested_ids:
            task_results.append(await self._process_one(task_id))

        results = build_batch_result(
            requested_ids,
            task_results,
            start_time,
            datetime.now(timezone.utc),
        )
        await self.result_handler.handle_complete(
            OfflineBatchContext(
                task_ids=requested_ids,
                total_tasks=results["total"],
                completed_tasks=results["completed"],
                failed_tasks=results["failed"],
                processing_time=results["processing_time"],
            )
        )
        return results

    async def _process_one(self, task_id: str) -> bool:
        try:
            task = await asyncio.to_thread(
                self.task_repository.get_task,
                task_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_task_failure(task_id, "lookup", exc)
            return False
        if task is None:
            return False

        key = ("offline", task_id)
        if not self.registry.try_begin_external(key):
            return False
        try:
            return bool(await self.task_service.process_task(task_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_task_failure(task_id, "process", exc)
            return False
        finally:
            self.registry.finish(key)

    @staticmethod
    def _log_task_failure(
        task_id: str,
        phase: str,
        exc: Exception,
    ) -> None:
        logger.error(
            "Immediate batch task failed: task_id=%s phase=%s error_type=%s",
            task_id,
            phase,
            type(exc).__name__,
        )
        logger.debug(
            "Immediate batch task failure details: task_id=%s phase=%s",
            task_id,
            phase,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


__all__ = ["ImmediateBatchProcessor"]
