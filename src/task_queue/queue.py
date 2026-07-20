"""OFFLINE/SPK task queue facade."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from src.application.tasks import TaskSubmissionUnavailableError
from src.task_queue.batch import ImmediateBatchProcessor
from src.task_queue.policy import is_task_retriable
from src.task_queue.recovery import recover_and_enqueue_tasks
from src.task_queue.workers import QueueWorkerRuntime

logger = logging.getLogger(__name__)


class OfflineTaskQueue:
    """Coordinate domain services around a shared priority worker runtime."""

    def __init__(
        self,
        *,
        config,
        task_repository,
        batch_result_handler=None,
        task_service=None,
        speaker_task_repository=None,
        spk_task_service=None,
    ):
        processing_config = config.get_processing_config()
        self.enabled = processing_config.offline_async_enabled
        self.allow_immediate = processing_config.offline_async_allow_immediate
        self.max_concurrent = processing_config.max_concurrent_tasks
        self.processing_timeout = processing_config.timeout_seconds

        self.task_repository = task_repository
        self.speaker_task_repository = speaker_task_repository
        self.task_service = task_service
        self.spk_task_service = spk_task_service

        self._runtime = QueueWorkerRuntime(
            process_task=self._process_queued_task,
            resolve_retry_vip=self._resolve_retry_vip,
            logger=logger,
        )
        self._batch_processor = ImmediateBatchProcessor(
            enabled=self.allow_immediate and self.task_service is not None,
            task_repository=task_repository,
            task_service=task_service,
            result_handler=batch_result_handler,
            registry=self._runtime.registry,
        )

        logger.info(
            "Queue scheduler initialized: enabled=%s max_concurrent=%s "
            "allow_immediate=%s",
            self.enabled,
            self.max_concurrent,
            self.allow_immediate,
        )

    @property
    def retry_delay_seconds(self) -> float:
        return self._runtime.retry_delay_seconds

    @retry_delay_seconds.setter
    def retry_delay_seconds(self, value: float) -> None:
        self._runtime.retry_delay_seconds = float(value)

    @property
    def _queue(self):
        return self._runtime.queue

    @property
    def _task_registry(self):
        return self._runtime.registry

    @property
    def _retry_scheduler(self):
        return self._runtime.retry_scheduler

    @property
    def _workers(self):
        return self._runtime.workers

    @property
    def _running(self) -> bool:
        return self._runtime.running

    @property
    def is_running(self) -> bool:
        return self._runtime.running

    def supports(self, kind: str) -> bool:
        if kind == "offline":
            return self.task_service is not None
        if kind == "spk":
            return self.spk_task_service is not None
        return False

    def can_accept(self, kind: str) -> bool:
        return bool(
            self.enabled
            and self._runtime.running
            and self.supports(kind)
        )

    def start(self) -> None:
        if not self.enabled:
            logger.info("Queue scheduler is disabled")
            return
        if self._runtime.running:
            return
        if not any(self.supports(kind) for kind in ("offline", "spk")):
            logger.info("Queue scheduler startup skipped: no_task_services=true")
            return

        self._recover_pending_tasks()
        self._runtime.start(self.max_concurrent)
        logger.info("OFFLINE/SPK queue scheduler started")

    async def stop(self) -> None:
        await self._runtime.stop()
        logger.info("OFFLINE/SPK queue scheduler stopped")

    def enqueue_offline(self, task_id: str, vip: bool = False) -> None:
        self._enqueue("offline", task_id, vip)

    def enqueue_spk(self, task_id: str, vip: bool = False) -> None:
        self._enqueue("spk", task_id, vip)

    def _enqueue(
        self,
        kind: str,
        task_id: str,
        vip: bool = False,
        *,
        require_available: bool = True,
    ) -> None:
        if require_available and not self.can_accept(kind):
            raise TaskSubmissionUnavailableError(
                f"{kind.upper()} task scheduling is unavailable"
            )
        if self._runtime.enqueue(kind, task_id, vip):
            logger.info("Task queued: kind=%s task_id=%s vip=%s", kind, task_id, vip)

    def _recover_pending_tasks(self) -> None:
        if self.supports("offline"):
            recover_and_enqueue_tasks(
                repository=self.task_repository,
                timeout_seconds=self.processing_timeout,
                enqueue=lambda task_id, vip=False: self._enqueue(
                    "offline", task_id, vip, require_available=False
                ),
                task_kind="OFFLINE",
                logger=logger,
            )
        if self.supports("spk"):
            recover_and_enqueue_tasks(
                repository=self.speaker_task_repository,
                timeout_seconds=self.processing_timeout,
                enqueue=lambda task_id, vip=False: self._enqueue(
                    "spk", task_id, vip, require_available=False
                ),
                task_kind="SPK",
                logger=logger,
                missing_repository_is_error=False,
            )

    async def _process_queued_task(self, kind: str, task_id: str) -> bool:
        if kind == "spk":
            return await self._process_spk_task(task_id)
        if kind == "offline":
            return await self._process_single_task(task_id)
        raise ValueError(f"Unsupported queue task kind: {kind}")

    async def _resolve_retry_vip(
        self,
        kind: str,
        task_id: str,
    ) -> Optional[bool]:
        if kind == "spk" and self.spk_task_service is None:
            return None
        repository = self._repository_for_kind(kind)
        if repository is None:
            return None
        task = await asyncio.to_thread(repository.get_task, task_id)
        if not is_task_retriable(task):
            return None
        return bool(task.vip)

    def _repository_for_kind(self, kind: str):
        if kind == "offline":
            return self.task_repository
        if kind == "spk":
            return self.speaker_task_repository
        raise ValueError(f"Unsupported queue task kind: {kind}")

    async def _process_single_task(self, task_id: str) -> bool:
        if self.task_service is None:
            logger.error("OFFLINE task service is not configured")
            return False
        return await self.task_service.process_task(task_id)

    async def _process_spk_task(self, task_id: str) -> bool:
        if self.spk_task_service is None:
            logger.error("SPK task service is not configured")
            return False
        return await self.spk_task_service.process_task(task_id)

    async def process_batch_now(self, task_ids: List[str]) -> dict:
        logger.info("Immediate OFFLINE batch requested: task_count=%s", len(task_ids))
        return await self._batch_processor.process(task_ids)


__all__ = ["OfflineTaskQueue"]
