"""Application services for task submission and queries.

Transport adapters pass upload objects through unchanged; this module owns the
workflow while depending only on injected ports.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Optional

from src.core.debug_logging import json_for_log, log_exception
from src.core.hotwords import parse_hotwords

logger = logging.getLogger(__name__)


class UploadTooLargeError(ValueError):
    """Raised when the existing temp-file store rejects an upload size."""


class TaskSubmissionUnavailableError(RuntimeError):
    """Raised before file IO when no queue consumer can accept a task."""


def build_submission_response(
    *,
    task_id: str,
    filename: str,
    file_size: int,
    email: Optional[str],
    vip: bool,
    message: str,
    extra: Optional[dict] = None,
) -> dict:
    payload = {
        "status": "success",
        "task_id": task_id,
        "filename": filename,
        "file_size": file_size,
        "file_size_mb": round(file_size / 1024 / 1024, 2),
        "email": email,
        "vip": vip,
        "message": message,
    }
    if extra:
        payload.update(extra)
    return payload


class TaskSubmissionService:
    def __init__(
        self,
        *,
        config: Any,
        temp_file_store: Any,
        task_repository: Any,
        speaker_task_repository: Any,
        scheduler: Any,
    ) -> None:
        self.config = config
        self.temp_file_store = temp_file_store
        self.task_repository = task_repository
        self.speaker_task_repository = speaker_task_repository
        self.scheduler = scheduler
        required_scheduler_methods = (
            "can_accept",
            "enqueue_offline",
            "enqueue_spk",
        )
        missing_methods = [
            name
            for name in required_scheduler_methods
            if not callable(getattr(scheduler, name, None))
        ]
        if missing_methods:
            raise TypeError(
                "scheduler is missing required methods: "
                + ", ".join(missing_methods)
            )

    def _max_file_size(self) -> int:
        return self.config.get("limits", {}).get("max_file_size_mb", 500) * 1024 * 1024

    def can_submit(self, mode: str) -> bool:
        return bool(self.scheduler.can_accept(mode))

    def _require_scheduler(self, mode: str) -> None:
        if not self.can_submit(mode):
            raise TaskSubmissionUnavailableError(
                f"{mode.upper()} task scheduling is unavailable"
            )

    async def submit_offline(
        self,
        upload: Any,
        *,
        email: Optional[str],
        hotwords: Optional[str],
        hotword_id: Optional[int],
        vip: bool,
    ) -> dict:
        self._require_scheduler("offline")
        if hotwords is not None:
            parse_hotwords(hotwords)
        return await self._submit_upload_task(
            upload,
            email=email,
            vip=vip,
            repository=self.task_repository,
            create_task=self.task_repository.create_task,
            enqueue=self.scheduler.enqueue_offline,
            create_extra={"hotwords": hotwords, "hotword_id": hotword_id},
            response_extra={"hotwords": hotwords, "hotword_id": hotword_id},
            message="任务已加入队列，正在处理",
        )

    async def submit_speaker(
        self,
        upload: Any,
        *,
        email: Optional[str],
        vip: bool,
    ) -> dict:
        self._require_scheduler("spk")
        return await self._submit_upload_task(
            upload,
            email=email,
            vip=vip,
            repository=self.speaker_task_repository,
            create_task=self.speaker_task_repository.create_task,
            enqueue=self.scheduler.enqueue_spk,
            message="SPK 任务已加入队列，正在处理",
        )

    async def _submit_upload_task(
        self,
        upload: Any,
        *,
        email: Optional[str],
        vip: bool,
        repository: Any,
        create_task: Callable[..., Any],
        enqueue: Callable[..., None],
        message: str,
        create_extra: Optional[dict] = None,
        response_extra: Optional[dict] = None,
    ) -> dict:
        task_id = str(uuid.uuid4())
        task_created = False
        max_file_size = self._max_file_size()
        logger.debug(
            "Task submission input: %s",
            json_for_log({
                "task_id": task_id,
                "filename": getattr(upload, "filename", None),
                "email": email,
                "vip": vip,
                "create_extra": create_extra,
                "response_extra": response_extra,
                "max_file_size": max_file_size,
            }),
        )
        try:
            try:
                file_path, file_size = await self.temp_file_store.save_upload(
                    upload, task_id, max_file_size
                )
                logger.debug(
                    "Task upload saved: %s",
                    json_for_log({
                        "task_id": task_id,
                        "file_path": str(file_path),
                        "file_size": file_size,
                    }),
                )
            except ValueError as exc:
                raise UploadTooLargeError(str(exc)) from exc

            await asyncio.to_thread(
                create_task,
                task_id=task_id,
                filename=upload.filename,
                file_size=file_size,
                email=email,
                vip=vip,
                **dict(create_extra or {}),
            )
            task_created = True
            logger.debug(
                "Task database record created: task_id=%s",
                task_id,
            )
            enqueue(task_id, vip=vip)
            response = build_submission_response(
                task_id=task_id,
                filename=upload.filename,
                file_size=file_size,
                email=email,
                vip=vip,
                message=message,
                extra=response_extra,
            )
            logger.debug(
                "Task submission output: %s",
                json_for_log(response),
            )
            return response
        except Exception:
            await self._compensate_failed_submission(
                task_id,
                repository=repository,
                task_created=task_created,
            )
            raise

    async def _compensate_failed_submission(
        self,
        task_id: str,
        *,
        repository: Any,
        task_created: bool,
    ) -> None:
        should_cleanup = not task_created
        if task_created:
            if repository is None:
                logger.error(
                    "Task submission compensation skipped: "
                    "task_id=%s repository_unresolved=true",
                    task_id,
                )
                return
            try:
                failed_task = await asyncio.to_thread(
                    repository.record_error,
                    task_id,
                    "Task submission failed before queue scheduling",
                    retry=False,
                )
                should_cleanup = failed_task is not None
                if failed_task is None:
                    logger.error(
                        "Task submission compensation failed: "
                        "task_id=%s task_not_found=true",
                        task_id,
                    )
            except Exception as exc:
                log_exception(
                    logger,
                    logging.ERROR,
                    "Task submission state compensation",
                    exc,
                    context={"task_id": task_id},
                )
                return

        if not should_cleanup:
            return
        try:
            await asyncio.to_thread(self.temp_file_store.cleanup, task_id)
        except Exception as exc:
            log_exception(
                logger,
                logging.WARNING,
                "Task submission temporary file cleanup",
                exc,
                context={"task_id": task_id},
            )

    async def get_speaker_task(self, task_id: str) -> Any:
        return await asyncio.to_thread(self.speaker_task_repository.get_task, task_id)

    async def get_offline_task(self, task_id: str) -> Any:
        return await asyncio.to_thread(self.task_repository.get_task, task_id)


__all__ = [
    "TaskSubmissionService",
    "TaskSubmissionUnavailableError",
    "UploadTooLargeError",
    "build_submission_response",
]
