"""Application services for task submission and queries.

Transport adapters pass upload objects through unchanged; this module owns the
workflow while depending only on injected ports.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from src.application.postprocess import calculate_file_hash
from src.core.debug_logging import json_for_log, log_exception
from src.core.hotwords import parse_hotwords
from src.core.ids import new_task_id

logger = logging.getLogger(__name__)


class UploadTooLargeError(ValueError):
    """Raised when the existing temp-file store rejects an upload size."""


class TaskSubmissionUnavailableError(RuntimeError):
    """Raised before file IO when no queue consumer can accept a task."""


class RerecognitionSourceNotFoundError(LookupError):
    """Raised when the source task does not exist or is soft-deleted."""


class RerecognitionValidationError(ValueError):
    """Raised when rerecognition override parameters are ambiguous."""


class RerecognitionConflictError(RuntimeError):
    """Raised when the source task cannot be used for rerecognition."""


class ArchiveRestoreError(RuntimeError):
    """Raised when archived audio cannot be restored or verified."""


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
        audio_backup_store: Any = None,
    ) -> None:
        self.config = config
        self.temp_file_store = temp_file_store
        self.task_repository = task_repository
        self.speaker_task_repository = speaker_task_repository
        self.scheduler = scheduler
        self.audio_backup_store = audio_backup_store
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

    async def rerecognize_offline(
        self,
        source_task_id: str,
        *,
        email: Optional[str] = None,
        vip: Optional[bool] = None,
        hotwords: Optional[str] = None,
        hotword_id: Optional[int] = None,
    ) -> dict:
        self._require_scheduler("offline")
        source_task = await asyncio.to_thread(
            self.task_repository.get_task,
            source_task_id,
        )
        self._validate_rerecognition_source(source_task)
        if hotwords is not None and hotword_id is not None:
            raise RerecognitionValidationError(
                "hotwords 与 hotword_id 不能同时覆盖"
            )

        effective_hotwords = source_task.hotwords
        effective_hotword_id = source_task.hotword_id
        if hotwords is not None:
            parse_hotwords(hotwords)
            effective_hotwords = hotwords
            effective_hotword_id = None
        elif hotword_id is not None:
            effective_hotwords = None
            effective_hotword_id = hotword_id

        effective_email = source_task.email if email is None else email
        effective_vip = bool(source_task.vip if vip is None else vip)
        return await self._submit_restored_task(
            mode="offline",
            source_task=source_task,
            email=effective_email,
            vip=effective_vip,
            repository=self.task_repository,
            create_task=self.task_repository.create_task,
            enqueue=self.scheduler.enqueue_offline,
            create_extra={
                "hotwords": effective_hotwords,
                "hotword_id": effective_hotword_id,
            },
            response_extra={
                "hotwords": effective_hotwords,
                "hotword_id": effective_hotword_id,
            },
            message="OFFLINE 重识别任务已加入队列，正在处理",
        )

    async def rerecognize_speaker(
        self,
        source_task_id: str,
        *,
        email: Optional[str] = None,
        vip: Optional[bool] = None,
    ) -> dict:
        self._require_scheduler("spk")
        source_task = await asyncio.to_thread(
            self.speaker_task_repository.get_task,
            source_task_id,
        )
        self._validate_rerecognition_source(source_task)
        effective_email = source_task.email if email is None else email
        effective_vip = bool(source_task.vip if vip is None else vip)
        return await self._submit_restored_task(
            mode="spk",
            source_task=source_task,
            email=effective_email,
            vip=effective_vip,
            repository=self.speaker_task_repository,
            create_task=self.speaker_task_repository.create_task,
            enqueue=self.scheduler.enqueue_spk,
            message="SPK 重识别任务已加入队列，正在处理",
        )

    @staticmethod
    def _validate_rerecognition_source(source_task: Any) -> None:
        if source_task is None or bool(getattr(source_task, "is_deleted", False)):
            raise RerecognitionSourceNotFoundError("来源任务不存在")
        if getattr(source_task, "status", None) not in {"completed", "failed"}:
            raise RerecognitionConflictError("来源任务当前状态不可重识别")
        if not getattr(source_task, "s3_key", None):
            raise RerecognitionConflictError("来源任务缺少归档文件信息")

    async def _submit_restored_task(
        self,
        *,
        mode: str,
        source_task: Any,
        email: Optional[str],
        vip: bool,
        repository: Any,
        create_task: Callable[..., Any],
        enqueue: Callable[..., None],
        message: str,
        create_extra: Optional[dict] = None,
        response_extra: Optional[dict] = None,
    ) -> dict:
        task_id = new_task_id()
        destination = Path(
            self.temp_file_store.resolve(task_id, source_task.filename)
        )
        file_size, file_hash = await self._restore_and_verify_audio(
            source_task=source_task,
            task_id=task_id,
            mode=mode,
            destination=destination,
        )

        task_created = False
        try:
            await asyncio.to_thread(
                create_task,
                task_id=task_id,
                filename=source_task.filename,
                file_size=file_size,
                email=email,
                vip=vip,
                source_task_id=source_task.id,
                s3_key=source_task.s3_key,
                file_hash=file_hash,
                **dict(create_extra or {}),
            )
            task_created = True
            enqueue(task_id, vip=vip)
        except Exception:
            await self._compensate_failed_submission(
                task_id,
                repository=repository,
                task_created=task_created,
            )
            raise

        response = build_submission_response(
            task_id=task_id,
            filename=source_task.filename,
            file_size=file_size,
            email=email,
            vip=vip,
            message=message,
            extra={
                "source_task_id": source_task.id,
                **dict(response_extra or {}),
            },
        )
        logger.info(
            "Rerecognition task submitted: mode=%s task_id=%s source_task_id=%s",
            mode,
            task_id,
            source_task.id,
        )
        logger.debug(
            "Rerecognition submission details: %s",
            json_for_log({
                "mode": mode,
                "response": response,
                "archive_key": source_task.s3_key,
                "destination_path": str(destination),
                "file_hash": file_hash,
            }),
        )
        return response

    async def _restore_and_verify_audio(
        self,
        *,
        source_task: Any,
        task_id: str,
        mode: str,
        destination: Path,
    ) -> tuple[int, str]:
        restore = getattr(self.audio_backup_store, "restore_original", None)
        if not callable(restore):
            raise ArchiveRestoreError("归档存储不支持文件恢复")

        logger.info(
            "Rerecognition archive restore started: mode=%s backend=%s task_id=%s",
            mode,
            getattr(self.audio_backup_store, "name", "unknown"),
            task_id,
        )
        try:
            await restore(source_task.s3_key, str(destination), task_id)
            file_size = await asyncio.to_thread(lambda: destination.stat().st_size)
            max_file_size = self._max_file_size()
            if file_size > max_file_size:
                raise UploadTooLargeError(
                    f"文件过大: 超过限制 {max_file_size / 1024 / 1024:.0f} MB"
                )
            file_hash = await asyncio.to_thread(
                calculate_file_hash,
                str(destination),
            )
            expected_hash = str(getattr(source_task, "file_hash", "") or "")
            if expected_hash and not hmac.compare_digest(
                expected_hash.lower(),
                file_hash.lower(),
            ):
                raise ArchiveRestoreError("归档文件完整性校验失败")
            return file_size, file_hash
        except UploadTooLargeError:
            await asyncio.to_thread(self.temp_file_store.cleanup, task_id)
            raise
        except ArchiveRestoreError:
            await asyncio.to_thread(self.temp_file_store.cleanup, task_id)
            raise
        except Exception as exc:
            await asyncio.to_thread(self.temp_file_store.cleanup, task_id)
            log_exception(
                logger,
                logging.ERROR,
                "Rerecognition archive restore",
                exc,
                context={"mode": mode, "task_id": task_id},
            )
            raise ArchiveRestoreError("归档文件恢复失败") from exc

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
        task_id = new_task_id()
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
                logger.info(
                    "Task upload saved: task_id=%s size_bytes=%s",
                    task_id,
                    file_size,
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
    "ArchiveRestoreError",
    "RerecognitionConflictError",
    "RerecognitionSourceNotFoundError",
    "RerecognitionValidationError",
    "TaskSubmissionService",
    "TaskSubmissionUnavailableError",
    "UploadTooLargeError",
    "build_submission_response",
]
