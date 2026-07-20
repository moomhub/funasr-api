"""Concrete OFFLINE task result side effects."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

from src.application.task_results import (
    OfflineTaskContext,
    build_offline_persistence_payload,
)
from src.core.debug_logging import json_for_log, text_preview
from src.core.results.types import RecognitionResult
from src.task_queue.hook_contracts import BaseOfflineTaskResultHook

logger = logging.getLogger(__name__)


class ResultPersistenceHook(BaseOfflineTaskResultHook):
    name = "result_persistence"
    critical = True

    def __init__(self, task_repository: Any):
        self.task_repository = task_repository

    async def on_success(
        self,
        context: OfflineTaskContext,
        result: RecognitionResult,
    ) -> None:
        payload = build_offline_persistence_payload(context, result)
        summary = payload.summary
        logger.info("OFFLINE persistence summary: %s", json_for_log(summary))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "OFFLINE persistence details: %s",
                json_for_log({
                    **summary,
                    "filename": context.filename,
                    "audio_path": context.audio_path,
                    "text_preview": text_preview(payload.full_text),
                    "full_text": payload.full_text,
                    "segments": payload.segment_payloads,
                    "word_timestamps": payload.word_timestamps,
                    "metadata": result.metadata,
                }),
            )
        saved = await asyncio.to_thread(
            self.task_repository.save_result,
            task_id=context.task_id,
            full_text=payload.full_text,
            segments=payload.segment_payloads,
            processing_time=payload.processing_time,
            word_timestamps=payload.word_timestamps,
        )
        if saved is None:
            raise RuntimeError(
                f"任务不存在，无法保存识别结果: {context.task_id}"
            )

    async def on_failure(
        self,
        context: OfflineTaskContext,
        error_message: str,
    ) -> None:
        await asyncio.to_thread(
            self.task_repository.record_error,
            task_id=context.task_id,
            error_message=error_message,
            retry=True,
        )


class TextResultFileHook(BaseOfflineTaskResultHook):
    name = "text_result_file"
    critical = True

    def __init__(self, result_dir: str):
        self.result_dir = result_dir

    async def on_success(
        self,
        context: OfflineTaskContext,
        result: RecognitionResult,
    ) -> None:
        result_dir = Path(self.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{context.task_id}.txt"
        await asyncio.to_thread(
            result_path.write_text,
            result.full_text or "",
            encoding="utf-8",
        )
        context.metadata["text_result_path"] = str(result_path)
        logger.info("OFFLINE text result saved: task_id=%s", context.task_id)
        logger.debug(
            "OFFLINE text result path: task_id=%s path=%s",
            context.task_id,
            result_path,
        )


class AudioBackupHook(BaseOfflineTaskResultHook):
    name = "audio_backup"

    def __init__(self, task_repository: Any, postprocessor: Any):
        self.task_repository = task_repository
        self.postprocessor = postprocessor

    async def on_success(
        self,
        context: OfflineTaskContext,
        result: RecognitionResult,
    ) -> None:
        stored = await self.postprocessor.handle_complete(
            local_path=context.audio_path,
            task_key=context.task_id,
            filename=context.filename,
            source="offline",
            email=context.metadata.get("email"),
            delete_local_on_success=False,
        )
        if stored:
            context.metadata["backup_key"] = stored.s3_key
            await asyncio.to_thread(
                self.task_repository.record_file_info,
                context.task_id,
                s3_key=stored.s3_key,
                file_hash=stored.file_sha256,
            )


class TempCleanupHook(BaseOfflineTaskResultHook):
    name = "temp_cleanup"

    def __init__(self, temp_file_store: Any = None):
        self.temp_file_store = temp_file_store

    async def on_success(
        self,
        context: OfflineTaskContext,
        result: RecognitionResult,
    ) -> None:
        if not context.metadata.get("backup_key"):
            return None
        try:
            if self.temp_file_store is not None:
                await asyncio.to_thread(
                    self.temp_file_store.cleanup,
                    context.task_id,
                )
            else:
                await asyncio.to_thread(
                    self._cleanup_path,
                    Path(context.audio_path),
                )
            logger.info("OFFLINE temp file cleaned: task_id=%s", context.task_id)
            logger.debug(
                "OFFLINE temp cleanup details: task_id=%s path=%s",
                context.task_id,
                context.audio_path,
            )
        except Exception as exc:
            logger.warning(
                "OFFLINE temp cleanup failed: task_id=%s error_type=%s",
                context.task_id,
                type(exc).__name__,
            )
            logger.debug(
                "OFFLINE temp cleanup failure details: task_id=%s path=%s",
                context.task_id,
                context.audio_path,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    @staticmethod
    def _cleanup_path(path: Path) -> None:
        path.unlink(missing_ok=True)
        with suppress(OSError):
            path.parent.rmdir()


__all__ = [
    "AudioBackupHook",
    "ResultPersistenceHook",
    "TempCleanupHook",
    "TextResultFileHook",
]
