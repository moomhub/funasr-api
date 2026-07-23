"""Application services for standalone speaker diarization."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from src.core.debug_logging import log_exception
from src.application.task_flow import (
    load_task_file_ref,
    mark_task_processing,
    record_task_error,
    require_audio_file,
)
from src.core.results import SpeakerResult
from src.engine_runtime.services.contracts import SpeakerRequest, SpeakerService

logger = logging.getLogger(__name__)


class SpkAsrService:
    """Application entrypoint for speaker diarization."""

    def __init__(self, speaker_service: SpeakerService = None):
        if speaker_service is None:
            raise ValueError("speaker_service is required")
        self.speaker_service = speaker_service

    def preload(self):
        return self.speaker_service.preload()

    def health(self):
        return self.speaker_service.health()

    async def diarize(
        self,
        audio_path: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> SpeakerResult:
        if not self.speaker_service.is_loaded:
            self.speaker_service.preload()
        return await self.speaker_service.diarize(
            SpeakerRequest(
                audio_path=audio_path,
                generate_kwargs=generate_kwargs or {},
                metadata=dict(metadata or {}),
            )
        )


class SpkTaskService:
    """Task-oriented standalone speaker diarization flow used by the queue."""

    def __init__(
        self,
        *,
        speaker_task_repository: Any = None,
        temp_file_store: Any = None,
        speaker_service: SpkAsrService | None = None,
        postprocessor: Any = None,
    ):
        if speaker_task_repository is None or temp_file_store is None:
            raise ValueError("speaker_task_repository and temp_file_store are required")
        if speaker_service is None:
            raise ValueError("speaker_service is required")
        if postprocessor is None:
            raise ValueError("postprocessor is required")
        self.speaker_task_repository = speaker_task_repository
        self.temp_file_store = temp_file_store
        self.speaker_service = speaker_service
        self.postprocessor = postprocessor

    async def process_task(self, task_id: str) -> bool:
        task_ref = await load_task_file_ref(
            repository=self.speaker_task_repository,
            temp_file_store=self.temp_file_store,
            task_id=task_id,
            logger=logger,
            missing_message="❌ SPK 任务不存在: %s",
        )
        if task_ref is None:
            return False

        task = task_ref.task
        audio_path = task_ref.audio_path
        start = time.perf_counter()
        try:
            require_audio_file(audio_path)

            await mark_task_processing(self.speaker_task_repository, task_id)
            result = await self.speaker_service.diarize(
                str(audio_path),
                metadata={"task_id": task_id, "filename": task_ref.filename},
            )
            if result is None or result.error:
                raise RuntimeError(result.error or "SPK 识别失败")

            stored = await self.postprocessor.handle_complete(
                local_path=str(audio_path),
                task_key=task_id,
                filename=task_ref.filename,
                source="spk",
                email=task.email,
                delete_local_on_success=False,
            )
            await asyncio.to_thread(
                self.speaker_task_repository.save_result,
                task_id=task_id,
                result=result.to_dict(),
                processing_time=time.perf_counter() - start,
                s3_key=stored.s3_key if stored else None,
                file_hash=stored.file_sha256 if stored else None,
            )
            if stored:
                await asyncio.to_thread(self.temp_file_store.cleanup, task_id)
            logger.info("✅ SPK 任务处理完成: %s", task_id)
            return True
        except Exception as exc:
            log_exception(
                logger,
                logging.ERROR,
                "SPK task processing",
                exc,
                context={"task_id": task_id, "audio_path": str(audio_path)},
            )
            await record_task_error(
                self.speaker_task_repository,
                task_id,
                str(exc),
                retry=False,
            )
            return False


__all__ = ["SpkAsrService", "SpkTaskService"]
