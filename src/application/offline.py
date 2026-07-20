"""Application services for offline recognition workflows."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.debug_logging import json_for_log, log_exception
from src.application.task_flow import load_task_file_ref, mark_task_processing, require_audio_file
from src.application.task_results import OfflineTaskContext
from src.core.hotwords import resolve_hotwords_with_priority
from src.core.results import RecognitionResult, build_error_recognition_result, normalize_recognition_result
from src.engine_runtime.services.contracts import OfflineAsrRequest, OfflineAsrService

logger = logging.getLogger(__name__)


def _log_loaded_hotwords(hotwords: Any, count: int) -> None:
    if not count:
        return
    logger.info(
        "🔑 加载热词: %s 个，内容: %s",
        count,
        json_for_log(hotwords),
    )


class OfflineRecognitionService:
    """Application service delegating OFFLINE ASR to one runtime facade."""

    def __init__(self, offline_service: OfflineAsrService):
        self.offline_service = offline_service

    async def recognize(
        self,
        audio_path: str,
        *,
        hotwords: Optional[List[str]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RecognitionResult:
        start_time = datetime.now(timezone.utc)
        metadata = dict(metadata or {})
        try:
            if not self.offline_service.is_loaded:
                status = self.offline_service.preload()
                if not status.loaded:
                    raise RuntimeError(status.error or "offline runtime preload failed")
            result = await self.offline_service.recognize(
                OfflineAsrRequest(
                    audio_path=audio_path,
                    hotwords=hotwords,
                    generate_kwargs=generate_kwargs or {},
                    metadata=metadata,
                )
            )
        except Exception as exc:
            return build_error_recognition_result(
                mode="offline",
                backend_name=self.offline_service.backend,
                exc=exc,
                start_time=start_time,
                operation="识别",
            )

        result.metadata["service"] = "offline_asr"
        return normalize_recognition_result(
            result,
            mode="offline",
            is_final=not bool(result.error),
            start_time=start_time,
        )


class OfflineTaskService:
    """Task-oriented offline application flow used by the task queue."""

    def __init__(
        self,
        *,
        task_repository: Any = None,
        temp_file_store: Any = None,
        result_handler: Any = None,
        recognition_service: OfflineRecognitionService | None = None,
        config: Any = None,
        hotword_lookup: Any = None,
    ):
        if task_repository is None or temp_file_store is None or result_handler is None:
            raise ValueError("task_repository, temp_file_store and result_handler are required")
        if config is None:
            raise ValueError("config is required")
        self.task_repository = task_repository
        self.temp_file_store = temp_file_store
        self.result_handler = result_handler
        if recognition_service is None:
            raise ValueError("recognition_service is required")
        self.recognition_service = recognition_service
        self.config = config
        self.hotword_lookup = hotword_lookup

    async def process_task(self, task_id: str) -> bool:
        task_ref = await load_task_file_ref(
            repository=self.task_repository,
            temp_file_store=self.temp_file_store,
            task_id=task_id,
            logger=logger,
            missing_message="❌ 任务不存在: %s",
        )
        if task_ref is None:
            return False

        task = task_ref.task
        filename = task_ref.filename
        audio_path = str(task_ref.audio_path)

        try:
            logger.info("🔄 开始处理任务: %s", task_id)
            logger.debug(
                "OFFLINE 任务输入: task_id=%s filename=%s audio_path=%s",
                task_id,
                filename,
                audio_path,
            )
            await mark_task_processing(self.task_repository, task_id)

            require_audio_file(task_ref.audio_path)

            hotwords = await asyncio.to_thread(self._load_hotwords, task)
            context = OfflineTaskContext(
                task_id=task_id,
                filename=filename,
                audio_path=audio_path,
                metadata={"email": task.email, "vip": task.vip},
            )
            result = await self.recognition_service.recognize(
                audio_path,
                hotwords=hotwords,
                metadata={"task_id": task_id, "filename": filename},
            )
            if result.error:
                raise RuntimeError(result.error)

            await self.result_handler.handle_success(context, result)
            logger.info("✅ 任务处理完成: %s", task_id)
            return True
        except Exception as exc:
            log_exception(
                logger,
                logging.ERROR,
                "OFFLINE task processing",
                exc,
                context={"task_id": task_id, "filename": filename, "audio_path": audio_path},
            )
            await self.result_handler.handle_failure(
                OfflineTaskContext(
                    task_id=task_id,
                    filename=filename,
                    audio_path=audio_path,
                ),
                str(exc),
            )
            return False

    def _load_hotwords(self, task: Any) -> List[Any]:
        default_hotword_ids = self.config.get("hotwords", {}).get("default_ids", [])
        if isinstance(task.hotwords, list):
            hotwords = list(task.hotwords)
            _log_loaded_hotwords(hotwords, len(hotwords))
            return hotwords
        resolved = resolve_hotwords_with_priority(
            custom_hotwords=task.hotwords,
            hotword_id=task.hotword_id,
            default_hotword_ids=default_hotword_ids,
            config=self.config,
            hotword_lookup=self.hotword_lookup,
        )
        _log_loaded_hotwords(resolved.value, resolved.count)
        return resolved.value


__all__ = [
    "OfflineRecognitionService",
    "OfflineTaskService",
]
