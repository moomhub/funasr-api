"""Application-owned dependency context exposed to transport adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.core.debug_logging import json_for_log

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    """All long-lived services owned by one FastAPI application instance."""

    config: Any
    container: Any
    hotword_manager: Any
    model_manager: Any
    runtime_services: Any
    runtime_application: Any
    online_service: Any
    speaker_service: Any
    task_submission_service: Any
    task_queue: Any

    @property
    def task_repository(self) -> Any:
        return self.container.task_repository

    @property
    def temp_file_store(self) -> Any:
        return self.container.temp_file_store

    @property
    def speaker_task_repository(self) -> Any:
        return self.container.speaker_task_repository

    @property
    def hotword_repository(self) -> Any:
        return self.container.hotword_repository

    def is_engine_enabled(self, mode: str) -> bool:
        return self.runtime_application.is_mode_available(mode)

    def is_task_submission_available(self, mode: str) -> bool:
        if mode not in {"offline", "spk"}:
            return False
        return bool(
            self.is_engine_enabled(mode)
            and self.task_submission_service.can_submit(mode)
        )

    def get_health_status(self) -> dict:
        processing_config = self.config.get_processing_config()
        runtime_status = self.runtime_application.get_runtime_status()
        module_status = self.container.get_status()
        logger.debug(
            "Health status details: %s",
            json_for_log({
                "runtime_services": runtime_status,
                "modules": module_status,
                "runtime_paths": self.config.get_runtime_paths(),
            }),
        )
        return {
            "status": "ok",
            "version": "4.0.0",
            "engines": self.runtime_application.get_engine_info(),
            "runtime_services": {
                name: self._status_summary(details)
                for name, details in runtime_status.items()
            },
            "modules": {
                name: self._status_summary(details)
                for name, details in module_status.items()
            },
            "storage_enabled": self.container.audio_backup_store.enabled,
            "models_loaded": self.runtime_application.get_loaded_models_count(),
            "inference_backends": self.runtime_application.get_inference_backends(),
            "task_queue_enabled": processing_config.offline_async_enabled,
            "task_queue_running": self.task_queue.is_running,
            "task_submission_available": {
                mode: self.is_task_submission_available(mode)
                for mode in ("offline", "spk")
            },
        }

    @staticmethod
    def _status_summary(details: Any) -> dict:
        if not isinstance(details, dict):
            return {"available": False, "has_error": True}
        summary = {
            key: details[key]
            for key in (
                "name",
                "type",
                "mode",
                "backend",
                "enabled",
                "loaded",
                "available",
            )
            if key in details
        }
        summary["has_error"] = bool(details.get("error") or details.get("last_error"))
        return summary


__all__ = ["AppServices"]
