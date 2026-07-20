"""Service assembly for optional infrastructure adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from src.core.debug_logging import json_for_log

from src.core.infrastructure import (
    build_audio_backup_store,
    build_hotword_provider,
    build_repository_bundle,
    build_task_repository,
    build_temp_file_store,
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    config: Any
    task_repository: Any
    temp_file_store: Any
    audio_backup_store: Any
    hotword_provider: Any
    speaker_task_repository: Any = None
    file_index_repository: Any = None
    hotword_repository: Any = None

    def get_status(self) -> Dict[str, Any]:
        return {
            "task_repository": self.task_repository.status(),
            "temp_file_store": self.temp_file_store.status(),
            "audio_backup_store": self.audio_backup_store.status(),
            "hotword_provider": self.hotword_provider.status(),
        }

    def shutdown(self) -> None:
        try:
            self.task_repository.close()
        except Exception as exc:
            logger.warning(
                "Component shutdown failed: error_type=%s",
                type(exc).__name__,
            )
            logger.debug(
                "Component shutdown failure details",
                exc_info=(type(exc), exc, exc.__traceback__),
            )


def build_container(config: Any) -> ServiceContainer:
    """Build a container without registering module-level state."""
    task_repository = build_task_repository(config)
    repositories = build_repository_bundle(task_repository)
    container = ServiceContainer(
        config=config,
        task_repository=repositories.task_repository,
        temp_file_store=build_temp_file_store(config),
        audio_backup_store=build_audio_backup_store(config),
        hotword_provider=build_hotword_provider(
            config,
            repositories.task_repository,
            repositories.hotword_repository,
        ),
        speaker_task_repository=repositories.speaker_task_repository,
        file_index_repository=repositories.file_index_repository,
        hotword_repository=repositories.hotword_repository,
    )
    status = container.get_status()
    summary = {
        component: {
            "name": details.get("name"),
            "enabled": details.get("enabled"),
            "available": details.get("available"),
        }
        for component, details in status.items()
    }
    logger.info("Service container initialized: components=%s", summary)
    logger.debug("Service container status details: %s", json_for_log(status))
    return container

