"""Builders for concrete infrastructure adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.infrastructure_adapters.audio_backup import (
    LocalAudioBackupStore,
    S3AudioBackupStore,
)
from src.core.infrastructure_adapters.hotwords import (
    DatabaseHotwordProvider,
    EmptyHotwordProvider,
)
from src.core.infrastructure_adapters.temp_files import LocalTempFileStore
from src.core.debug_logging import mask_url
from src.database.repositories import (
    SqlFileIndexRepository,
    SqlHotwordRepository,
    SqlSpeakerTaskRepository,
    SqlTaskRepository,
)
from src.database.session import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepositoryBundle:
    task_repository: Any
    speaker_task_repository: Any
    file_index_repository: Any
    hotword_repository: Any


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def build_task_repository(config: Any) -> Any:
    database_config = config.get_database_config()
    database_type = database_config.type
    if database_type == "sqlite":
        sqlite = database_config.sqlite
        Path(sqlite.path).parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{sqlite.path}"
        pool_size = sqlite.pool_size
        pool_recycle = sqlite.pool_recycle
        echo = sqlite.echo
    elif database_type == "mysql":
        mysql = database_config.mysql
        db_url = mysql.url
        pool_size = mysql.pool_size
        pool_recycle = mysql.pool_recycle
        echo = mysql.echo
    else:
        raise ValueError(f"Unsupported database type: {database_type}")

    db = DatabaseManager(db_url, pool_size, pool_recycle, echo)
    logger.info("Task repository configured: backend=%s", database_type)
    logger.debug("Task repository connection URL: %s", mask_url(str(db_url)))
    return SqlTaskRepository(db)


def build_repository_bundle(task_repository: Any) -> RepositoryBundle:
    db_manager = task_repository.db
    return RepositoryBundle(
        task_repository=task_repository,
        speaker_task_repository=SqlSpeakerTaskRepository(db_manager),
        file_index_repository=SqlFileIndexRepository(db_manager),
        hotword_repository=SqlHotwordRepository(db_manager),
    )


def build_temp_file_store(config: Any) -> Any:
    processing_config = config.get_processing_config()
    return LocalTempFileStore(processing_config.temp_dir)


def build_audio_backup_store(config: Any) -> Any:
    storage_config = config.get_storage_config()
    if storage_config.type == "local":
        local = storage_config.local
        return LocalAudioBackupStore(local.root, local.prefix)
    if storage_config.type == "s3":
        s3 = storage_config.s3
        return S3AudioBackupStore(
            s3.endpoint,
            s3.access_key,
            s3.secret_key,
            s3.bucket,
            s3.region,
            s3.prefix,
        )
    raise ValueError(f"Unsupported storage type: {storage_config.type}")


def build_hotword_provider(config: Any, task_repository: Any, hotword_repository: Any = None) -> Any:
    hotword_config = config.get_hotword_config()
    if not as_bool(hotword_config.enabled, True):
        return EmptyHotwordProvider()
    if hotword_config.source == "database" and task_repository.name == "sql":
        return DatabaseHotwordProvider(hotword_repository)
    return EmptyHotwordProvider()


__all__ = [
    "RepositoryBundle",
    "as_bool",
    "build_audio_backup_store",
    "build_hotword_provider",
    "build_repository_bundle",
    "build_task_repository",
    "build_temp_file_store",
]
