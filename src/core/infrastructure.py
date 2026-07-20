"""Builders for concrete infrastructure adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.exc import InterfaceError, OperationalError

from src.core.infrastructure_adapters.audio_backup import (
    NoopAudioBackupStore,
    S3AudioBackupStore,
)
from src.core.infrastructure_adapters.hotwords import (
    DatabaseHotwordProvider,
    EmptyHotwordProvider,
)
from src.core.infrastructure_adapters.temp_files import LocalTempFileStore
from src.core.debug_logging import mask_url, log_exception
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
    database_config = config.get("database", {})
    enabled = as_bool(database_config.get("enabled"), True)
    if not enabled:
        logger.warning("Database disabled, using SQLite fallback repository")
        database_config = {}

    db = None
    db_url = None
    try:
        db_url = config.get_env("database.url", "DATABASE_URL", database_config.get("url"))
        mysql_config = (database_config or {}).get("mysql") or {}
        has_mysql_config = bool(mysql_config) or any(
            config.get_env(f"database.mysql.{key}", env_name, None) is not None
            for key, env_name in {
                "host": "DB_HOST",
                "username": "DB_USER",
                "password": "DB_PASSWORD",
                "database": "DB_NAME",
            }.items()
        )
        if not db_url:
            if has_mysql_config:
                db_config = config.get_database_config()
                mysql = db_config.mysql
                db_url = mysql.url
                pool_size = mysql.pool_size
                pool_recycle = mysql.pool_recycle
                echo = mysql.echo
            else:
                sqlite_path = config.get_runtime_paths()["sqlite_path"]
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
                db_url = f"sqlite:///{sqlite_path}"
                pool_size = int(config.get_env("database.sqlite.pool_size", "SQLITE_POOL_SIZE", 5))
                pool_recycle = int(config.get_env("database.sqlite.pool_recycle", "SQLITE_POOL_RECYCLE", 3600))
                echo = as_bool(config.get_env("database.sqlite.echo", "SQLITE_ECHO", False), False)
        else:
            db_config = config.get_database_config()
            pool_size = db_config.mysql.pool_size
            pool_recycle = db_config.mysql.pool_recycle
            echo = db_config.mysql.echo

        db = DatabaseManager(
            db_url,
            pool_size,
            pool_recycle,
            echo,
        )
        db.init_db()
        logger.info("Task repository initialized: backend=sql")
        logger.debug("Task repository connection URL: %s", mask_url(str(db_url)))
        return SqlTaskRepository(db)
    except (ImportError, InterfaceError, OperationalError) as exc:
        if db is not None:
            db.close()
        if db_url and str(db_url).startswith("sqlite"):
            raise
        return build_sqlite_fallback_repository(config, exc)


def build_sqlite_fallback_repository(config: Any, cause: Exception) -> SqlTaskRepository:
    log_exception(
        logger,
        logging.WARNING,
        "Configured SQL repository fallback",
        cause,
    )
    sqlite_path = config.get_runtime_paths()["sqlite_path"]
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    db = DatabaseManager(f"sqlite:///{sqlite_path}", 5, 3600, False)
    db.init_db()
    repository = SqlTaskRepository(db)
    repository.last_error = str(cause)
    return repository


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
    s3_config = config.get("storage.s3", {}) or {}
    enabled = as_bool(s3_config.get("enabled"), False) and as_bool(s3_config.get("save_original"), True)
    if not enabled:
        return NoopAudioBackupStore()

    endpoint = config.get_env("storage.s3.endpoint", "S3_ENDPOINT", s3_config.get("endpoint"))
    access_key = config.get_env("storage.s3.access_key", "S3_ACCESS_KEY", s3_config.get("access_key"))
    secret_key = config.get_env("storage.s3.secret_key", "S3_SECRET_KEY", s3_config.get("secret_key"))
    bucket = config.get_env("storage.s3.bucket", "S3_BUCKET", s3_config.get("bucket", "funasr-audio"))
    region = config.get_env("storage.s3.region", "S3_REGION", s3_config.get("region", "us-east-1"))
    prefix = config.get_env("storage.s3.prefix", "S3_PREFIX", s3_config.get("prefix", "audio"))

    if not all([endpoint, access_key, secret_key, bucket]):
        store = NoopAudioBackupStore()
        store.last_error = "S3 enabled but endpoint/access_key/secret_key/bucket is incomplete"
        logger.warning(store.last_error)
        return store

    try:
        return S3AudioBackupStore(endpoint, access_key, secret_key, bucket, region, str(prefix or ""))
    except Exception as exc:
        log_exception(logger, logging.WARNING, "S3 backup initialization", exc)
        store = NoopAudioBackupStore()
        store.last_error = str(exc)
        return store


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
    "build_sqlite_fallback_repository",
    "build_task_repository",
    "build_temp_file_store",
]
