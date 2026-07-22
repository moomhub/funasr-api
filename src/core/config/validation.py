"""Validation for configuration keys that are no longer supported."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .errors import EngineConfigurationError


REMOVED_CONFIG_KEYS = {
    "engines.model_dir": "runtime.models_dir",
    "database.sqlite.path": "runtime.sqlite_path",
    "processing.temp_dir": "runtime.temp_dir",
    "results.offline.dir": "runtime.offline_result_dir",
    "storage.local.dir": "runtime.local_files_dir",
    "messaging": "notifications",
    "engines.models.spk.pt": "engines.models.spk.spk",
    "engines.models.spk.onnx": "engines.models.spk.spk",
    "engines.models.spk.enabled": "engines.models.spk.spk",
    "database.url": "database.type + database.mysql/database.sqlite",
    "database.enabled": "database.type",
    "storage.s3.enabled": "storage.type",
    "storage.s3.save_original": "storage.type",
}

REMOVED_ENV_VARS = {
    "ENGINES_MODEL_DIR": "RUNTIME_MODELS_DIR",
    "SQLITE_PATH": "RUNTIME_SQLITE_PATH",
    "PROCESSING_TEMP_DIR": "RUNTIME_TEMP_DIR",
    "OFFLINE_RESULT_DIR": "RUNTIME_OFFLINE_RESULT_DIR",
    "LOCAL_STORAGE_DIR": "RUNTIME_LOCAL_FILES_DIR",
    "DATABASE_URL": "DATABASE_TYPE + DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME",
}

DATABASE_TYPES = {"sqlite", "mysql"}
STORAGE_TYPES = {"local", "s3"}


def reject_removed_configuration(
    config: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> None:
    violations = []
    for old_key, replacement in REMOVED_CONFIG_KEYS.items():
        if _contains_dotted_key(config, old_key):
            violations.append(f"配置键 {old_key} 已删除，请改用 {replacement}")

    environment = os.environ if environ is None else environ
    for old_env, replacement in REMOVED_ENV_VARS.items():
        if old_env in environment:
            violations.append(f"环境变量 {old_env} 已删除，请改用 {replacement}")

    database = config.get("database") if isinstance(config, Mapping) else None
    database = database if isinstance(database, Mapping) else {}
    database_type = str(
        environment.get("DATABASE_TYPE", database.get("type", "sqlite")) or "sqlite"
    ).strip().lower()
    if database_type not in DATABASE_TYPES:
        violations.append(
            "配置键 database.type 仅支持 sqlite/mysql，"
            f"当前值为 {database_type!r}"
        )
    elif database_type == "mysql":
        mysql = database.get("mysql")
        mysql = mysql if isinstance(mysql, Mapping) else {}
        required_mysql = {
            "host": environment.get("DB_HOST", mysql.get("host")),
            "username": environment.get("DB_USER", mysql.get("username")),
            "database": environment.get("DB_NAME", mysql.get("database")),
        }
        missing_mysql = [key for key, value in required_mysql.items() if not str(value or "").strip()]
        if missing_mysql:
            violations.append(
                "database.type=mysql 时缺少必需配置：database.mysql."
                + ", database.mysql.".join(missing_mysql)
            )

    storage = config.get("storage") if isinstance(config, Mapping) else None
    storage = storage if isinstance(storage, Mapping) else {}
    storage_type = str(
        environment.get("STORAGE_TYPE", storage.get("type", "local")) or "local"
    ).strip().lower()
    if storage_type not in STORAGE_TYPES:
        violations.append(
            "配置键 storage.type 仅支持 local/s3，"
            f"当前值为 {storage_type!r}"
        )
    elif storage_type == "s3":
        s3 = storage.get("s3")
        s3 = s3 if isinstance(s3, Mapping) else {}
        required_s3 = {
            "endpoint": environment.get("S3_ENDPOINT", s3.get("endpoint")),
            "access_key": environment.get("S3_ACCESS_KEY", s3.get("access_key")),
            "secret_key": environment.get("S3_SECRET_KEY", s3.get("secret_key")),
            "bucket": environment.get("S3_BUCKET", s3.get("bucket")),
        }
        missing_s3 = [key for key, value in required_s3.items() if not str(value or "").strip()]
        if missing_s3:
            violations.append(
                "storage.type=s3 时缺少必需配置：storage.s3."
                + ", storage.s3.".join(missing_s3)
            )

    if "type" not in database and "DATABASE_TYPE" not in environment:
        mysql = database.get("mysql")
        mysql = mysql if isinstance(mysql, Mapping) else {}
        if any(mysql.get(key) for key in ("host", "username", "password", "database")) or any(
            key in environment for key in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
        ):
            violations.append(
                "检测到 MySQL 参数但未配置 database.type；请显式设置 database.type: mysql"
            )

    if "type" not in storage and "STORAGE_TYPE" not in environment:
        s3 = storage.get("s3")
        s3 = s3 if isinstance(s3, Mapping) else {}
        if any(s3.get(key) for key in ("endpoint", "access_key", "secret_key")) or any(
            key in environment for key in ("S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY")
        ):
            violations.append(
                "检测到 S3 参数但未配置 storage.type；请显式设置 storage.type: s3"
            )

    notifications = config.get("notifications") if isinstance(config, Mapping) else None
    notifications = notifications if isinstance(notifications, Mapping) else {}
    if notifications.get("enabled") and str(notifications.get("type", "")).strip().lower() != "rocketmq":
        violations.append("notifications.enabled=true 时 notifications.type 仅支持 rocketmq")

    if violations:
        raise EngineConfigurationError("配置校验失败：\n- " + "\n- ".join(violations))


def _contains_dotted_key(config: Mapping[str, Any], dotted_key: str) -> bool:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return True


__all__ = [
    "DATABASE_TYPES",
    "REMOVED_CONFIG_KEYS",
    "REMOVED_ENV_VARS",
    "STORAGE_TYPES",
    "reject_removed_configuration",
]
