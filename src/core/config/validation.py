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
}

REMOVED_ENV_VARS = {
    "ENGINES_MODEL_DIR": "RUNTIME_MODELS_DIR",
    "SQLITE_PATH": "RUNTIME_SQLITE_PATH",
    "PROCESSING_TEMP_DIR": "RUNTIME_TEMP_DIR",
    "OFFLINE_RESULT_DIR": "RUNTIME_OFFLINE_RESULT_DIR",
    "LOCAL_STORAGE_DIR": "RUNTIME_LOCAL_FILES_DIR",
}


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

    if violations:
        raise EngineConfigurationError("检测到已删除的旧配置：\n- " + "\n- ".join(violations))


def _contains_dotted_key(config: Mapping[str, Any], dotted_key: str) -> bool:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return True


__all__ = [
    "REMOVED_CONFIG_KEYS",
    "REMOVED_ENV_VARS",
    "reject_removed_configuration",
]
