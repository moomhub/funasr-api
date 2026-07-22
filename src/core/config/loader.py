"""YAML configuration loader with environment overrides."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .builders import TypedConfigBuilder
from .coercion import as_bool, as_list
from .paths import RuntimePathResolver
from .schema import (
    DatabaseConfig,
    EnginesConfig,
    HotwordConfig,
    ProcessingConfig,
    StorageConfig,
)
from .source import env_or_value, load_yaml
from .validation import (
    REMOVED_CONFIG_KEYS,
    REMOVED_ENV_VARS,
    reject_removed_configuration,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load raw YAML values and expose stable typed configuration accessors."""

    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path or "./config.yaml")
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._reject_removed_configuration()

    def _load_config(self) -> None:
        self._config = load_yaml(self.config_path, logger)

    def _reject_removed_configuration(self) -> None:
        reject_removed_configuration(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._config
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(part)
            if value is None:
                return default
        return value

    def get_env(
        self,
        key: str,
        env_var: str | None = None,
        default: Any = None,
    ) -> Any:
        effective_env_var = env_var or f"APP_{key.upper().replace('.', '_')}"
        return env_or_value(effective_env_var, self.get(key, default), logger)

    def get_runtime_paths(self) -> Dict[str, str]:
        return RuntimePathResolver(self._config, self.get_env).all_paths()

    def get_database_config(self) -> DatabaseConfig:
        return self._typed_builder().database()

    def get_storage_config(self) -> StorageConfig:
        return self._typed_builder().storage()

    def get_engines_config(self) -> EnginesConfig:
        return self._typed_builder().engines()

    def get_processing_config(self) -> ProcessingConfig:
        return self._typed_builder().processing()

    def get_hotword_config(self) -> HotwordConfig:
        return self._typed_builder().hotwords()

    @property
    def config_dict(self) -> Dict[str, Any]:
        return self._config.copy()

    def _typed_builder(self) -> TypedConfigBuilder:
        return TypedConfigBuilder(
            self._config,
            self.get_env,
            self.get_runtime_paths,
        )

    def _as_list(self, value: Any, default: list | None = None) -> list:
        return as_list(value, default)

    def _as_bool(self, value: Any, default: bool = False) -> bool:
        return as_bool(value, default)


__all__ = [
    "ConfigLoader",
    "REMOVED_CONFIG_KEYS",
    "REMOVED_ENV_VARS",
]
