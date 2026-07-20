"""YAML and environment configuration sources."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.core.config.errors import EngineConfigurationError
from src.core.debug_logging import is_sensitive_key


def load_yaml(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    if not path.exists():
        logger.warning("配置文件不存在，使用默认配置")
        logger.debug("缺失的配置文件路径: %s", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    except Exception as exc:
        logger.error("配置文件加载失败: error_type=%s", type(exc).__name__)
        logger.debug(
            "配置文件加载失败详情: path=%s",
            path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        raise EngineConfigurationError("配置文件加载失败") from exc

    if loaded is None:
        config: Dict[str, Any] = {}
    elif isinstance(loaded, dict):
        config = loaded
    else:
        logger.error("配置文件结构非法: root_type=%s", type(loaded).__name__)
        logger.debug("配置文件结构非法详情: path=%s root_value=%r", path, loaded)
        raise EngineConfigurationError("配置文件根节点必须是键值映射")

    logger.info("配置文件加载成功")
    logger.debug("配置文件加载详情: path=%s keys=%s", path, sorted(config))
    return config


def env_or_value(env_var: str, value: Any, logger: logging.Logger) -> Any:
    env_value = os.getenv(env_var)
    if env_value is not None:
        display_value = "***" if is_sensitive_key(env_var) else env_value
        logger.debug("从环境变量读取配置: %s = %s", env_var, display_value)
        return env_value
    return value


__all__ = ["env_or_value", "load_yaml"]
