"""Application-owned console and rotating-file logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.core.config.coercion import as_bool


DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_PATH = "./logs/funasr.log"
_OWNED_HANDLER_MARKER = "_funasr_owned_handler"
_logging_status: dict[str, dict[str, Any]] = {}


def initialize_logging(level_name: str = "INFO") -> None:
    """Install a temporary console handler until application config is loaded."""
    root_logger = logging.getLogger()
    if any(getattr(handler, _OWNED_HANDLER_MARKER, False) for handler in root_logger.handlers):
        return
    handler = logging.StreamHandler()
    setattr(handler, _OWNED_HANDLER_MARKER, True)
    handler.setLevel(_resolve_level(level_name))
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    root_logger.addHandler(handler)
    root_logger.setLevel(_resolve_level(level_name))


def configure_logging(config_or_level: Any = "INFO") -> dict[str, dict[str, Any]]:
    """Apply idempotent application logging and return a safe status summary."""
    if not hasattr(config_or_level, "get"):
        level = _resolve_level(str(config_or_level or "INFO"))
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return get_logging_status()

    config = config_or_level
    level_name = str(config.get("logging.level", "INFO") or "INFO")
    level = _resolve_level(level_name)
    log_format = str(config.get("logging.format", DEFAULT_LOG_FORMAT) or DEFAULT_LOG_FORMAT)
    formatter = logging.Formatter(log_format)
    console_enabled = as_bool(config.get("logging.console.enabled", True), True)
    file_enabled = as_bool(config.get("logging.file.enabled", True), True)
    root_logger = logging.getLogger()

    for handler in list(root_logger.handlers):
        if getattr(handler, _OWNED_HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()

    status: dict[str, dict[str, Any]] = {
        "console": {
            "provider": "console",
            "enabled": console_enabled,
            "status": "disabled",
        },
        "file": {
            "provider": "rotating_file",
            "enabled": file_enabled,
            "status": "disabled",
        },
    }

    if console_enabled:
        console_handler = logging.StreamHandler()
        _prepare_handler(console_handler, level, formatter)
        root_logger.addHandler(console_handler)
        status["console"]["status"] = "ready"

    if file_enabled:
        file_path = Path(str(config.get("logging.file.path", DEFAULT_LOG_PATH) or DEFAULT_LOG_PATH))
        max_size_mb = _positive_int(config.get("logging.file.max_size_mb", 100), 100)
        backup_count = _nonnegative_int(config.get("logging.file.backup_count", 10), 10)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        _prepare_handler(file_handler, level, formatter)
        root_logger.addHandler(file_handler)
        status["file"]["status"] = "ready"
        status["file"]["path"] = str(file_path)

    root_logger.setLevel(level)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(level)

    global _logging_status
    _logging_status = status
    logging.getLogger(__name__).info(
        "Logging configured: level=%s console=%s file=%s",
        level_name.upper(),
        status["console"]["status"],
        status["file"]["status"],
    )
    logging.getLogger(__name__).debug(
        "Logging file target: path=%s",
        status["file"].get("path"),
    )
    return get_logging_status()


def get_logging_status() -> dict[str, dict[str, Any]]:
    return {name: dict(details) for name, details in _logging_status.items()}


def _prepare_handler(
    handler: logging.Handler,
    level: int,
    formatter: logging.Formatter,
) -> None:
    setattr(handler, _OWNED_HANDLER_MARKER, True)
    handler.setLevel(level)
    handler.setFormatter(formatter)


def _resolve_level(level_name: str) -> int:
    return getattr(logging, str(level_name or "INFO").upper(), logging.INFO)


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


__all__ = [
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_LOG_PATH",
    "configure_logging",
    "get_logging_status",
    "initialize_logging",
]
