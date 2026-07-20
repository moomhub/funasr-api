"""Application logging configuration."""

from __future__ import annotations

import logging


DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def initialize_logging(level_name: str = "INFO") -> None:
    """Install a default handler when the host process has not configured logging."""
    logging.basicConfig(
        level=_resolve_level(level_name),
        format=DEFAULT_LOG_FORMAT,
    )
    configure_logging(level_name)


def configure_logging(level_name: str = "INFO") -> None:
    """Apply one log level to the root logger and its existing handlers."""
    level = _resolve_level(level_name)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def _resolve_level(level_name: str) -> int:
    return getattr(logging, str(level_name or "INFO").upper(), logging.INFO)


__all__ = ["DEFAULT_LOG_FORMAT", "configure_logging", "initialize_logging"]
