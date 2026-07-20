"""Hotword provider adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .status import component_status

logger = logging.getLogger(__name__)


class EmptyHotwordProvider:
    name = "empty"
    enabled = False
    available = True
    last_error = None

    def get_hotwords(self) -> List[Any]:
        return []

    def status(self) -> Dict[str, Any]:
        return component_status(self)


class DatabaseHotwordProvider:
    name = "database"
    enabled = True

    def __init__(self, repository: Any):
        self.repository = repository
        self.available = True
        self.last_error = None

    def get_hotwords(self) -> List[Any]:
        try:
            hotwords = self.repository.get_formatted_list()
            self.available = True
            self.last_error = None
            return hotwords
        except Exception as exc:
            self.available = False
            self.last_error = str(exc)
            logger.warning(
                "Database hotwords unavailable, using empty list: error_type=%s",
                type(exc).__name__,
            )
            logger.debug(
                "Database hotword failure details",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return []

    def status(self) -> Dict[str, Any]:
        return component_status(self)


__all__ = ["DatabaseHotwordProvider", "EmptyHotwordProvider"]
