"""Shared helpers for concrete runtime services."""

from __future__ import annotations

import time
from typing import Callable, Optional

from .contracts import ModelPreloadStatus, ServiceHealth


class RuntimeServiceBase:
    """Base state for services that preload a concrete runtime model."""

    name = "runtime_service"
    mode = "unknown"
    backend = "unknown"

    def __init__(self, manager):
        self.manager = manager
        self._loaded = False
        self._last_error: Optional[str] = None
        self._preload_seconds = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preload_with(self, loader: Callable[[], object]) -> ModelPreloadStatus:
        start = time.perf_counter()
        try:
            loader()
            self._loaded = True
            self._last_error = None
        except Exception as exc:
            self._loaded = False
            self._last_error = str(exc)
        self._preload_seconds = time.perf_counter() - start
        return ModelPreloadStatus(
            service_name=self.name,
            mode=self.mode,
            backend=self.backend,
            loaded=self._loaded,
            error=self._last_error,
            elapsed_seconds=self._preload_seconds,
        )

    def _not_loaded_error(self) -> str:
        return f"{self.name} model is not preloaded"

    def health(self) -> ServiceHealth:
        return ServiceHealth(
            name=self.name,
            mode=self.mode,
            backend=self.backend,
            loaded=self._loaded,
            available=self._loaded and self._last_error is None,
            error=self._last_error,
            metadata={"preload_seconds": self._preload_seconds},
        )
