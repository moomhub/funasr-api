"""Application services for online recognition workflows."""

from __future__ import annotations

from typing import Any

from src.engine_runtime.services.contracts import OnlineAsrService as OnlineRuntimeService

class OnlineAsrService:
    """Application entrypoint for realtime online ASR sessions."""

    def __init__(self, online_service: OnlineRuntimeService = None):
        if online_service is None:
            raise ValueError("online_service is required")
        self.online_service = online_service

    def preload(self):
        return self.online_service.preload()

    def health(self):
        return self.online_service.health()

    def create_realtime_session(self, **kwargs: Any) -> Any:
        if not self.online_service.is_loaded:
            self.online_service.preload()
        return self.online_service.create_realtime_session(**kwargs)


__all__ = ["OnlineAsrService"]
