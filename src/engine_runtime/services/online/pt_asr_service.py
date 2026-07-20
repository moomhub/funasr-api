"""PT-backed online ASR service."""

from __future__ import annotations

from typing import Any

from src.engine_runtime.engines.online.base import OnlineSessionRequest
from src.engine_runtime.services.base import RuntimeServiceBase
from src.engine_runtime.services.contracts import ModelPreloadStatus


class PTOnlineAsrService(RuntimeServiceBase):
    name = "online_asr_pt"
    mode = "online"
    backend = "pt"

    def preload(self) -> ModelPreloadStatus:
        return self._preload_with(lambda: self.manager.get_online_pt_model_bundle())

    def create_realtime_session(self, **kwargs: Any) -> Any:
        if not self._loaded:
            raise RuntimeError(self._not_loaded_error())
        return self.manager.get_online_pt_recognizer().create_session(
            OnlineSessionRequest(**kwargs)
        )
