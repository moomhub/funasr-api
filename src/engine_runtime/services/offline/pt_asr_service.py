"""PT-backed offline ASR service."""

from __future__ import annotations

from src.core.results import RecognitionResult
from src.engine_runtime.engines.offline.base import OfflineRecognitionRequest
from src.engine_runtime.services.base import RuntimeServiceBase
from src.engine_runtime.services.contracts import ModelPreloadStatus, OfflineAsrRequest


class PTOfflineAsrService(RuntimeServiceBase):
    name = "offline_asr_pt"
    mode = "offline"
    backend = "pt"

    def preload(self) -> ModelPreloadStatus:
        return self._preload_with(lambda: self.manager.get_offline_pt_recognizer().load_model())

    async def recognize(self, request: OfflineAsrRequest) -> RecognitionResult:
        if not self._loaded:
            return RecognitionResult(mode="offline", error=self._not_loaded_error(), metadata=request.metadata)

        result = await self.manager.get_offline_pt_recognizer().recognize(
            OfflineRecognitionRequest(
                audio_path=request.audio_path,
                hotwords=request.hotwords,
                generate_kwargs=request.generate_kwargs,
            )
        )
        result.metadata.update(request.metadata)
        result.metadata["service"] = self.name
        return result
