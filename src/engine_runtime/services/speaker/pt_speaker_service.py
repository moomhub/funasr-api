"""PT-backed speaker service."""

from __future__ import annotations

from src.core.results import SpeakerResult
from src.engine_runtime.engines.spk.base import SpeakerRecognitionRequest
from src.engine_runtime.engines.spk.runner import StandaloneSpeakerRunner
from src.engine_runtime.services.base import RuntimeServiceBase
from src.engine_runtime.services.contracts import ModelPreloadStatus, SpeakerRequest


class PTSpeakerService(RuntimeServiceBase):
    name = "speaker_pt"
    mode = "spk"
    backend = "pt"

    def __init__(self, manager):
        super().__init__(manager)
        self._speaker_runner = StandaloneSpeakerRunner(manager)

    def preload(self) -> ModelPreloadStatus:
        return self._preload_with(lambda: self.manager.get_spk_pt_recognizer().load_model())

    async def diarize(self, request: SpeakerRequest) -> SpeakerResult:
        if not self._loaded:
            return SpeakerResult(error=self._not_loaded_error(), metadata=request.metadata)

        return await self._speaker_runner.recognize(
            SpeakerRecognitionRequest(
                audio_path=request.audio_path,
                generate_kwargs=request.generate_kwargs,
            ),
            metadata={
                **request.metadata,
                "service": self.name,
            },
        )
