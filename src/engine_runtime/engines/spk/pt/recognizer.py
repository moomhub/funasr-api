"""PT-backed standalone speaker recognizer."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from src.engine_runtime.engines.spk.base import BaseSpeakerRecognizer, SpeakerRecognitionRequest


class PTSpeakerRecognizer(BaseSpeakerRecognizer):
    backend_name = "pt"
    sample_rate = 16000

    def __init__(self, model_manager: Any):
        super().__init__(model_manager)
        self._inference_lock = threading.Lock()

    def load_model(self) -> Any:
        return self.model_manager.get_spk_model()

    async def recognize(self, request: SpeakerRecognitionRequest) -> Any:
        model = self.load_model()
        kwargs = dict(request.generate_kwargs or {})
        sample_rate = int(kwargs.pop("sample_rate", self.sample_rate))

        # Load one normalized waveform before entering the locked model pipeline.
        audio_data = await asyncio.to_thread(
            self._load_audio,
            request.audio_path,
            sample_rate,
        )
        return await asyncio.to_thread(self._run_model, model, audio_data, kwargs)

    def _run_model(self, model: Any, audio_data: Any, kwargs: dict) -> Any:
        with self._inference_lock:
            return model(audio_data, **kwargs)

    @staticmethod
    def _load_audio(audio_path: str, sample_rate: int):
        import librosa

        audio_data, _ = librosa.load(audio_path, sr=sample_rate)
        return audio_data
