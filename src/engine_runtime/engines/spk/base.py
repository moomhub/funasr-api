"""Shared speaker recognizer interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional



@dataclass
class SpeakerRecognitionRequest:
    audio_path: str
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseSpeakerRecognizer(ABC):
    """Shared interface for standalone SPK recognizers."""

    mode = "spk"
    backend_name = "unknown"

    def __init__(self, model_manager: Any):
        self.model_manager = model_manager

    @abstractmethod
    def load_model(self) -> Any:
        """Load or retrieve the backend-specific speaker model."""

    async def recognize(self, request: SpeakerRecognitionRequest) -> Optional[Any]:
        model = self.load_model()
        generate = getattr(model, "generate", None)
        if generate:
            return await asyncio.to_thread(
                generate,
                input=request.audio_path,
                **request.generate_kwargs,
            )
        if callable(model):
            return await asyncio.to_thread(
                model,
                request.audio_path,
                **request.generate_kwargs,
            )
        raise TypeError(f"SPK {self.backend_name} model is not callable")
