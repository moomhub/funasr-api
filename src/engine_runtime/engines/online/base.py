"""Shared online recognizer interface and helper utilities."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from src.core.debug_logging import log_exception
from src.core.results.types import RecognitionResult
from src.core.results.builders import build_error_recognition_result, build_recognition_result
from src.core.results.normalizers import normalize_recognition_result
from src.core.text import clean_online_asr_text, extract_online_text, merge_online_partial_text

logger = logging.getLogger(__name__)


@dataclass
class OnlineRecognitionRequest:
    audio_path: str
    hotwords: Optional[List[Any]] = None
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineSessionRequest:
    hotwords: Optional[List[Any]] = None
    sample_rate: int = 16000
    decode_interval: float = 0.48
    vad_merge_gap_ms: int = 1200
    vad_min_final_ms: int = 2500
    vad_max_final_ms: int = 12000


class SupportsGenerate(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        ...


class SupportsRealtimeVad(Protocol):
    current_speech_start: Optional[int]

    def reset(self) -> None:
        ...

    def feed(self, audio: Any, is_final: bool = False) -> List[List[float]]:
        ...


@dataclass
class OnlineModelBundle:
    """Models required by the ONLINE realtime two-pass flow."""

    streaming_asr: SupportsGenerate
    vad: SupportsRealtimeVad
    final_asr: SupportsGenerate
    punc: Optional[SupportsGenerate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "streaming_asr": self.streaming_asr,
            "vad": self.vad,
            "final_asr": self.final_asr,
            "punc": self.punc,
            "metadata": dict(self.metadata),
        }


@dataclass
class OnlinePTModelBundle(OnlineModelBundle):
    """PyTorch ONLINE 2pass model group."""


@dataclass
class OnlineONNXModelBundle(OnlineModelBundle):
    """ONNX ONLINE 2pass model group."""


class BaseOnlineRecognizer(ABC):
    """Shared interface for online recognizers across PT/ONNX backends."""

    mode = "online"
    backend_name = "unknown"

    def __init__(self, model_manager: Any):
        self.model_manager = model_manager

    @abstractmethod
    def load_models(self) -> Any:
        """Load or retrieve backend-specific online model bundle."""

    @abstractmethod
    def create_session(self, request: OnlineSessionRequest) -> Any:
        """Create a realtime session for WebSocket streaming."""

    @abstractmethod
    async def run_inference(self, models: Any, request: OnlineRecognitionRequest) -> Any:
        """Execute backend-specific online inference and return raw payload."""

    def parse_result(self, payload: Any) -> RecognitionResult:
        result = build_recognition_result(self.mode)
        result.full_text = extract_online_text(payload)
        return normalize_recognition_result(result, mode=self.mode)

    async def recognize(self, request: OnlineRecognitionRequest) -> RecognitionResult:
        start_time = datetime.now(timezone.utc)

        try:
            models = self.load_models()
            payload = await self.run_inference(models, request)
            result = self.parse_result(payload)
            return normalize_recognition_result(
                result,
                mode=self.mode,
                is_final=True,
                start_time=start_time,
            )
        except Exception as exc:
            log_exception(
                logger,
                logging.ERROR,
                "ONLINE recognition",
                exc,
                context={"backend": self.backend_name, "audio_path": request.audio_path},
            )
            return build_error_recognition_result(
                mode=self.mode,
                backend_name=self.backend_name,
                exc=exc,
                start_time=start_time,
            )
