"""Shared offline recognizer interface and execution template."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.debug_logging import log_exception
from src.core.results.types import RecognitionResult
from src.core.results.builders import build_error_recognition_result
from src.core.results.normalizers import normalize_recognition_result
from src.core.results import SpeakerResult
from src.engine_runtime.engines.spk.runner import StandaloneSpeakerRunner
from src.engine_runtime.engines.spk.base import SpeakerRecognitionRequest

logger = logging.getLogger(__name__)


@dataclass
class OfflineRecognitionRequest:
    audio_path: str
    hotwords: Optional[List[str]] = None
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseOfflineRecognizer(ABC):
    """Shared interface for offline recognizers across PT/ONNX backends."""

    mode = "offline"
    backend_name = "unknown"

    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self._speaker_runner = StandaloneSpeakerRunner(model_manager)

    @abstractmethod
    def load_model(self) -> Any:
        """Load or retrieve the backend-specific model object."""

    @abstractmethod
    async def run_inference(self, model: Any, request: OfflineRecognitionRequest) -> Any:
        """Execute backend-specific inference and return raw payload."""

    @abstractmethod
    def parse_result(self, payload: Any) -> RecognitionResult:
        """Normalize backend output into the shared recognition result."""

    async def _recognize_required_speaker(
        self,
        audio_path: str,
        generate_kwargs: Dict[str, Any],
        *,
        source: str,
    ) -> SpeakerResult:
        result = await self._speaker_runner.recognize(
            SpeakerRecognitionRequest(
                audio_path=audio_path,
                generate_kwargs=generate_kwargs or {},
            ),
            metadata={"source": source},
        )
        self._raise_if_speaker_unusable(result)
        return result

    @staticmethod
    def _raise_if_speaker_unusable(result: SpeakerResult) -> None:
        if result is None:
            raise RuntimeError("OFFLINE SPK 二次校验失败: 未返回结果")
        if result.error:
            raise RuntimeError(f"OFFLINE SPK 二次校验失败: {result.error}")
        if not result.segments:
            raise RuntimeError("OFFLINE SPK 二次校验失败: 未返回有效说话人分段")

    async def recognize(self, request: OfflineRecognitionRequest) -> RecognitionResult:
        start_time = datetime.now(timezone.utc)

        try:
            model = self.load_model()
            payload = await self.run_inference(model, request)
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
                "OFFLINE recognition",
                exc,
                context={"backend": self.backend_name, "audio_path": request.audio_path},
            )
            return build_error_recognition_result(
                mode=self.mode,
                backend_name=self.backend_name,
                exc=exc,
                start_time=start_time,
            )
