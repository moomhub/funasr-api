"""Shared orchestration for standalone speaker recognition."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.debug_logging import json_for_log, log_exception
from src.core.results import SpeakerResult

from .base import SpeakerRecognitionRequest
from .normalizers import normalize_speaker_result

logger = logging.getLogger(__name__)


class StandaloneSpeakerRunner:
    """Resolve the shared recognizer, execute it, and normalize its payload."""

    def __init__(self, model_manager: Any):
        self.model_manager = model_manager

    async def recognize(
        self,
        request: SpeakerRecognitionRequest,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpeakerResult:
        logger.debug(
            "Standalone SPK input: %s",
            json_for_log({
                "audio_path": request.audio_path,
                "generate_kwargs": request.generate_kwargs,
                "metadata": metadata,
            }),
        )
        try:
            recognizer = self.model_manager.get_spk_recognizer()
            raw_payload = await recognizer.recognize(request)
            result = normalize_speaker_result(raw_payload)
            logger.debug(
                "Standalone SPK output: %s",
                json_for_log({
                    "raw_payload": raw_payload,
                    "normalized_result": result.to_dict(),
                }),
            )
        except Exception as exc:
            log_exception(
                logger,
                logging.ERROR,
                "Standalone SPK recognition",
                exc,
                context={"audio_path": request.audio_path, "metadata": metadata},
            )
            result = SpeakerResult(
                error=str(exc),
                metadata={"raw_payload": None},
            )

        if metadata:
            result.metadata.update(metadata)
        logger.debug(
            "Standalone SPK final result: %s",
            json_for_log(result.to_dict()),
        )
        return result


__all__ = ["StandaloneSpeakerRunner"]
