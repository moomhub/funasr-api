"""Recognition result builders."""

from datetime import datetime
from typing import Optional

from .types import RecognitionResult
from .normalizers import normalize_recognition_result


def build_recognition_result(
    mode: str,
    *,
    is_final: bool = False,
    stage: Optional[str] = None,
) -> RecognitionResult:
    """Create a fully initialized result object for all runtime backends."""
    return RecognitionResult(
        mode=mode,
        segments=[],
        speakers=[],
        speaker_count=0,
        speaker_ids=[],
        full_text="",
        processing_time=0.0,
        is_final=is_final,
        stage=stage,
        error=None,
    )


def build_error_recognition_result(
    *,
    mode: str,
    backend_name: str,
    exc: Exception,
    start_time: Optional[datetime] = None,
    operation: str = "推理",
) -> RecognitionResult:
    """Build the standard error result returned by engine/runtime boundaries."""
    from src.core.config.errors import format_runtime_error
    
    return normalize_recognition_result(
        build_recognition_result(mode),
        mode=mode,
        is_final=False,
        start_time=start_time,
        error=format_runtime_error(
            mode=mode,
            backend_name=backend_name,
            exc=exc,
            operation=operation,
        ),
    )
