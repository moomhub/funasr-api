"""Recognition result normalizers."""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .types import RecognitionResult, Segment


def normalize_recognition_result(
    result: Optional[RecognitionResult],
    *,
    mode: str,
    is_final: Optional[bool] = None,
    start_time: Optional[datetime] = None,
    error: Optional[str] = None,
) -> RecognitionResult:
    """Fill in shared RecognitionResult details consistently."""
    if result is None:
        from .builders import build_recognition_result
        result = build_recognition_result(mode)

    result.mode = mode
    result.segments = list(result.segments or [])
    result.speakers = list(result.speakers or [])
    result.speaker_ids = list(result.speaker_ids or [])
    result.full_text = result.full_text or ""

    if result.segments:
        segment_speaker_ids = _speaker_ids_from_segments(result.segments)
        if not result.speaker_ids:
            result.speaker_ids = segment_speaker_ids
        if not result.speakers:
            result.speakers = _speaker_ranges_from_segments(result.segments)
        if not result.full_text:
            result.full_text = "".join(segment.text for segment in result.segments)

    if result.speaker_count == 0:
        if result.speaker_ids:
            result.speaker_count = len(result.speaker_ids)
        elif result.speakers:
            result.speaker_count = len(result.speakers)

    if is_final is not None:
        result.is_final = is_final
    if error is not None:
        result.error = error
    if start_time is not None:
        result.processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

    return result


def _speaker_ids_from_segments(segments: Iterable[Segment]) -> List[Any]:
    return sorted(
        {segment.speaker for segment in segments},
        key=lambda value: (str(type(value)), str(value)),
    )


def _speaker_ranges_from_segments(segments: Iterable[Segment]) -> List[Dict[str, Any]]:
    segments = list(segments)
    ranges: Dict[Any, Dict[str, Any]] = {}
    for segment in segments:
        current = ranges.setdefault(
            segment.speaker,
            {
                "speaker_id": segment.speaker,
                "start": segment.start,
                "end": segment.end,
            },
        )
        current["start"] = min(current["start"], segment.start)
        current["end"] = max(current["end"], segment.end)

    return [ranges[speaker_id] for speaker_id in _speaker_ids_from_segments(segments)]
