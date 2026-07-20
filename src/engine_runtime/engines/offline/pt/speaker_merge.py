"""Speaker/ASR merge helpers for offline PT recognition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.results import SpeakerResult
from src.core.results.types import SpeakerSegment
from src.engine_runtime.engines.offline.speaker_utils import (
    merge_timed_units_by_speaker,
    merge_timestamps,
    speaker_for_time_range_with_boundary,
    speaker_segments_from_result,
)
from src.engine_runtime.engines.offline.timestamp_utils import (
    align_text_to_timestamps,
    maybe_offset_timestamps,
    normalize_timestamps,
)


def merge_pt_sentence_info_with_speaker(
    sentence_info: List[Dict[str, Any]],
    speaker_result: Optional[SpeakerResult],
) -> List[Dict[str, Any]]:
    if speaker_result is None or speaker_result.error or not speaker_result.segments:
        return [dict(item) for item in sentence_info]

    speaker_segments = speaker_segments_from_result(speaker_result)
    token_units = build_timed_units(sentence_info)

    if token_units:
        return merge_units_by_speaker(token_units, speaker_segments)
    return assign_speaker_to_sentences(sentence_info, speaker_segments)


def build_timed_units(sentence_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for sentence in sentence_info:
        text = sentence.get("text", "")
        start_ms = int(sentence.get("start", 0) or 0)
        end_ms = int(sentence.get("end", 0) or 0)
        normalized_timestamps = normalize_timestamps(sentence.get("timestamp"))
        if normalized_timestamps:
            normalized_timestamps = maybe_offset_timestamps(
                normalized_timestamps,
                start_ms,
                end_ms,
            )
            aligned_tokens = align_text_to_timestamps(
                text,
                normalized_timestamps,
                include_timestamp_detail=True,
            )
            if aligned_tokens:
                units.extend(aligned_tokens)
                continue
        if text:
            units.append(
                {
                    "text": text,
                    "start": start_ms,
                    "end": end_ms,
                    "timestamp": sentence.get("timestamp"),
                }
            )
    return units


def merge_units_by_speaker(
    units: List[Dict[str, Any]],
    speaker_segments: List[SpeakerSegment],
) -> List[Dict[str, Any]]:
    return merge_timed_units_by_speaker(units, speaker_segments)


def assign_speaker_to_sentences(
    sentence_info: List[Dict[str, Any]],
    speaker_segments: List[SpeakerSegment],
) -> List[Dict[str, Any]]:
    merged = []
    for sentence in sentence_info:
        merged.append(
            {
                **dict(sentence),
                "spk": speaker_for_time_range_with_boundary(
                    int(sentence.get("start", 0) or 0),
                    int(sentence.get("end", 0) or 0),
                    speaker_segments,
                ),
            }
        )
    return merged


__all__ = [
    "assign_speaker_to_sentences",
    "build_timed_units",
    "merge_pt_sentence_info_with_speaker",
    "merge_timestamps",
    "merge_units_by_speaker",
]
