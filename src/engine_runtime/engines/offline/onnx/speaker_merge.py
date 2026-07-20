"""Speaker/ASR merge helpers for offline ONNX recognition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.results import SpeakerResult
from src.core.results.types import SpeakerSegment
from src.engine_runtime.engines.offline.sentence_builder import (
    build_sentence_info_from_timestamps,
    build_sentence_info_from_vad,
)
from src.engine_runtime.engines.offline.speaker_utils import (
    merge_timed_units_by_speaker,
    speaker_for_time_range_with_boundary,
    speaker_segments_from_result,
)
from src.engine_runtime.engines.offline.timestamp_utils import align_text_to_timestamps


def merge_onnx_speaker_result(
    payload: Dict[str, Any],
    speaker: Optional[SpeakerResult],
) -> Dict[str, Any]:
    payload = dict(payload or {})
    if speaker is None or speaker.error or not speaker.segments:
        if speaker and speaker.error:
            payload["speaker_error"] = speaker.error
        payload["sentence_info"] = build_asr_sentence_info(payload)
        return payload

    speaker_segments = speaker_segments_from_result(speaker)
    sentence_info: List[Dict[str, Any]] = []
    timestamps = payload.get("timestamps") or []
    if timestamps:
        sentence_info = build_speaker_aligned_text(
            payload.get("text") or payload.get("raw_text") or "",
            timestamps,
            speaker_segments,
        )
    if not sentence_info:
        sentence_info = assign_speakers_to_asr_segments(payload, speaker_segments)
    if not sentence_info:
        sentence_info = build_asr_sentence_info(payload)

    payload["sentence_info"] = sentence_info
    payload["speaker_result"] = speaker.to_dict()
    return payload


def assign_speakers_to_asr_segments(
    payload: Dict[str, Any],
    speaker_segments: List[SpeakerSegment],
) -> List[Dict[str, Any]]:
    sentence_info = []
    for segment in payload.get("asr_segments") or []:
        if not segment.get("text"):
            continue
        speaker_id = speaker_for_time_range_with_boundary(
            segment["start"],
            segment["end"],
            speaker_segments,
        )
        sentence_info.append(
            {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "spk": speaker_id,
                "timestamp": segment.get("timestamp"),
            }
        )
    return sentence_info


def build_asr_sentence_info(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    asr_segments = [
        {
            "text": segment.get("text", ""),
            "start": int(segment.get("start", 0)),
            "end": int(segment.get("end", 0)),
            "spk": 0,
            "timestamp": segment.get("timestamp"),
        }
        for segment in payload.get("asr_segments") or []
        if segment.get("text")
    ]
    if asr_segments:
        return asr_segments

    timestamps = payload.get("timestamps") or []
    text = payload.get("text") or payload.get("raw_text") or ""
    if timestamps:
        return [
            {**item, "spk": 0}
            for item in build_sentence_info_from_timestamps(text, timestamps)
        ]
    vad_segments = payload.get("vad_segments") or []
    if vad_segments:
        return [
            {**item, "spk": 0}
            for item in build_sentence_info_from_vad(text, vad_segments)
        ]
    return [{"text": text, "start": 0, "end": 0, "spk": 0}] if text else []


def build_speaker_aligned_text(
    text: str,
    timestamps: List[List[Any]],
    speaker_segments: List[SpeakerSegment],
) -> List[Dict[str, Any]]:
    timed_tokens = align_text_to_timestamps(text, timestamps)
    if not timed_tokens:
        return []
    return merge_timed_units_by_speaker(timed_tokens, speaker_segments)


__all__ = [
    "assign_speakers_to_asr_segments",
    "build_asr_sentence_info",
    "build_speaker_aligned_text",
    "merge_onnx_speaker_result",
    "speaker_for_time_range_with_boundary",
]
