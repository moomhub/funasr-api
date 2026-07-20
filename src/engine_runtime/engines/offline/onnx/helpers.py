"""Pure helpers for offline ONNX recognition payload shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from src.engine_runtime.engines.offline.timestamp_utils import normalize_timestamps


@dataclass(frozen=True)
class ASRSegmentDecode:
    index: int
    text: str
    timestamps: List[List[Any]]
    payload: dict[str, Any]


def slice_audio_by_ms(
    audio_data: Any,
    *,
    start_ms: int,
    end_ms: int,
    padding_ms: int,
    sample_rate: int,
) -> tuple[Any, int]:
    padded_start_ms = max(0, int(start_ms) - int(padding_ms))
    duration_ms = int(len(audio_data) * 1000 / int(sample_rate))
    padded_end_ms = min(duration_ms, int(end_ms) + int(padding_ms))
    start_sample = int(padded_start_ms * int(sample_rate) / 1000)
    end_sample = int(padded_end_ms * int(sample_rate) / 1000)
    return audio_data[start_sample:end_sample], padded_start_ms


def empty_asr_segment_decode(index: int, start_ms: int, end_ms: int) -> ASRSegmentDecode:
    return ASRSegmentDecode(
        index=index,
        text="",
        timestamps=[],
        payload={
            "text": "",
            "start": int(start_ms),
            "end": int(end_ms),
            "timestamp": [],
        },
    )


def asr_segment_decode(
    *,
    index: int,
    text: str,
    start_ms: int,
    end_ms: int,
    timestamps: List[List[Any]],
) -> ASRSegmentDecode:
    return ASRSegmentDecode(
        index=index,
        text=text,
        timestamps=timestamps,
        payload={
            "text": text,
            "start": int(start_ms),
            "end": int(end_ms),
            "timestamp": timestamps,
        },
    )


def combine_asr_segment_decodes(
    decodes: Sequence[ASRSegmentDecode],
    segment_count: int,
) -> tuple[str, List[List[Any]], List[dict[str, Any]]]:
    text_parts = [""] * segment_count
    timestamp_parts: List[List[List[Any]]] = [[] for _ in range(segment_count)]
    payload_parts: List[Optional[dict[str, Any]]] = [None] * segment_count

    for decode in decodes:
        text_parts[decode.index] = decode.text
        timestamp_parts[decode.index] = decode.timestamps
        payload_parts[decode.index] = decode.payload

    return (
        "".join(text_parts),
        [item for part in timestamp_parts for item in part],
        [payload for payload in payload_parts if payload is not None],
    )


def extract_vad_segments(result: Any) -> List[List[int]]:
    segments: List[List[int]] = []

    def extract(data: Any) -> None:
        if data is None:
            return
        if isinstance(data, dict):
            for key in ("value", "segments", "sentence_info"):
                if key in data:
                    extract(data[key])
        elif isinstance(data, (list, tuple)):
            if len(data) == 2 and all(isinstance(value, (int, float)) for value in data):
                start, end = int(data[0]), int(data[1])
                if start >= 0 and end > start:
                    segments.append([start, end])
            else:
                for item in data:
                    extract(item)

    extract(result)
    return sorted(segments, key=lambda item: (item[0], item[1]))


def extract_timestamps(result: Any) -> Optional[List[List[Any]]]:
    if result is None:
        return None
    if isinstance(result, dict):
        if "timestamp" in result:
            return normalize_timestamps(result["timestamp"])
        collected: List[List[Any]] = []
        for key in ("timestamps", "value", "result", "results", "preds", "sentence", "raw", "raw_result"):
            item_timestamps = extract_timestamps(result.get(key))
            if item_timestamps:
                collected.extend(item_timestamps)
        return collected or None
    if isinstance(result, list):
        if is_timestamp_format(result):
            return normalize_timestamps(result)
        collected: List[List[Any]] = []
        for item in result:
            item_timestamps = extract_timestamps(item)
            if item_timestamps:
                collected.extend(item_timestamps)
        return collected or None
    if isinstance(result, tuple):
        collected: List[List[Any]] = []
        for item in result:
            if isinstance(item, (list, tuple)) and is_timestamp_format(item):
                item_timestamps = normalize_timestamps(item)
            else:
                item_timestamps = extract_timestamps(item)
            if item_timestamps:
                collected.extend(item_timestamps)
        return collected or None
    return None


def is_timestamp_format(data: Any) -> bool:
    if not isinstance(data, (list, tuple)) or not data:
        return False
    first = data[0]
    return isinstance(first, (list, tuple)) and len(first) in {2, 3}


__all__ = [
    "ASRSegmentDecode",
    "asr_segment_decode",
    "combine_asr_segment_decodes",
    "empty_asr_segment_decode",
    "extract_timestamps",
    "extract_vad_segments",
    "is_timestamp_format",
    "slice_audio_by_ms",
]
