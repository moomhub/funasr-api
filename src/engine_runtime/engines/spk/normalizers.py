"""Normalize speaker backend payloads into shared result objects."""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.results import SpeakerResult, SpeakerSegment


def normalize_speaker_result(payload: Any) -> SpeakerResult:
    if isinstance(payload, SpeakerResult):
        return _copy_speaker_result(payload)
    if payload is None:
        return SpeakerResult(
            error="SPK 识别失败，请查看服务日志",
            metadata={"raw_payload": None},
        )
    if isinstance(payload, dict) and payload.get("error"):
        return SpeakerResult(
            error=str(payload.get("error")),
            metadata={"raw_payload": payload.get("raw_payload", payload)},
        )

    segments: List[SpeakerSegment] = []

    def add_segment(start: Any, end: Any, speaker: Any, seconds: bool = False) -> None:
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError):
            return
        if seconds:
            start_value *= 1000
            end_value *= 1000
        if end_value <= start_value:
            return
        segments.append(
            SpeakerSegment(
                speaker=speaker,
                start=start_value,
                end=end_value,
            )
        )

    def extract(data: Any) -> None:
        if data is None:
            return
        if isinstance(data, dict):
            if isinstance(data.get("sentences"), list):
                for item in data["sentences"]:
                    if isinstance(item, dict):
                        add_segment(
                            item.get("start", 0),
                            item.get("end", 0),
                            item.get("speaker", item.get("spk", "0")),
                        )
            if isinstance(data.get("text"), list):
                for item in data["text"]:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        add_segment(item[0], item[1], item[2], seconds=True)
            if isinstance(data.get("segments"), list):
                extract(data["segments"])
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    add_segment(
                        item.get("start", 0),
                        item.get("end", 0),
                        item.get("speaker", item.get("spk", "0")),
                    )
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    add_segment(item[0], item[1], item[2], seconds=True)

    extract(payload)
    speaker_ids = sorted(
        {segment.speaker for segment in segments},
        key=lambda value: (str(type(value)), str(value)),
    )
    metadata: Dict[str, Any] = {"raw_payload": payload}
    return SpeakerResult(
        segments=segments,
        speaker_ids=speaker_ids,
        speaker_count=len(speaker_ids),
        metadata=metadata,
    )


def _copy_speaker_result(result: SpeakerResult) -> SpeakerResult:
    return SpeakerResult(
        segments=[
            SpeakerSegment(
                speaker=segment.speaker,
                start=segment.start,
                end=segment.end,
                confidence=segment.confidence,
            )
            for segment in result.segments
        ],
        speaker_ids=list(result.speaker_ids),
        speaker_count=result.speaker_count,
        error=result.error,
        metadata=dict(result.metadata),
    )


__all__ = ["normalize_speaker_result"]
