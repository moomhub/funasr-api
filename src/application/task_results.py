"""Application contracts for task result side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from src.core.results.types import RecognitionResult


@dataclass
class OfflineTaskContext:
    task_id: str
    filename: str
    audio_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OfflineBatchContext:
    task_ids: List[str]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OfflinePersistencePayload:
    full_text: str
    segment_payloads: List[Dict[str, Any]]
    word_timestamps: List[Any]
    processing_time: float
    summary: Dict[str, Any]


def extract_word_timestamps(segment_payloads: List[Dict[str, Any]]) -> List[Any]:
    timestamps: List[Any] = []
    for segment in segment_payloads:
        timestamp = segment.get("timestamp") if isinstance(segment, dict) else None
        if isinstance(timestamp, list):
            timestamps.extend(timestamp)
    return timestamps


def build_offline_persistence_payload(
    context: OfflineTaskContext,
    result: RecognitionResult,
) -> OfflinePersistencePayload:
    segment_payloads = [segment.to_dict() for segment in result.segments]
    word_timestamps = extract_word_timestamps(segment_payloads)
    full_text = result.full_text or ""
    summary = {
        "task_id": context.task_id,
        "text_length": len(full_text),
        "segment_count": len(segment_payloads),
        "word_timestamp_count": len(word_timestamps),
        "speaker_ids": list(result.speaker_ids or []),
        "speaker_count": result.speaker_count,
        "metadata_keys": sorted((result.metadata or {}).keys()),
    }
    return OfflinePersistencePayload(
        full_text=full_text,
        segment_payloads=segment_payloads,
        word_timestamps=word_timestamps,
        processing_time=result.processing_time,
        summary=summary,
    )


class OfflineTaskResultHandlerPort(Protocol):
    async def handle_success(self, context: OfflineTaskContext, result: RecognitionResult) -> None:
        ...

    async def handle_failure(self, context: OfflineTaskContext, error_message: str) -> None:
        ...


class OfflineBatchResultHandlerPort(Protocol):
    async def handle_complete(self, context: OfflineBatchContext) -> None:
        ...


__all__ = [
    "OfflineBatchContext",
    "OfflineBatchResultHandlerPort",
    "OfflinePersistencePayload",
    "OfflineTaskContext",
    "OfflineTaskResultHandlerPort",
    "build_offline_persistence_payload",
    "extract_word_timestamps",
]
