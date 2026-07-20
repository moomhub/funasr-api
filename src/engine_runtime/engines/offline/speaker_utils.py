"""说话人识别工具函数 - 消除重复代码"""

from typing import Any, Dict, List, Optional

from src.core.results.types import SpeakerSegment


def speaker_for_time_range(
    start_ms: int,
    end_ms: int,
    speaker_segments: List[SpeakerSegment],
) -> Any:
    """根据时间范围查找最匹配的说话人
    
    算法：
    1. 计算每个说话人段与目标时间范围的重叠
    2. 返回重叠最大的说话人
    3. 如果没有重叠，返回时间中点最近的说话人
    4. 如果没有说话人段，返回 0
    
    参数：
        start_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒）
        speaker_segments: 说话人段列表（已排序）
    
    返回：
        说话人 ID
    """
    best_segment = None
    best_overlap = -1.0
    
    for segment in speaker_segments:
        overlap = min(end_ms, segment.end) - max(start_ms, segment.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_segment = segment
    
    if best_segment and best_overlap > 0:
        return best_segment.speaker
    
    if speaker_segments:
        midpoint = (start_ms + end_ms) / 2
        nearest = min(
            speaker_segments,
            key=lambda seg: min(abs(midpoint - seg.start), abs(midpoint - seg.end)),
        )
        return nearest.speaker
    
    return 0


def speaker_for_short_boundary_token(
    start_ms: int,
    end_ms: int,
    speaker_segments: List[SpeakerSegment],
) -> Optional[Any]:
    """为短边界 token 分配说话人
    
    当 token 持续时间 <= 250ms 且跨越说话人边界时，
    返回边界后的说话人。
    
    参数：
        start_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒）
        speaker_segments: 说话人段列表
    
    返回：
        说话人 ID 或 None
    """
    duration = int(end_ms) - int(start_ms)
    if duration <= 0 or duration > 250:
        return None

    ordered = sorted(speaker_segments, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.speaker == current.speaker:
            continue
        boundary = int(current.start)
        if int(start_ms) < boundary < int(end_ms):
            return current.speaker
    return None


def speaker_for_time_range_with_boundary(
    start_ms: int,
    end_ms: int,
    speaker_segments: List[SpeakerSegment],
) -> Any:
    """Resolve speaker with a short-token correction around speaker boundaries."""
    boundary_speaker = speaker_for_short_boundary_token(
        start_ms,
        end_ms,
        speaker_segments,
    )
    if boundary_speaker is not None:
        return boundary_speaker
    return speaker_for_time_range(start_ms, end_ms, speaker_segments)


def merge_timestamps(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, list) and isinstance(incoming, list):
        return [*existing, *incoming]
    return existing or incoming


def merge_timed_units_by_speaker(
    units: List[Dict[str, Any]],
    speaker_segments: List[SpeakerSegment],
) -> List[Dict[str, Any]]:
    """Group timed text units into speaker-aligned sentence payloads."""
    merged: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for unit in sorted(units, key=lambda item: (item["start"], item["end"])):
        speaker_id = speaker_for_time_range_with_boundary(
            int(unit["start"]),
            int(unit["end"]),
            speaker_segments,
        )
        unit_payload = {
            "text": unit["text"],
            "start": int(unit["start"]),
            "end": int(unit["end"]),
            "spk": speaker_id,
        }
        if "timestamp" in unit:
            unit_payload["timestamp"] = unit.get("timestamp")

        if current and current["spk"] == speaker_id:
            current["text"] += unit_payload["text"]
            current["end"] = unit_payload["end"]
            if "timestamp" in current or "timestamp" in unit_payload:
                current["timestamp"] = merge_timestamps(
                    current.get("timestamp"),
                    unit_payload.get("timestamp"),
                )
        else:
            if current and current["text"].strip():
                merged.append(current)
            current = unit_payload

    if current and current["text"].strip():
        merged.append(current)
    return merged


def speaker_segments_from_result(speaker_result: Any) -> List[SpeakerSegment]:
    """从 SpeakerResult 提取排序后的说话人段列表"""
    return sorted(speaker_result.segments, key=lambda item: (item.start, item.end))
