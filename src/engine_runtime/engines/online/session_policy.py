"""Pure policy helpers for online realtime session state transitions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional


def effective_segment_merge_gap_ms(vad_merge_gap_ms: int, vad_post_padding_ms: int) -> int:
    return int(vad_merge_gap_ms) + int(vad_post_padding_ms)


def required_tail_gap_ms(
    duration_ms: int,
    *,
    vad_min_final_ms: int,
    segment_merge_gap_ms: int,
) -> int:
    if duration_ms >= vad_min_final_ms:
        return segment_merge_gap_ms
    return max(segment_merge_gap_ms, vad_min_final_ms - duration_ms)


def should_flush_pending_segment(
    segment: Mapping[str, int],
    *,
    force: bool = False,
    current_duration_ms: int,
    vad_max_final_ms: int,
    vad_min_final_ms: int,
    segment_merge_gap_ms: int,
    active_speech_start: Optional[Any] = None,
) -> bool:
    if force:
        return True

    duration_ms = max(0, int(segment["end"]) - int(segment["start"]))
    if duration_ms >= vad_max_final_ms:
        return True

    tail_gap_ms = max(0, int(current_duration_ms) - int(segment["end"]))
    required_gap_ms = required_tail_gap_ms(
        duration_ms,
        vad_min_final_ms=vad_min_final_ms,
        segment_merge_gap_ms=segment_merge_gap_ms,
    )

    if active_speech_start is not None:
        active_gap_ms = int(active_speech_start) - int(segment["end"])
        if active_gap_ms <= segment_merge_gap_ms:
            return False

    if tail_gap_ms < required_gap_ms:
        return False
    return duration_ms > 0


def audio_trim_keep_from_sample(
    *,
    last_stream_samples: int,
    vad_fed_samples: int,
    pending_final_segments: Iterable[Mapping[str, int]],
    vad_pre_padding_ms: int,
    sample_rate: int,
    active_speech_start: Optional[Any] = None,
) -> int:
    keep_from_sample = min(int(last_stream_samples), int(vad_fed_samples))

    pending_starts = [
        max(
            0,
            int(segment["start"] * sample_rate / 1000)
            - int(vad_pre_padding_ms * sample_rate / 1000),
        )
        for segment in pending_final_segments
    ]
    if pending_starts:
        keep_from_sample = min(keep_from_sample, min(pending_starts))

    if active_speech_start is not None:
        keep_from_sample = min(
            keep_from_sample,
            max(0, int((int(active_speech_start) - vad_pre_padding_ms) * sample_rate / 1000)),
        )

    return max(0, keep_from_sample)


__all__ = [
    "audio_trim_keep_from_sample",
    "effective_segment_merge_gap_ms",
    "required_tail_gap_ms",
    "should_flush_pending_segment",
]
