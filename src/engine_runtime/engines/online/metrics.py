"""Realtime session metric state and snapshots."""

from __future__ import annotations

from typing import Dict, Iterable


def new_metrics() -> Dict[str, object]:
    return {
        "received_chunks": 0,
        "received_bytes": 0,
        "partial_decodes": 0,
        "final_decodes": 0,
        "partial_decode_time_ms": 0,
        "final_decode_time_ms": 0,
        "last_partial_decode_time_ms": 0,
        "last_final_decode_time_ms": 0,
        "max_partial_decode_time_ms": 0,
        "max_final_decode_time_ms": 0,
        "queue_high_watermark": 0,
        "dropped_chunks": 0,
        "backpressure_events": 0,
        "final_flush_text": "",
        "rejected_final_decodes": 0,
        "rejected_streaming_tails": 0,
    }


def build_metrics(
    metrics: Dict[str, object],
    duration_ms: int,
    pending_segments: Iterable[dict],
) -> Dict[str, object]:
    pending = list(pending_segments)
    total_decode_ms = int(metrics["partial_decode_time_ms"]) + int(metrics["final_decode_time_ms"])
    rtf = round(total_decode_ms / duration_ms, 3) if duration_ms > 0 else 0
    return {
        **metrics,
        "pending_final_segments": len(pending),
        "pending_final_duration_ms": sum(
            max(0, segment["end"] - segment["start"])
            for segment in pending
        ),
        "rtf": rtf,
    }


def record_decode_metrics(metrics: Dict[str, object], kind: str, decode_time_ms: int) -> None:
    if kind not in {"partial", "final"}:
        raise ValueError(f"Unsupported decode metric kind: {kind}")

    count_key = f"{kind}_decodes"
    total_key = f"{kind}_decode_time_ms"
    last_key = f"last_{kind}_decode_time_ms"
    max_key = f"max_{kind}_decode_time_ms"

    metrics[count_key] = int(metrics[count_key]) + 1
    metrics[total_key] = int(metrics[total_key]) + int(decode_time_ms)
    metrics[last_key] = int(decode_time_ms)
    metrics[max_key] = max(int(metrics[max_key]), int(decode_time_ms))


__all__ = ["build_metrics", "new_metrics", "record_decode_metrics"]
