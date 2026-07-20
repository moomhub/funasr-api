"""Event payload helpers for ONLINE WebSocket streams."""

from __future__ import annotations

from typing import Any, List, Optional


def normalize_stream_command(command: str) -> str:
    return (command or "").strip().upper()


def ready_event(
    *,
    sample_rate: int,
    decode_interval: float,
    queue_max_chunks: int,
    hotword_id: Optional[int],
    resolved_hotwords: Optional[List[Any]],
) -> dict:
    return {
        "event": "ready",
        "mode": "2pass",
        "sample_rate": sample_rate,
        "decode_interval": decode_interval,
        "queue_max_chunks": queue_max_chunks,
        "hotword_id": hotword_id,
        "hotword_count": len(resolved_hotwords or []),
    }


def started_event() -> dict:
    return {"event": "started", "mode": "2pass"}


def stopped_event(metrics: dict) -> dict:
    return {"event": "stopped", "metrics": metrics}


def error_event(message: str) -> dict:
    return {"event": "error", "error": message}


def unknown_command_event(command: str) -> dict:
    return error_event(f"未知命令: {command}")


def inactive_session_event() -> dict:
    return error_event("会话未开始，请先发送 START")


def backpressure_event(*, queue_size: int, queue_max_chunks: int, metrics: dict) -> dict:
    return {
        "event": "backpressure",
        "mode": "2pass",
        "queue_size": queue_size,
        "queue_max_chunks": queue_max_chunks,
        "metrics": metrics,
    }


__all__ = [
    "backpressure_event",
    "error_event",
    "inactive_session_event",
    "normalize_stream_command",
    "ready_event",
    "started_event",
    "stopped_event",
    "unknown_command_event",
]
