"""Stable WebSocket event payload assembly."""

from __future__ import annotations

from typing import Any, Dict, List


def online_event(
    *,
    partial: str,
    partial_start_ms: int,
    duration_ms: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "mode": "2pass-online",
        "text": partial,
        "partial": partial,
        "partial_start_ms": partial_start_ms,
        "duration_ms": duration_ms,
        "is_final": False,
        "metrics": metrics,
    }


def offline_event(
    *,
    sentences: List[Dict[str, Any]],
    duration_ms: int,
    is_final: bool,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "mode": "2pass-offline",
        "text": "".join(sentence["text"] for sentence in sentences),
        "sentences": list(sentences),
        "duration_ms": duration_ms,
        "is_final": is_final,
        "metrics": metrics,
    }


__all__ = ["offline_event", "online_event"]
