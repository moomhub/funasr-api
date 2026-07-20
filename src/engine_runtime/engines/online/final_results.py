"""Pure result-shaping helpers for online final recognition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .text_boundary import collapse_adjacent_repeated_phrases
from .timestamp_utils import extract_global_timestamps


def empty_final_result(
    start_ms: int,
    end_ms: int,
    padded_start_ms: int,
    padded_end_ms: int,
) -> Dict[str, Any]:
    return {
        "text": "",
        "final_text": "",
        "raw_text": "",
        "timestamp": [],
        "tokens": [],
        "start": start_ms,
        "end": end_ms,
        "asr_start": padded_start_ms,
        "asr_end": padded_end_ms,
    }


def build_final_result(
    *,
    final_text: str,
    raw_text: str,
    raw_payload: Any,
    start_ms: int,
    end_ms: int,
    padded_start_ms: int,
    padded_end_ms: int,
) -> Dict[str, Any]:
    timestamps = extract_global_timestamps(
        raw_payload,
        raw_text=raw_text,
        start_ms=start_ms,
        end_ms=end_ms,
        padded_start_ms=padded_start_ms,
        padded_end_ms=padded_end_ms,
    )
    return {
        "text": final_text,
        "final_text": final_text,
        "raw_text": raw_text,
        "timestamp": timestamps,
        "tokens": [
            {"text": item[0], "start": item[1], "end": item[2]}
            for item in timestamps
        ],
        "start": start_ms,
        "end": end_ms,
        "asr_start": padded_start_ms,
        "asr_end": padded_end_ms,
    }


def final_text_from_result(final_result: Any) -> str:
    if isinstance(final_result, dict):
        return str(final_result.get("text") or final_result.get("final_text") or "")
    return str(final_result or "")


def build_locked_sentence(
    final_result: Any,
    start_ms: int,
    end_ms: int,
    source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(final_result, dict):
        text = str(final_result.get("text") or final_result.get("final_text") or "")
        raw_text = str(final_result.get("raw_text") or "")
        timestamps = list(final_result.get("timestamp") or [])
        tokens = list(final_result.get("tokens") or [])
        asr_start = final_result.get("asr_start")
        asr_end = final_result.get("asr_end")
    else:
        text = str(final_result or "")
        raw_text = ""
        timestamps: List[Any] = []
        tokens: List[Any] = []
        asr_start = None
        asr_end = None

    text = collapse_adjacent_repeated_phrases(text.strip())
    if not text:
        return None

    sentence: Dict[str, Any] = {
        "text": text,
        "start": start_ms,
        "end": end_ms,
        "is_final": True,
    }
    if raw_text:
        sentence["raw_text"] = raw_text
    if timestamps:
        sentence["timestamp"] = timestamps
    if tokens:
        sentence["tokens"] = tokens
    if asr_start is not None:
        sentence["asr_start"] = asr_start
    if asr_end is not None:
        sentence["asr_end"] = asr_end
    if source:
        sentence["source"] = source
    return sentence


__all__ = [
    "build_final_result",
    "build_locked_sentence",
    "empty_final_result",
    "final_text_from_result",
]
