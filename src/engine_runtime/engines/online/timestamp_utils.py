"""Pure timestamp extraction and normalization helpers for online ASR."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from src.core.text_utils import compact_token_text, text_tokens


def extract_timestamp_payload(payload: Any) -> Any:
    """Find the first timestamp-like payload in a nested model response."""
    if payload is None:
        return None

    if isinstance(payload, dict):
        for key in ("timestamp", "timestamps", "time_stamp", "time_stamps"):
            if key in payload and payload[key]:
                return payload[key]
        for key in ("value", "result", "results", "preds", "sentence", "raw", "raw_result"):
            nested = extract_timestamp_payload(payload.get(key))
            if nested:
                return nested
        return None

    if isinstance(payload, tuple):
        if len(payload) == 2 and isinstance(payload[1], list):
            return payload[1]
        for item in payload:
            nested = extract_timestamp_payload(item)
            if nested:
                return nested
        return None

    if isinstance(payload, list):
        if payload and all(_looks_like_timestamp_item(item) for item in payload):
            return payload
        for item in payload:
            nested = extract_timestamp_payload(item)
            if nested:
                return nested
        return None

    return None


def extract_global_timestamps(
    raw_payload: Any,
    *,
    raw_text: str,
    start_ms: int,
    end_ms: int,
    padded_start_ms: int,
    padded_end_ms: int,
) -> List[List[Any]]:
    """Normalize model timestamps and map them into session-global time."""
    timestamp_payload = extract_timestamp_payload(raw_payload)
    if not timestamp_payload:
        return []

    raw_tokens = text_tokens(raw_text)
    local_duration_ms = max(0, padded_end_ms - padded_start_ms)
    max_observed = 0
    normalized: List[List[Any]] = []
    for index, item in enumerate(timestamp_payload):
        token = raw_tokens[index] if index < len(raw_tokens) else ""
        if not isinstance(item, (list, tuple)):
            continue
        if len(item) == 2 and is_number(item[0]) and is_number(item[1]):
            token_start = to_int_ms(item[0])
            token_end = to_int_ms(item[1])
        elif len(item) >= 3 and is_number(item[-2]) and is_number(item[-1]):
            item_token = compact_token_text(item[0])
            token = item_token or token
            token_start = to_int_ms(item[-2])
            token_end = to_int_ms(item[-1])
        else:
            continue

        max_observed = max(max_observed, token_start, token_end)
        normalized.append([token, token_start, token_end])

    if not normalized:
        return []

    should_offset = max_observed <= local_duration_ms + 1000
    global_timestamps: List[List[Any]] = []
    for token, token_start, token_end in normalized:
        if should_offset:
            token_start += padded_start_ms
            token_end += padded_start_ms
        token_start = max(start_ms, min(token_start, end_ms))
        token_end = max(start_ms, min(token_end, end_ms))
        if token_end < token_start:
            token_end = token_start
        global_timestamps.append([token, int(token_start), int(token_end)])
    return global_timestamps


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def to_int_ms(value: Any) -> int:
    return int(round(float(value)))


def _looks_like_timestamp_item(item: Any) -> bool:
    if not isinstance(item, (list, tuple)):
        return False
    if len(item) == 2:
        return all(is_number(value) for value in item)
    if len(item) >= 3:
        return is_number(item[-2]) and is_number(item[-1])
    return False


__all__ = [
    "extract_global_timestamps",
    "extract_timestamp_payload",
    "is_number",
    "to_int_ms",
]
