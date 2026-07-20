"""Shared helpers for ONLINE funasr-onnx adapters."""

from __future__ import annotations

import inspect
from typing import Any, List, Optional, Sequence, Union

import numpy as np


class ONNXRealtimeUnsupportedError(RuntimeError):
    """Raised when funasr-onnx cannot satisfy the ONLINE realtime contract."""


def call_with_supported_kwargs(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)

    if any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return callable_obj(*args, **kwargs)

    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return callable_obj(*args, **supported)


def ensure_float32_audio(
    input_audio: Union[str, np.ndarray, Sequence[float]],
) -> Union[str, np.ndarray]:
    if isinstance(input_audio, str):
        return input_audio
    audio = np.asarray(input_audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    return audio


def normalize_online_chunk_size(chunk_size: Optional[List[int]]) -> List[int]:
    if not chunk_size:
        return [5, 10, 5]
    normalized = list(chunk_size)
    if len(normalized) != 3:
        return [5, 10, 5]
    if normalized[0] <= 0:
        normalized[0] = 5
    return normalized


def normalize_device_id(device_id: Union[str, int]) -> str:
    return str(device_id)


__all__ = [
    "ONNXRealtimeUnsupportedError",
    "call_with_supported_kwargs",
    "ensure_float32_audio",
    "normalize_device_id",
    "normalize_online_chunk_size",
]
