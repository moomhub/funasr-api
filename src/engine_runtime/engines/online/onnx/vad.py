"""VAD runtime and per-WebSocket state for ONLINE funasr-onnx."""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from .common import (
    ONNXRealtimeUnsupportedError,
    call_with_supported_kwargs,
    ensure_float32_audio,
    normalize_device_id,
)

logger = logging.getLogger(__name__)


def extract_vad_segments(result: Any) -> List[List[float]]:
    if result is None:
        return []
    if isinstance(result, dict):
        for key in ("value", "segments", "timestamp"):
            if key in result:
                return extract_vad_segments(result[key])
        return []
    if isinstance(result, (list, tuple)):
        if len(result) == 2 and all(isinstance(item, (int, float)) for item in result):
            start = float(result[0])
            end = float(result[1])
            if start < 0 and end < 0:
                return []
            return [[start, end]]
        segments: List[List[float]] = []
        for item in result:
            segments.extend(extract_vad_segments(item))
        return segments
    return []


class ONNXVADSession:
    """Per-WebSocket state for a shared ONNX VAD model."""

    def __init__(self, runtime: "ONNXVADWrapper"):
        self.runtime = runtime
        self.reset()

    def reset(self) -> None:
        self.cache: List[Any] = []
        self.current_speech_start: Optional[int] = None

    def feed(
        self,
        audio: Union[np.ndarray, Sequence[float]],
        is_final: bool = False,
    ) -> List[List[float]]:
        audio_data = ensure_float32_audio(audio)
        param_dict = {
            "in_cache": self.cache,
            "is_final": is_final,
        }
        try:
            with getattr(self.runtime, "_model_lock", contextlib.nullcontext()):
                result = call_with_supported_kwargs(
                    self.runtime.model,
                    audio_data,
                    param_dict=param_dict,
                )
            self.cache = param_dict.get("in_cache", self.cache)
        except TypeError as exc:
            raise ONNXRealtimeUnsupportedError(
                "当前 funasr-onnx FsmnVAD 不支持 ONLINE 所需的 ndarray/cache/is_final 调用。"
            ) from exc

        segments = extract_vad_segments(result)
        if not segments:
            return []

        open_segments = [
            segment
            for segment in segments
            if len(segment) == 2 and segment[1] < 0 and segment[0] >= 0
        ]
        if open_segments:
            self.current_speech_start = int(open_segments[-1][0])

        closed_segments: List[List[float]] = []
        for segment in segments:
            if len(segment) != 2 or segment[1] < 0:
                continue
            start, end = segment
            if start < 0 and self.current_speech_start is not None:
                start = float(self.current_speech_start)
            closed_segments.append([start, end])
            self.current_speech_start = None
        return closed_segments


class ONNXVADWrapper:
    """Shared ONNX VAD model with isolated per-session state."""

    def __init__(
        self,
        model_dir: str,
        batch_size: int = 1,
        quantize: bool = True,
        num_threads: int = 4,
        device_id: int = -1,
    ):
        try:
            try:
                from funasr_onnx import Fsmn_vad_online as FsmnVAD
            except ImportError:
                from funasr_onnx import FsmnVAD
        except ImportError as exc:
            raise ONNXRealtimeUnsupportedError(
                "funasr-onnx 已安装但未提供 FsmnVAD/Fsmn_vad_online，当前版本无法启用 ONLINE ONNX VAD。"
            ) from exc

        self.model_dir = model_dir
        self.model = FsmnVAD(
            model_dir=model_dir,
            batch_size=batch_size,
            quantize=quantize,
            intra_op_num_threads=num_threads,
            device_id=normalize_device_id(device_id),
        )
        self._model_lock = threading.Lock()
        self._default_session = ONNXVADSession(self)
        if not callable(self.model):
            raise ONNXRealtimeUnsupportedError(
                "FsmnVAD 不提供可调用接口，无法用于 ONLINE ONNX VAD。"
            )
        logger.info("ONNX VAD loaded")
        logger.debug("ONNX VAD model directory: %s", model_dir)

    def reset(self) -> None:
        self._default_session.reset()

    @property
    def cache(self) -> List[Any]:
        return self._default_session.cache

    @property
    def current_speech_start(self) -> Optional[int]:
        return self._default_session.current_speech_start

    @current_speech_start.setter
    def current_speech_start(self, value: Optional[int]) -> None:
        self._default_session.current_speech_start = value

    def create_session(self) -> ONNXVADSession:
        return ONNXVADSession(self)

    def feed(
        self,
        audio: Union[np.ndarray, Sequence[float]],
        is_final: bool = False,
    ) -> List[List[float]]:
        return self._default_session.feed(audio, is_final)


__all__ = ["ONNXVADSession", "ONNXVADWrapper", "extract_vad_segments"]
