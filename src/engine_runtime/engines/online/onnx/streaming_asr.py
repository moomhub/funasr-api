"""Streaming ASR adapter for ONLINE funasr-onnx."""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .common import (
    ONNXRealtimeUnsupportedError,
    call_with_supported_kwargs,
    ensure_float32_audio,
    normalize_device_id,
    normalize_online_chunk_size,
)

logger = logging.getLogger(__name__)


class ONNXStreamingASRWrapper:
    """Adapter for funasr_onnx.ParaformerOnline."""

    def __init__(
        self,
        model_dir: str,
        batch_size: int = 1,
        quantize: bool = True,
        num_threads: int = 4,
        device_id: int = -1,
        chunk_size: Optional[List[int]] = None,
    ):
        try:
            try:
                from funasr_onnx import ParaformerOnline
            except ImportError:
                from funasr_onnx.paraformer_online_bin import Paraformer as ParaformerOnline
        except ImportError as exc:
            raise ONNXRealtimeUnsupportedError(
                "funasr-onnx 已安装但未提供 ParaformerOnline/paraformer_online_bin.Paraformer，"
                "当前版本无法启用 ONLINE ONNX streaming ASR。"
            ) from exc

        self.model_dir = model_dir
        self.chunk_size = normalize_online_chunk_size(chunk_size)
        self.model = ParaformerOnline(
            model_dir=model_dir,
            batch_size=batch_size,
            chunk_size=self.chunk_size,
            quantize=quantize,
            intra_op_num_threads=num_threads,
            device_id=normalize_device_id(device_id),
        )
        self._model_lock = threading.Lock()
        self._validate_callable("ParaformerOnline")
        logger.info("ONNX streaming ASR loaded")
        logger.debug("ONNX streaming ASR model directory: %s", model_dir)

    def _validate_callable(self, class_name: str) -> None:
        if not callable(self.model) and not hasattr(self.model, "generate"):
            raise ONNXRealtimeUnsupportedError(
                f"{class_name} 不提供可调用接口或 generate()，无法用于 ONLINE 实时识别。"
            )

    def generate(
        self,
        input: Union[str, np.ndarray, Sequence[float]],
        cache: Optional[Dict[str, Any]] = None,
        is_final: bool = False,
        chunk_size: Optional[List[int]] = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        hotwords: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        audio = ensure_float32_audio(input)
        cache = cache if cache is not None else {}
        call_kwargs = {
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "decoder_chunk_look_back": decoder_chunk_look_back,
            **kwargs,
        }
        if hotwords:
            call_kwargs["hotwords"] = hotwords

        try:
            target = self.model.generate if hasattr(self.model, "generate") else self.model
            with getattr(self, "_model_lock", contextlib.nullcontext()):
                return call_with_supported_kwargs(
                    target,
                    audio,
                    param_dict={
                        "cache": cache,
                        "is_final": is_final,
                    },
                    **call_kwargs,
                )
        except TypeError as exc:
            raise ONNXRealtimeUnsupportedError(
                "当前 funasr-onnx ParaformerOnline 不支持 ONLINE 所需的 ndarray/chunk/cache 调用。"
            ) from exc

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate(*args, **kwargs)


__all__ = ["ONNXStreamingASRWrapper"]
