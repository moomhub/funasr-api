"""Punctuation adapter for ONLINE funasr-onnx."""

from __future__ import annotations

import contextlib
import logging
import re
import threading
from typing import Any, List, Sequence, Union

import numpy as np

from src.core.text import extract_model_text

from .common import (
    ONNXRealtimeUnsupportedError,
    call_with_supported_kwargs,
    normalize_device_id,
)

logger = logging.getLogger(__name__)


class ONNXPuncWrapper:
    """Punctuation model wrapper for ONLINE final text."""

    def __init__(
        self,
        model_dir: str,
        batch_size: int = 1,
        quantize: bool = True,
        num_threads: int = 4,
        device_id: int = -1,
    ):
        try:
            from funasr_onnx import CT_Transformer
        except ImportError as exc:
            raise ONNXRealtimeUnsupportedError(
                "funasr-onnx 未安装，无法加载 ONNX PUNC。"
            ) from exc

        self.model = CT_Transformer(
            model_dir=model_dir,
            batch_size=batch_size,
            quantize=quantize,
            intra_op_num_threads=num_threads,
            device_id=normalize_device_id(device_id),
        )
        self._model_lock = threading.Lock()
        logger.info("ONNX punctuation model loaded")
        logger.debug("ONNX punctuation model directory: %s", model_dir)

    def generate(
        self,
        input: Union[str, np.ndarray, Sequence[float]],
        **kwargs: Any,
    ) -> Any:
        text = str(input or "")
        if not text.strip():
            return [{"text": text}]
        try:
            punc_result = self._generate_with_model(text)
            return [{"text": extract_model_text(punc_result) or text}]
        except Exception as exc:
            logger.warning(
                "ONLINE ONNX punctuation failed, using heuristic fallback: error=%s",
                type(exc).__name__,
            )
            logger.debug("ONLINE ONNX punctuation error details", exc_info=True)
            return [{"text": self._fallback_punctuation(text)}]

    def _generate_with_model(self, text: str) -> Any:
        if len(text) <= 120:
            with getattr(self, "_model_lock", contextlib.nullcontext()):
                return call_with_supported_kwargs(self.model, text, split_size=10)

        punctuated_parts: List[str] = []
        for part in self._split_text_for_punc(text, max_chars=80):
            try:
                with getattr(self, "_model_lock", contextlib.nullcontext()):
                    result = call_with_supported_kwargs(
                        self.model,
                        part,
                        split_size=10,
                    )
                punctuated_parts.append(extract_model_text(result) or part)
            except Exception as exc:
                logger.warning(
                    "ONLINE ONNX punctuation chunk failed, using heuristic fallback: "
                    "error=%s",
                    type(exc).__name__,
                )
                logger.debug(
                    "ONLINE ONNX punctuation chunk error details",
                    exc_info=True,
                )
                punctuated_parts.append(self._fallback_punctuation(part))
        return [{"text": self._join_punctuated_parts(punctuated_parts)}]

    @staticmethod
    def _split_text_for_punc(text: str, max_chars: int = 80) -> List[str]:
        compact = re.sub(r"\s+", "", text)
        if len(compact) <= max_chars:
            return [compact]

        parts: List[str] = []
        start = 0
        soft_break_chars = set("了的呢啊呀吗吧嘛么着儿啦喽")
        while start < len(compact):
            end = min(start + max_chars, len(compact))
            if end < len(compact):
                for index in range(end, max(start + 30, end - 24), -1):
                    if compact[index - 1] in soft_break_chars:
                        end = index
                        break
            parts.append(compact[start:end])
            start = end
        return [part for part in parts if part]

    @staticmethod
    def _fallback_punctuation(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return ""
        if compact[-1] in "。！？!?":
            return compact
        return f"{compact}。"

    @staticmethod
    def _join_punctuated_parts(parts: List[str]) -> str:
        cleaned = [part.strip() for part in parts if part and part.strip()]
        if not cleaned:
            return ""
        result = "".join(cleaned)
        if result and result[-1] not in "。！？!?":
            result += "。"
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.generate(*args, **kwargs)


__all__ = ["ONNXPuncWrapper"]
