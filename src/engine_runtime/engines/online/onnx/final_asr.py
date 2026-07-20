"""Final ASR adapter for ONLINE funasr-onnx."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import threading
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from src.core.text import extract_model_text

from .common import (
    ONNXRealtimeUnsupportedError,
    call_with_supported_kwargs,
    ensure_float32_audio,
    normalize_device_id,
)

logger = logging.getLogger(__name__)

ASR_HOTWORD_MODE_PLAIN = "plain"
ASR_HOTWORD_MODE_SEACO = "seaco"


def _read_model_type(model_dir: str) -> str:
    config_path = Path(model_dir) / "config.yaml"
    if not config_path.is_file():
        return ""
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    return str(config.get("model") or "").strip()


def _is_seaco_model(model_dir: str) -> bool:
    normalized = str(model_dir).replace("\\", "/").lower()
    model_path = Path(model_dir)
    return (
        "seaco_paraformer" in normalized
        or _read_model_type(model_dir).lower() == "seacoparaformer"
        or (model_path / "model_eb.onnx").is_file()
        or (model_path / "model_eb_quant.onnx").is_file()
    )


def _seaco_quantize_mode(model_dir: str, requested: bool) -> bool:
    model_path = Path(model_dir)
    if requested and (model_path / "model_quant.onnx").is_file() and (model_path / "model_eb_quant.onnx").is_file():
        return True
    if requested:
        logger.info("ONLINE ONNX SeAcO final ASR 未提供完整量化模型，使用非量化模型")
    if model_path.exists():
        missing = [
            name
            for name in ("model.onnx", "model_eb.onnx")
            if not (model_path / name).is_file()
        ]
        if missing:
            raise ONNXRealtimeUnsupportedError(
                "ONLINE ONNX SeAcO final ASR 模型不完整，缺少: " + ", ".join(missing)
            )
    return False


class ONNXFinalASRWrapper:
    """Final ASR for ONLINE completed segments."""

    def __init__(
        self,
        asr_model_dir: str,
        batch_size: int = 1,
        quantize: bool = True,
        num_threads: int = 4,
        device_id: int = -1,
    ):
        try:
            import funasr_onnx
        except ImportError as exc:
            raise ONNXRealtimeUnsupportedError(
                "funasr-onnx 未安装，无法加载 ONNX final ASR。"
            ) from exc

        self.asr_hotword_mode = ASR_HOTWORD_MODE_SEACO if _is_seaco_model(asr_model_dir) else ASR_HOTWORD_MODE_PLAIN
        model_factory = funasr_onnx.Paraformer
        asr_quantize = quantize
        if self.asr_hotword_mode == ASR_HOTWORD_MODE_SEACO:
            model_factory = getattr(funasr_onnx, "SeacoParaformer", None)
            if model_factory is None:
                raise ONNXRealtimeUnsupportedError("当前 funasr-onnx 未提供 SeacoParaformer")
            asr_quantize = _seaco_quantize_mode(asr_model_dir, quantize)

        self.asr_model = model_factory(
            model_dir=asr_model_dir,
            batch_size=batch_size,
            quantize=asr_quantize,
            intra_op_num_threads=num_threads,
            device_id=normalize_device_id(device_id),
        )
        self._model_lock = threading.Lock()
        if not callable(self.asr_model):
            raise ONNXRealtimeUnsupportedError(
                "Paraformer 不提供可调用接口，无法用于 ONLINE final decode。"
            )
        logger.info("ONNX final ASR loaded: hotword_mode=%s", self.asr_hotword_mode)
        logger.debug("ONNX final ASR model directory: %s", asr_model_dir)

    def generate(
        self,
        input: Union[str, np.ndarray, Sequence[float]],
        hotwords: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        audio = ensure_float32_audio(input)
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"batch_size_s", "return_spk_res"}
        }
        asr_hotword_mode = getattr(self, "asr_hotword_mode", ASR_HOTWORD_MODE_PLAIN)
        if asr_hotword_mode == ASR_HOTWORD_MODE_SEACO:
            filtered_kwargs["hotwords"] = self._format_seaco_hotwords(hotwords)
        elif hotwords:
            filtered_kwargs["hotwords"] = hotwords

        try:
            with getattr(self, "_model_lock", contextlib.nullcontext()):
                asr_result = call_with_supported_kwargs(
                    self.asr_model,
                    audio,
                    **filtered_kwargs,
                )
        except TypeError as exc:
            raise ONNXRealtimeUnsupportedError(
                "当前 funasr-onnx Paraformer 不支持 ndarray final decode。"
            ) from exc

        if asr_result is None or (
            isinstance(asr_result, list) and len(asr_result) == 0
        ):
            return [{"text": ""}]

        text = extract_model_text(asr_result)
        if isinstance(asr_result, dict):
            asr_result.setdefault("text", text)
            return asr_result
        if isinstance(asr_result, list):
            if text and all(
                not (isinstance(item, dict) and item.get("text"))
                for item in asr_result
            ):
                return [{"text": text, "raw": asr_result}]
            return asr_result
        return [{"text": text, "raw": asr_result}]

    @staticmethod
    def _format_seaco_hotwords(hotwords: Any) -> str:
        if not hotwords:
            return ""
        if isinstance(hotwords, str):
            return hotwords.strip()
        words = []
        for item in hotwords:
            if isinstance(item, str):
                word = item.strip()
            elif isinstance(item, dict):
                word = str(item.get("hotword") or "").strip()
            elif isinstance(item, (list, tuple)) and item:
                word = str(item[-1] or "").strip()
            else:
                word = ""
            if word:
                words.append(word)
        return " ".join(words)


__all__ = ["ONNXFinalASRWrapper"]
