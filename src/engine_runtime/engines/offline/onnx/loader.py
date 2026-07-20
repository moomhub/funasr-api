"""Model loading helpers for offline ONNX ASR bundles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, List

from src.core.config.errors import ModelLoadError

logger = logging.getLogger(__name__)

ASR_HOTWORD_MODE_PLAIN = "plain"
ASR_HOTWORD_MODE_SEACO = "seaco"


@dataclass(frozen=True)
class OfflineONNXLoadedModels:
    vad_model: Any
    punc_model: Any
    asr_models: List[Any]
    asr_hotword_mode: str


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
        logger.info("OFFLINE ONNX SeACo 未提供完整量化模型，使用非量化模型")
    if model_path.exists():
        missing = [
            name
            for name in ("model.onnx", "model_eb.onnx")
            if not (model_path / name).is_file()
        ]
        if missing:
            raise ModelLoadError(
                "OFFLINE ONNX SeACo 模型不完整，缺少: " + ", ".join(missing)
            )
    return False


def load_offline_onnx_models(
    *,
    asr_model_dir: str,
    vad_model_dir: str,
    punc_model_dir: str,
    quantize: bool,
    num_threads: int,
    device_id: int,
    asr_workers: int,
    load_workers: int,
) -> OfflineONNXLoadedModels:
    import funasr_onnx

    asr_hotword_mode = ASR_HOTWORD_MODE_SEACO if _is_seaco_model(asr_model_dir) else ASR_HOTWORD_MODE_PLAIN
    asr_model_factory = funasr_onnx.Paraformer
    asr_quantize = quantize
    if asr_hotword_mode == ASR_HOTWORD_MODE_SEACO:
        asr_model_factory = getattr(funasr_onnx, "SeacoParaformer", None)
        if asr_model_factory is None:
            raise ModelLoadError("当前 funasr-onnx 未提供 SeacoParaformer")
        asr_quantize = _seaco_quantize_mode(asr_model_dir, quantize)

    common_kwargs = {
        "quantize": quantize,
        "intra_op_num_threads": num_threads,
        "device_id": device_id,
    }
    asr_kwargs = {
        **common_kwargs,
        "quantize": asr_quantize,
    }
    asr_models: List[Any] = [None] * asr_workers
    vad_model = None
    punc_model = None

    with ThreadPoolExecutor(max_workers=load_workers) as executor:
        future_map = {
            executor.submit(funasr_onnx.Fsmn_vad, model_dir=vad_model_dir, **common_kwargs): "vad",
            executor.submit(funasr_onnx.CT_Transformer, model_dir=punc_model_dir, **common_kwargs): "punc",
        }
        for index in range(asr_workers):
            future_map[executor.submit(asr_model_factory, model_dir=asr_model_dir, **asr_kwargs)] = f"asr:{index}"

        try:
            for future in as_completed(future_map):
                label = future_map[future]
                model = future.result()
                if label == "vad":
                    vad_model = model
                elif label == "punc":
                    punc_model = model
                elif label.startswith("asr:"):
                    asr_models[int(label.split(":", 1)[1])] = model
        except Exception as exc:
            raise ModelLoadError(f"OFFLINE ONNX 模型加载失败: {exc}") from exc

    loaded_asr_models = [model for model in asr_models if model is not None]
    if not vad_model or not punc_model or not loaded_asr_models:
        raise ModelLoadError("OFFLINE ONNX 模型未完整加载，缺少 VAD/PUNC/ASR")

    return OfflineONNXLoadedModels(
        vad_model=vad_model,
        punc_model=punc_model,
        asr_models=loaded_asr_models,
        asr_hotword_mode=asr_hotword_mode,
    )


__all__ = [
    "ASR_HOTWORD_MODE_PLAIN",
    "ASR_HOTWORD_MODE_SEACO",
    "OfflineONNXLoadedModels",
    "load_offline_onnx_models",
]
