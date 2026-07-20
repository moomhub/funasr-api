"""Runtime engine configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.core.config.errors import EngineConfigurationError
from src.engine_runtime.engines.offline.catalog import DEFAULT_MODELS as OFFLINE_DEFAULT_MODELS
from src.engine_runtime.engines.online.catalog import DEFAULT_MODELS as ONLINE_DEFAULT_MODELS
from src.engine_runtime.engines.spk.catalog import DEFAULT_MODELS as SPK_DEFAULT_MODELS


DEFAULT_MODELS_BY_MODE = {
    "offline": OFFLINE_DEFAULT_MODELS,
    "online": ONLINE_DEFAULT_MODELS,
    "spk": SPK_DEFAULT_MODELS,
}
ENGINE_MODES = {"offline", "online", "spk"}


@dataclass(frozen=True)
class OfflineOnnxRuntimeOptions:
    asr_workers: int = 2
    load_workers: int = 4
    sample_rate: int = 16000
    vad_padding_ms: int = 300


class EngineRuntimeConfiguration:
    """Interpret engine config without owning model loading or caches."""

    def __init__(self, config_loader: Any, engines_config: Any):
        self.config_loader = config_loader
        self.engines_config = engines_config

    def enabled_engine_modes(self) -> list[str]:
        enabled_modes = list(self.engines_config.enabled)
        unsupported = [mode for mode in enabled_modes if mode not in ENGINE_MODES]
        if unsupported:
            raise EngineConfigurationError(
                f"engines.enabled 只能配置 offline/online/spk，非法值: {', '.join(unsupported)}"
            )
        return enabled_modes

    def backend_for_mode(self, mode: str) -> str:
        if mode == "spk":
            return "pt"
        mode_config = self.mode_config(mode)
        backend = getattr(mode_config, "enabled", None)
        if backend not in {"pt", "onnx"}:
            raise EngineConfigurationError(
                f"{mode.upper()} backend 配置非法: {backend}，仅支持 pt/onnx"
            )
        return backend

    def model_name(self, mode: str, model_type: str, backend: Optional[str] = None) -> str:
        if mode == "spk":
            mode_config = self.mode_config(mode)
            configured_name = getattr(mode_config, model_type, None)
            if configured_name:
                return configured_name
            return DEFAULT_MODELS_BY_MODE.get(mode, {}).get("pt", {}).get(model_type, "")

        backend = backend or self.backend_for_mode(mode)
        mode_config = self.mode_config(mode)
        variant_config = getattr(mode_config, backend)
        configured_name = getattr(variant_config, model_type, None)
        if configured_name:
            return configured_name
        return DEFAULT_MODELS_BY_MODE.get(mode, {}).get(backend, {}).get(model_type, "")

    def mode_config(self, mode: str) -> Any:
        if mode not in ENGINE_MODES:
            raise EngineConfigurationError(f"配置的模式不支持: {mode}，仅支持: offline, online, spk")
        return getattr(self.engines_config.models, mode)

    def offline_onnx_options(self) -> OfflineOnnxRuntimeOptions:
        offline_onnx_cfg = (
            self.config_loader.config_dict
            .get("engines", {})
            .get("models", {})
            .get("offline", {})
            .get("onnx_runtime", {})
        )

        def int_setting(name: str, default: int) -> int:
            try:
                return int(offline_onnx_cfg.get(name, default))
            except (TypeError, ValueError) as exc:
                raise EngineConfigurationError(
                    f"OFFLINE ONNX 运行时配置非法：{name} 必须是整数"
                ) from exc

        options = OfflineOnnxRuntimeOptions(
            asr_workers=int_setting("asr_workers", 2),
            load_workers=int_setting("load_workers", 4),
            sample_rate=int_setting("sample_rate", 16000),
            vad_padding_ms=int_setting("vad_padding_ms", 300),
        )
        if options.asr_workers < 1 or options.load_workers < 1:
            raise EngineConfigurationError("OFFLINE ONNX 运行时配置非法：workers 必须 >= 1")
        if options.sample_rate < 1:
            raise EngineConfigurationError("OFFLINE ONNX 运行时配置非法：sample_rate 必须 >= 1")
        if options.vad_padding_ms < 0:
            raise EngineConfigurationError("OFFLINE ONNX 运行时配置非法：vad_padding_ms 必须 >= 0")
        return options


__all__ = [
    "ENGINE_MODES",
    "DEFAULT_MODELS_BY_MODE",
    "EngineRuntimeConfiguration",
    "OfflineOnnxRuntimeOptions",
]
