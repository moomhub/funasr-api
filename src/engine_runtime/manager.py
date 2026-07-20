"""Recognizer facade for the engine runtime."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.config.errors import EngineConfigurationError
from src.engine_runtime.engines.offline import BaseOfflineRecognizer
from src.engine_runtime.engines.offline.onnx import OfflineONNXModelBundle, OfflineONNXRecognizer
from src.engine_runtime.engines.offline.pt import PTOfflineRecognizer
from src.engine_runtime.engines.online import BaseOnlineRecognizer, OnlineONNXModelBundle, OnlinePTModelBundle
from src.engine_runtime.engines.online.onnx import ONNXOnlineRecognizer
from src.engine_runtime.engines.online.pt import PTOnlineRecognizer
from src.engine_runtime.engines.spk import BaseSpeakerRecognizer
from src.engine_runtime.engines.spk.pt import PTSpeakerRecognizer
from src.engine_runtime.model_provider import RuntimeModelProvider
from src.engine_runtime.state import RuntimeRecognizerCache

logger = logging.getLogger(__name__)


class EngineModelManager:
    """Stable runtime facade used by recognizers and application services."""

    def __init__(self, config_loader: Any):
        self.config_loader = config_loader
        self.model_provider = RuntimeModelProvider(config_loader)
        self.recognizer_cache = RuntimeRecognizerCache()

        # Compatibility attributes retained for current recognizers and callers.
        self.engines_config = self.model_provider.engines_config
        self.processing_config = self.model_provider.processing_config
        self.runtime_config = self.model_provider.runtime_config
        self.enabled_modes = self.model_provider.enabled_modes
        self.model_dir = self.model_provider.model_dir
        self.auto_download = self.model_provider.auto_download
        self.pt_device = self.model_provider.pt_device
        self.pt_disable_update = self.model_provider.pt_disable_update
        self.onnx_quantize = self.model_provider.onnx_quantize
        self.onnx_threads = self.model_provider.onnx_threads
        self.onnx_device_id = self.model_provider.onnx_device_id
        self.offline_onnx = self.model_provider.offline_onnx
        self.loaders = self.model_provider.loaders
        self.model_cache = self.model_provider.model_cache

        logger.info(
            "模型运行时初始化完成: enabled_modes=%s auto_download=%s",
            self.enabled_modes,
            self.auto_download,
        )
        logger.debug(
            "模型运行时详细配置: model_dir=%s device=%s disable_update=%s "
            "onnx_quantize=%s onnx_threads=%s onnx_device_id=%s",
            self.model_dir,
            self.pt_device,
            self.pt_disable_update,
            self.onnx_quantize,
            self.onnx_threads,
            self.onnx_device_id,
        )

    @property
    def downloader(self):
        return self.model_provider.downloader

    @downloader.setter
    def downloader(self, value):
        self.model_provider.downloader = value

    @property
    def pt_loader(self):
        return self.model_provider.pt_loader

    @pt_loader.setter
    def pt_loader(self, value):
        self.model_provider.pt_loader = value

    @property
    def online_onnx_loader(self):
        return self.model_provider.online_onnx_loader

    @online_onnx_loader.setter
    def online_onnx_loader(self, value):
        self.model_provider.online_onnx_loader = value

    @property
    def spk_pt_loader(self):
        return self.model_provider.spk_pt_loader

    @spk_pt_loader.setter
    def spk_pt_loader(self, value):
        self.model_provider.spk_pt_loader = value

    def get_backend_for_mode(self, mode: str) -> str:
        return self.model_provider.get_backend_for_mode(mode)

    def _get_model_name(
        self,
        mode: str,
        model_type: str,
        backend: Optional[str] = None,
    ) -> str:
        return self.model_provider.model_name(mode, model_type, backend)

    def get_offline_model(self):
        return self.model_provider.get_offline_model()

    def get_offline_onnx_model_bundle(self) -> OfflineONNXModelBundle:
        return self.model_provider.get_offline_onnx_model_bundle()

    def get_online_pt_model_bundle(self) -> OnlinePTModelBundle:
        return self.model_provider.get_online_pt_model_bundle()

    def get_online_onnx_model_bundle(self) -> OnlineONNXModelBundle:
        return self.model_provider.get_online_onnx_model_bundle()

    def get_spk_model(self):
        return self.model_provider.get_spk_model()

    def _spk_runtime_required(self) -> bool:
        return self.model_provider.spk_runtime_required()

    def get_offline_pt_recognizer(self) -> PTOfflineRecognizer:
        if self.get_backend_for_mode("offline") != "pt":
            raise EngineConfigurationError("当前 OFFLINE 后端不是 PyTorch")
        return self.recognizer_cache.get_or_create(
            "offline_pt_recognizer",
            lambda: PTOfflineRecognizer(self),
        )

    def get_offline_onnx_recognizer(self) -> OfflineONNXRecognizer:
        if self.get_backend_for_mode("offline") != "onnx":
            raise EngineConfigurationError("当前 OFFLINE 后端不是 ONNX")
        return self.recognizer_cache.get_or_create(
            "offline_onnx_recognizer",
            lambda: OfflineONNXRecognizer(self),
        )

    def get_offline_recognizer(self) -> BaseOfflineRecognizer:
        if self.get_backend_for_mode("offline") == "pt":
            return self.get_offline_pt_recognizer()
        return self.get_offline_onnx_recognizer()

    def get_online_pt_recognizer(self) -> PTOnlineRecognizer:
        if self.get_backend_for_mode("online") != "pt":
            raise EngineConfigurationError("当前 ONLINE 后端不是 PT")
        return self.recognizer_cache.get_or_create(
            "online_pt_recognizer",
            lambda: PTOnlineRecognizer(self),
        )

    def get_online_onnx_recognizer(self) -> ONNXOnlineRecognizer:
        if self.get_backend_for_mode("online") != "onnx":
            raise EngineConfigurationError("当前 ONLINE 后端不是 ONNX")
        return self.recognizer_cache.get_or_create(
            "online_onnx_recognizer",
            lambda: ONNXOnlineRecognizer(self),
        )

    def get_online_recognizer(self) -> BaseOnlineRecognizer:
        if self.get_backend_for_mode("online") == "pt":
            return self.get_online_pt_recognizer()
        return self.get_online_onnx_recognizer()

    def get_spk_pt_recognizer(self) -> PTSpeakerRecognizer:
        return self.recognizer_cache.get_or_create(
            "spk_pt_recognizer",
            lambda: PTSpeakerRecognizer(self),
        )

    def get_spk_recognizer(self) -> BaseSpeakerRecognizer:
        return self.get_spk_pt_recognizer()

    def get_model_for_mode(self, mode: str):
        return self.model_provider.get_model_for_mode(mode)

    def load_enabled_models(self) -> None:
        self.model_provider.load_enabled_models()

    def unload_model(self, model_key: str = None) -> None:
        self.model_provider.unload_model(model_key)

    def get_loaded_models_count(self) -> int:
        return self.model_provider.get_loaded_models_count()

    def get_loaded_onnx_models_count(self) -> int:
        return self.model_provider.get_loaded_onnx_models_count()

    def get_inference_backend(self, mode: Optional[str] = None) -> str:
        effective_mode = mode or (self.enabled_modes[0] if self.enabled_modes else "online")
        backend = self.get_backend_for_mode(effective_mode)
        return "onnx" if backend == "onnx" else "pytorch"

    def get_inference_backends(self) -> Dict[str, str]:
        return {
            mode: self.get_inference_backend(mode)
            for mode in self.enabled_modes
        }


__all__ = ["EngineModelManager"]
