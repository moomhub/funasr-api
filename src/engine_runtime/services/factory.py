"""Factory for runtime services and preload orchestration."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .contracts import (
    ModelPreloadStatus,
    OfflineAsrRequest,
    OfflineAsrService,
    OnlineAsrService,
    PreloadableService,
    ResultMergeService,
    ServiceHealth,
    SpeakerRequest,
    SpeakerService,
)
from .offline import ONNXOfflineAsrService, PTOfflineAsrService
from .online import ONNXOnlineAsrService, PTOnlineAsrService
from .service_plan import enabled_service_keys, required_service_modes
from .speaker import PTSpeakerService


class RuntimeServiceFactory:
    """Builds concrete services from config and coordinates model preloading."""

    def __init__(self, manager):
        self.manager = manager
        self._services: Dict[str, PreloadableService] = {}

    def offline_asr(self) -> OfflineAsrService:
        backend = self.manager.get_backend_for_mode("offline")
        if backend == "pt":
            return self._get_or_create("offline_asr", PTOfflineAsrService)
        if backend == "onnx":
            return self._get_or_create("offline_asr", ONNXOfflineAsrService)
        from src.core.config.errors import EngineConfigurationError

        raise EngineConfigurationError(f"Unsupported offline ASR backend: {backend}")

    def online_asr(self) -> OnlineAsrService:
        backend = self.manager.get_backend_for_mode("online")
        if backend == "pt":
            return self._get_or_create("online_asr", PTOnlineAsrService)
        if backend == "onnx":
            return self._get_or_create("online_asr", ONNXOnlineAsrService)
        from src.core.config.errors import EngineConfigurationError

        raise EngineConfigurationError(f"Unsupported online ASR backend: {backend}")

    def speaker(self) -> SpeakerService:
        return self._get_or_create("speaker", PTSpeakerService)

    def preload_enabled_models(self) -> List[ModelPreloadStatus]:
        return [service.preload() for service in self._enabled_services()]

    def get_model_status(self) -> Dict[str, Dict]:
        return {
            name: service.health().to_dict()
            for name, service in self._services.items()
        }

    def health(self) -> Dict[str, ServiceHealth]:
        return {
            name: service.health()
            for name, service in self._services.items()
        }

    def _enabled_services(self) -> Iterable[PreloadableService]:
        for key in enabled_service_keys(
            self.manager.enabled_modes,
            offline_backend=self._offline_backend(),
            offline_spk_verification_enabled=self._offline_spk_verification_enabled(),
        ):
            yield self._service_for_key(key)

    def required_service_modes(self, mode: str) -> List[str]:
        return required_service_modes(
            mode,
            offline_backend=self._offline_backend(),
            offline_spk_verification_enabled=self._offline_spk_verification_enabled(),
        )

    def _offline_backend(self) -> str:
        return str(self.manager.get_backend_for_mode("offline"))

    def _offline_spk_verification_enabled(self) -> bool:
        processing_config = getattr(self.manager, "processing_config", None)
        return bool(getattr(processing_config, "offline_spk_verification_enabled", True))

    def _service_for_key(self, key: str) -> PreloadableService:
        if key == "offline_asr":
            return self.offline_asr()
        if key == "online_asr":
            return self.online_asr()
        if key == "speaker":
            return self.speaker()
        raise KeyError(key)

    def _get_or_create(self, key: str, service_cls):
        if key not in self._services:
            self._services[key] = service_cls(self.manager)
        return self._services[key]


__all__ = ["RuntimeServiceFactory"]
