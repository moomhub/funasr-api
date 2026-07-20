"""Application-level runtime facade for preload and health reporting."""

from __future__ import annotations

from typing import Any, Dict, Optional

class RuntimeApplication:
    """Coordinates runtime preload and status reporting at the application layer."""

    def __init__(
        self,
        manager: Any = None,
        runtime_services: Any = None,
    ):
        if manager is None or runtime_services is None:
            raise ValueError("manager and runtime_services are required")
        self.manager = manager
        self.runtime_services = runtime_services
        self._mode_availability: Dict[str, bool] = {
            mode: True
            for mode in self.manager.enabled_modes
        }

    def preload_enabled_models(self) -> Dict[str, Dict[str, Optional[str]]]:
        loaded: list[str] = []
        failed: Dict[str, str] = {}
        statuses = list(self.runtime_services.preload_enabled_models())
        statuses_by_mode: Dict[str, list] = {}

        for status in statuses:
            statuses_by_mode.setdefault(status.mode, []).append(status)

        availability: Dict[str, bool] = {}
        for mode in self.manager.enabled_modes:
            required_modes = self.runtime_services.required_service_modes(mode)
            errors = []
            for required_mode in required_modes:
                required_statuses = statuses_by_mode.get(required_mode, [])
                if not required_statuses:
                    errors.append(f"{required_mode}: runtime preload status missing")
                    continue
                errors.extend(
                    f"{status.service_name}: {status.error or 'runtime preload failed'}"
                    for status in required_statuses
                    if not status.loaded
                )
            if errors:
                availability[mode] = False
                failed[mode] = "; ".join(errors)
            else:
                availability[mode] = True
                loaded.append(mode)

        self._mode_availability = availability

        return {
            "loaded": loaded,
            "failed": failed,
        }

    def get_enabled_modes(self) -> list[str]:
        return list(self.manager.enabled_modes)

    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            mode: {
                "backend": self.manager.get_backend_for_mode(mode),
                "enabled": mode in self.manager.enabled_modes,
                "available": self.is_mode_available(mode),
            }
            for mode in ("offline", "online", "spk")
        }

    def is_mode_available(self, mode: str) -> bool:
        if mode not in self.manager.enabled_modes:
            return False
        return bool(self._mode_availability.get(mode, True))

    def get_runtime_status(self) -> Dict[str, Dict[str, Any]]:
        return self.runtime_services.get_model_status()

    def get_inference_backends(self) -> Dict[str, str]:
        return self.manager.get_inference_backends()

    def get_loaded_models_count(self) -> int:
        return self.manager.get_loaded_models_count()


__all__ = [
    "RuntimeApplication",
]
