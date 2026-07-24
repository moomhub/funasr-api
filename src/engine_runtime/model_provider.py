"""Runtime model loading and cache coordination."""

from __future__ import annotations

from typing import Any, Optional

from src.core.config.errors import EngineConfigurationError
from src.engine_runtime.configuration import EngineRuntimeConfiguration
from src.engine_runtime.engines.offline.onnx import OfflineONNXModelBundle
from src.engine_runtime.engines.online import OnlineONNXModelBundle, OnlinePTModelBundle
from src.engine_runtime.state import RuntimeModelCache, build_runtime_loader_bundle


class RuntimeModelProvider:
    """Own model resolution, loader access, and loaded model state."""

    def __init__(self, config_loader: Any):
        self.config_loader = config_loader
        self.engines_config = config_loader.get_engines_config()
        self.processing_config = config_loader.get_processing_config()
        self.runtime_config = EngineRuntimeConfiguration(config_loader, self.engines_config)
        self.enabled_modes = self.runtime_config.enabled_engine_modes()

        self.model_dir = self.engines_config.model_dir
        self.auto_download = self.engines_config.auto_model_download
        self.pt_device = self.engines_config.device
        self.pt_disable_update = self.engines_config.disable_model_update
        self.onnx_quantize = True
        self.onnx_threads = 4
        self.onnx_device_id = -1
        self.offline_onnx = self.runtime_config.offline_onnx_options()

        self.loaders = build_runtime_loader_bundle(
            model_dir=self.model_dir,
            auto_download=self.auto_download,
            pt_device=self.pt_device,
            pt_disable_update=self.pt_disable_update,
            onnx_quantize=self.onnx_quantize,
            onnx_threads=self.onnx_threads,
            onnx_device_id=self.onnx_device_id,
        )
        self.model_cache = RuntimeModelCache()

    @property
    def downloader(self):
        return self.loaders.downloader

    @downloader.setter
    def downloader(self, value):
        self.loaders.downloader = value
        self.loaders.pt_loader.downloader = value
        self.loaders.online_onnx_loader.downloader = value
        self.loaders.spk_pt_loader.downloader = value

    @property
    def pt_loader(self):
        return self.loaders.pt_loader

    @pt_loader.setter
    def pt_loader(self, value):
        self.loaders.pt_loader = value
        self.model_cache.clear("online_pt_model_bundle")

    @property
    def online_onnx_loader(self):
        return self.loaders.online_onnx_loader

    @online_onnx_loader.setter
    def online_onnx_loader(self, value):
        self.loaders.online_onnx_loader = value
        self.model_cache.clear("online_onnx_model_bundle")

    @property
    def spk_pt_loader(self):
        return self.loaders.spk_pt_loader

    @spk_pt_loader.setter
    def spk_pt_loader(self, value):
        self.loaders.spk_pt_loader = value

    def get_backend_for_mode(self, mode: str) -> str:
        return self.runtime_config.backend_for_mode(mode)

    def model_name(self, mode: str, model_type: str, backend: Optional[str] = None) -> str:
        return self.runtime_config.model_name(mode, model_type, backend)

    def get_offline_model(self):
        if "offline" not in self.enabled_modes:
            raise EngineConfigurationError("OFFLINE 模式未启用")

        backend = self.get_backend_for_mode("offline")
        if backend == "onnx":
            return self.get_offline_onnx_model_bundle()

        asr_name = self.model_name("offline", "asr", backend)
        vad_name = self.model_name("offline", "vad", backend)
        punc_name = self.model_name("offline", "punc", backend)
        spk_name = self.model_name("offline", "spk", backend)
        return self.pt_loader.load_model(
            model_name=asr_name,
            vad_model=vad_name,
            punc_model=punc_name,
            spk_model=spk_name,
            cache_key=f"offline:{asr_name}:{vad_name}:{punc_name}:{spk_name}",
        )

    def get_offline_onnx_model_bundle(self) -> OfflineONNXModelBundle:
        if self.get_backend_for_mode("offline") != "onnx":
            raise EngineConfigurationError("当前 OFFLINE 后端不是 ONNX")

        def create_bundle() -> OfflineONNXModelBundle:
            asr_name = self.model_name("offline", "asr", "onnx")
            vad_name = self.model_name("offline", "vad", "onnx")
            punc_name = self.model_name("offline", "punc", "onnx")
            return OfflineONNXModelBundle(
                asr_model_dir=self.downloader.ensure_model(asr_name, required_files=["config.yaml"]),
                vad_model_dir=self.downloader.ensure_model(vad_name, required_files=["config.yaml"]),
                punc_model_dir=self.downloader.ensure_model(punc_name, required_files=["config.yaml"]),
                quantize=self.onnx_quantize,
                num_threads=self.onnx_threads,
                device_id=self.onnx_device_id,
                asr_workers=self.offline_onnx.asr_workers,
                load_workers=self.offline_onnx.load_workers,
                sample_rate=self.offline_onnx.sample_rate,
                vad_padding_ms=self.offline_onnx.vad_padding_ms,
            )

        return self.model_cache.get_or_create("offline_onnx_model_bundle", create_bundle)

    def get_online_pt_model_bundle(self) -> OnlinePTModelBundle:
        self._require_enabled_mode("online")
        if self.get_backend_for_mode("online") != "pt":
            raise EngineConfigurationError("当前 ONLINE 后端不是 PT")

        def create_bundle() -> OnlinePTModelBundle:
            streaming_asr = self.model_name("online", "streaming_asr", "pt")
            vad_name = self.model_name("online", "vad", "pt")
            final_asr = self.model_name("online", "final_asr", "pt")
            punc_name = self.model_name("online", "punc", "pt")
            return OnlinePTModelBundle(
                streaming_asr=self.pt_loader.load_single_model(
                    model_name=streaming_asr,
                    cache_key=f"online-pt-streaming-asr:{streaming_asr}",
                ),
                vad=self.pt_loader.load_single_model(
                    model_name=vad_name,
                    cache_key=f"online-pt-vad:{vad_name}",
                ),
                final_asr=self.pt_loader.load_single_model(
                    model_name=final_asr,
                    cache_key=f"online-pt-final-asr:{final_asr}",
                ),
                punc=(
                    self.pt_loader.load_single_model(
                        model_name=punc_name,
                        cache_key=f"online-pt-punc:{punc_name}",
                    )
                    if punc_name
                    else None
                ),
                metadata={"backend": "pt"},
            )

        return self.model_cache.get_or_create("online_pt_model_bundle", create_bundle)

    def get_online_onnx_model_bundle(self) -> OnlineONNXModelBundle:
        self._require_enabled_mode("online")
        if self.get_backend_for_mode("online") != "onnx":
            raise EngineConfigurationError("当前 ONLINE 后端不是 ONNX")
        return self.model_cache.get_or_create(
            "online_onnx_model_bundle",
            lambda: self.online_onnx_loader.load_models(
                streaming_asr_name=self.model_name("online", "streaming_asr", "onnx"),
                vad_name=self.model_name("online", "vad", "onnx"),
                final_asr_name=self.model_name("online", "final_asr", "onnx"),
                punc_name=self.model_name("online", "punc", "onnx"),
                chunk_size=self.processing_config.online_chunk_size,
            ),
        )

    def get_spk_model(self):
        if not self.spk_runtime_required():
            raise EngineConfigurationError("SPK runtime 未启用，需要启用 offline 或 spk")
        spk_name = self.model_name("spk", "spk", "pt")
        if not spk_name:
            raise EngineConfigurationError("SPK 模式必须配置 spk 模型")
        return self.spk_pt_loader.load_model(
            spk_name=spk_name,
            cache_key=f"spk-pt:{spk_name}",
        )

    def spk_runtime_required(self) -> bool:
        return "offline" in self.enabled_modes or "spk" in self.enabled_modes

    def get_model_for_mode(self, mode: str):
        if mode == "offline":
            return self.get_offline_model()
        if mode == "online":
            if self.get_backend_for_mode("online") == "pt":
                return self.get_online_pt_model_bundle()
            return self.get_online_onnx_model_bundle()
        if mode == "spk":
            return self.get_spk_model()
        raise ValueError(f"未知的模式: {mode}")

    def load_enabled_models(self) -> None:
        for mode in self.enabled_modes:
            self.get_model_for_mode(mode)

    def unload_model(self, model_key: str = None) -> None:
        self.pt_loader.unload_model(model_key)
        self.online_onnx_loader.unload_model(model_key)
        self.spk_pt_loader.unload_model(model_key)
        self._invalidate_bundle_cache(model_key)

    def get_loaded_models_count(self) -> int:
        return self.pt_loader.get_loaded_models_count() + self.get_loaded_onnx_models_count()

    def get_loaded_onnx_models_count(self) -> int:
        return self.model_cache.loaded_onnx_model_count(
            online_onnx_loader=self.online_onnx_loader,
            spk_pt_loader=self.spk_pt_loader,
        )

    def _require_enabled_mode(self, mode: str) -> None:
        if mode not in self.enabled_modes:
            raise EngineConfigurationError(f"{mode.upper()} 模式未启用")

    def _invalidate_bundle_cache(self, model_key: Optional[str]) -> None:
        if model_key is None:
            self.model_cache.clear()
            return
        if model_key.startswith("online-pt-"):
            self.model_cache.clear("online_pt_model_bundle")
        elif model_key in {"online-onnx", "online_onnx_model_bundle"}:
            self.model_cache.clear("online_onnx_model_bundle")
        elif model_key in {"offline-onnx", "offline_onnx_model_bundle"}:
            self.model_cache.clear("offline_onnx_model_bundle")


__all__ = ["RuntimeModelProvider"]
