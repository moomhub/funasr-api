"""Runtime loader bundle and cache state objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.core.models import ModelDownloader
from src.engine_runtime.engines.online.onnx.loader import OnlineONNXModelLoader
from src.engine_runtime.engines.spk.pt.loader import SpeakerPTPipelineLoader
from src.engine_runtime.loaders import PTModelLoader


@dataclass
class RuntimeLoaderBundle:
    downloader: Any
    pt_loader: Any
    online_onnx_loader: Any
    spk_pt_loader: Any


@dataclass
class RuntimeModelCache:
    offline_onnx_model_bundle: Any = None
    online_pt_model_bundle: Any = None
    online_onnx_model_bundle: Any = None

    def get_or_create(self, name: str, factory: Callable[[], Any]) -> Any:
        current = getattr(self, name)
        if current is None:
            current = factory()
            setattr(self, name, current)
        return current

    def clear(self, name: str | None = None) -> None:
        names = (name,) if name else (
            "offline_onnx_model_bundle",
            "online_pt_model_bundle",
            "online_onnx_model_bundle",
        )
        for cache_name in names:
            setattr(self, cache_name, None)

    def loaded_onnx_model_count(self, *, online_onnx_loader: Any, spk_pt_loader: Any) -> int:
        offline_count = 3 if self.offline_onnx_model_bundle is not None else 0
        return (
            offline_count
            + online_onnx_loader.get_loaded_models_count()
            + spk_pt_loader.get_loaded_models_count()
        )


@dataclass
class RuntimeRecognizerCache:
    offline_pt_recognizer: Any = None
    offline_onnx_recognizer: Any = None
    online_pt_recognizer: Any = None
    online_onnx_recognizer: Any = None
    spk_pt_recognizer: Any = None

    def get_or_create(self, name: str, factory: Callable[[], Any]) -> Any:
        current = getattr(self, name)
        if current is None:
            current = factory()
            setattr(self, name, current)
        return current


def build_runtime_loader_bundle(
    *,
    model_dir: str,
    auto_download: bool,
    pt_device: str,
    pt_disable_update: bool,
    onnx_quantize: bool,
    onnx_threads: int,
    onnx_device_id: int,
) -> RuntimeLoaderBundle:
    downloader = ModelDownloader(
        model_dir=model_dir,
        auto_download=auto_download,
    )
    return RuntimeLoaderBundle(
        downloader=downloader,
        pt_loader=PTModelLoader(
            downloader=downloader,
            device=pt_device,
            disable_update=pt_disable_update,
        ),
        online_onnx_loader=OnlineONNXModelLoader(
            downloader=downloader,
            quantize=onnx_quantize,
            num_threads=onnx_threads,
            device_id=onnx_device_id,
        ),
        spk_pt_loader=SpeakerPTPipelineLoader(
            downloader=downloader,
        ),
    )


__all__ = [
    "RuntimeLoaderBundle",
    "RuntimeModelCache",
    "RuntimeRecognizerCache",
    "build_runtime_loader_bundle",
]
