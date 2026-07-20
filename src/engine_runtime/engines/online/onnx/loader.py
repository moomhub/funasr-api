"""ONLINE ONNX model loading."""

from typing import Any, Dict, Optional

from .final_asr import ONNXFinalASRWrapper
from .punctuation import ONNXPuncWrapper
from .streaming_asr import ONNXStreamingASRWrapper
from .vad import ONNXVADWrapper
from src.core.config.errors import ModelLoadError
from src.core.models import ModelDownloader
from src.engine_runtime.engines.online.base import OnlineONNXModelBundle
from .catalog import REQUIRED_FILES


class OnlineONNXModelLoader:
    """Loads the ONNX models used by ONLINE realtime recognition."""

    def __init__(
        self,
        downloader: ModelDownloader,
        quantize: bool = True,
        num_threads: int = 4,
        device_id: int = -1,
    ):
        self.downloader = downloader
        self.quantize = quantize
        self.num_threads = num_threads
        self.device_id = device_id
        self._models: Dict[str, Any] = {}

    def load_models(
        self,
        *,
        streaming_asr_name: str,
        vad_name: str,
        final_asr_name: str,
        punc_name: str,
        chunk_size: Optional[list] = None,
        cache_key: str = "online-onnx",
    ) -> OnlineONNXModelBundle:
        if cache_key in self._models:
            return self._models[cache_key]

        try:
            bundle = OnlineONNXModelBundle(
                streaming_asr=ONNXStreamingASRWrapper(
                    self.downloader.ensure_model(
                        streaming_asr_name,
                        required_files=REQUIRED_FILES["streaming_asr"],
                    ),
                    quantize=self.quantize,
                    num_threads=self.num_threads,
                    device_id=self.device_id,
                    chunk_size=chunk_size,
                ),
                vad=ONNXVADWrapper(
                    self.downloader.ensure_model(
                        vad_name,
                        required_files=REQUIRED_FILES["vad"],
                    ),
                    quantize=self.quantize,
                    num_threads=self.num_threads,
                    device_id=self.device_id,
                ),
                final_asr=ONNXFinalASRWrapper(
                    self.downloader.ensure_model(
                        final_asr_name,
                        required_files=REQUIRED_FILES["final_asr"],
                    ),
                    quantize=self.quantize,
                    num_threads=self.num_threads,
                    device_id=self.device_id,
                ),
                punc=ONNXPuncWrapper(
                    self.downloader.ensure_model(
                        punc_name,
                        required_files=REQUIRED_FILES["punc"],
                    ),
                    quantize=self.quantize,
                    num_threads=self.num_threads,
                    device_id=self.device_id,
                ) if punc_name else None,
                metadata={"backend": "onnx"},
            )
        except Exception as exc:
            raise ModelLoadError(f"ONNX ONLINE 模型堆栈加载失败: {cache_key}") from exc
        self._models[cache_key] = bundle
        return self._models[cache_key]

    def unload_model(self, model_key: str = None) -> None:
        if model_key:
            self._models.pop(model_key, None)
            return
        self._models.clear()

    def get_loaded_models_count(self) -> int:
        return sum(
            sum(
                1
                for key, model in value.as_dict().items()
                if key != "metadata" and model is not None
            )
            for value in self._models.values()
        )
