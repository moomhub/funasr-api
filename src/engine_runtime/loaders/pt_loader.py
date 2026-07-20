"""PyTorch model loader."""

import logging
from typing import Any, Dict, Optional

from src.core.models import ModelDownloader
from src.core.config.errors import ModelLoadError

logger = logging.getLogger(__name__)


class PTModelLoader:
    def __init__(self, downloader: ModelDownloader, device: str = "cpu", disable_update: bool = True):
        self.downloader = downloader
        self.device = device
        self.disable_update = disable_update
        self._models: Dict[str, Any] = {}

    def load_single_model(self, model_name: str, cache_key: str) -> Any:
        if cache_key in self._models:
            return self._models[cache_key]
        from funasr import AutoModel

        resolved = self.downloader.ensure_model(model_name, prefer_repo_id=True)
        try:
            model = AutoModel(
                model=resolved,
                device=self.device,
                disable_update=self.disable_update,
            )
        except Exception as exc:
            raise ModelLoadError(f"PT 单模型加载失败: {model_name}") from exc
        self._models[cache_key] = model
        return model

    def load_model(
        self,
        model_name: str,
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        spk_model: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> Any:
        cache_key = cache_key or f"{model_name}:{vad_model}:{punc_model}:{spk_model}"
        if cache_key in self._models:
            return self._models[cache_key]
        from funasr import AutoModel
        
        try:
            model = AutoModel(
                model=self.downloader.ensure_model(model_name, prefer_repo_id=True),
                vad_model=self.downloader.ensure_model(vad_model, prefer_repo_id=True) if vad_model else None,
                punc_model=self.downloader.ensure_model(punc_model, prefer_repo_id=True) if punc_model else None,
                spk_model=self.downloader.ensure_model(spk_model, prefer_repo_id=True) if spk_model else None,
                device=self.device,
                disable_update=self.disable_update,
            )
        except Exception as exc:
            raise ModelLoadError(f"PT 模型堆栈加载失败: {cache_key}") from exc
        self._models[cache_key] = model
        return model

    def unload_model(self, model_key: str = None) -> None:
        if model_key:
            self._models.pop(model_key, None)
            return
        self._models.clear()

    def get_loaded_models_count(self) -> int:
        return len(self._models)
