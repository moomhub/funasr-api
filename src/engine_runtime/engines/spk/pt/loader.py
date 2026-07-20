"""PT speaker diarization pipeline loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.core.config.errors import ModelLoadError
from src.core.models import ModelDownloader

logger = logging.getLogger(__name__)


class SpeakerPTPipelineLoader:
    """Loads standalone PT speaker diarization through ModelScope pipeline."""

    def __init__(self, downloader: ModelDownloader):
        self.downloader = downloader
        self._models: dict[str, Any] = {}

    def load_model(self, *, spk_name: str, cache_key: Optional[str] = None) -> Any:
        cache_key = cache_key or f"spk-pt:{spk_name}"
        if cache_key in self._models:
            return self._models[cache_key]

        model_path = self.downloader.ensure_model(spk_name, prefer_repo_id=True)
        self._validate_diarization_model(model_path, spk_name)

        try:
            from modelscope.pipelines import pipeline

            model = pipeline(task="speaker-diarization", model=model_path)
        except Exception as exc:
            raise ModelLoadError(f"PT SPK pipeline 加载失败: {cache_key}") from exc

        self._models[cache_key] = model
        return model

    def unload_model(self, model_key: str = None) -> None:
        if model_key:
            self._models.pop(model_key, None)
            return
        self._models.clear()

    def get_loaded_models_count(self) -> int:
        return len(self._models)

    @staticmethod
    def _read_model_task(model_path: str) -> str:
        config_path = Path(model_path) / "configuration.json"
        if not config_path.exists():
            return ""

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return ""

        return str(data.get("task") or "").strip()

    @staticmethod
    def _validate_diarization_model(model_path: str, spk_name: str) -> None:
        task = SpeakerPTPipelineLoader._read_model_task(model_path)
        if task and task != "speaker-diarization":
            raise ModelLoadError(
                "Standalone SPK 仅支持 speaker-diarization 模型，"
                f"当前模型任务为: {task} ({spk_name})"
            )
