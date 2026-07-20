"""Model cache detection and optional download handling."""

import logging
import os
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Dict, Iterable, Optional

from src.core.config.errors import ModelLoadError, ModelResolutionError

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Resolves model identifiers to local cache paths and downloads when needed."""

    def __init__(self, model_dir: str, auto_download: bool = True):
        self.model_dir = Path(model_dir).absolute()
        # ModelScope appends its own ``models/<namespace>/<repo>`` suffix to
        # cache_dir.  When our configured model root is already named
        # ``models``, pass its parent to avoid ``models/models``.
        self.modelscope_cache_dir = (
            self.model_dir.parent
            if self.model_dir.name.lower() == "models"
            else self.model_dir
        )
        self.auto_download = auto_download
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.modelscope_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = str(self.modelscope_cache_dir)
        self._local_model_paths: Dict[str, str] = {}
        self.scan_local_models()

    def scan_local_models(self) -> Dict[str, str]:
        self._local_model_paths = {}
        search_roots = [self.model_dir]
        if self.model_dir.name.lower() != "models":
            search_roots.append(self.model_dir / "models")
        for models_root in search_roots:
            if not models_root.exists():
                continue
            for namespace_dir in models_root.iterdir():
                if not namespace_dir.is_dir():
                    continue
                for repo_dir in namespace_dir.iterdir():
                    if not repo_dir.is_dir() or not self._is_valid_model_dir(repo_dir):
                        continue
                    model_id = f"{namespace_dir.name}/{repo_dir.name}"
                    self._local_model_paths.setdefault(model_id, str(repo_dir))
        return dict(self._local_model_paths)

    def ensure_model(
        self,
        model_name: str,
        *,
        required_files: Optional[Iterable[str]] = None,
        prefer_repo_id: bool = False,
    ) -> str:
        if not model_name:
            return ""

        if self._looks_like_path(model_name):
            self._validate_existing_path(model_name)
            self._validate_required_files(model_name, required_files)
            return model_name

        repo_id = model_name
        cached_path = self._local_model_paths.get(repo_id)
        if cached_path:
            self._validate_required_files(cached_path, required_files)
            return cached_path

        if not self.auto_download:
            raise ModelResolutionError(
                f"模型 {model_name}({repo_id}) 未在本地缓存中找到，且 auto_download=False。"
            )

        downloaded = self._download(repo_id)
        self.scan_local_models()
        resolved_path = self._local_model_paths.get(repo_id, downloaded)
        if required_files and not Path(resolved_path).exists():
            raise ModelResolutionError(
                f"模型 {model_name}({repo_id}) 需要本地模型目录，但未能解析到有效路径: {resolved_path}。"
            )
        self._validate_required_files(resolved_path, required_files)
        if prefer_repo_id and not Path(resolved_path).exists():
            return repo_id
        return resolved_path

    def _download(self, repo_id: str) -> str:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as exc:
            logger.warning("ModelScope snapshot_download 不可用，将回退到仓库 ID: %s", repo_id)
            return repo_id

        logger.info("🔄 下载模型到本地缓存: %s", repo_id)
        try:
            downloaded = snapshot_download(repo_id, cache_dir=str(self.modelscope_cache_dir))
        except TypeError:
            downloaded = snapshot_download(
                repo_id,
                cache_dir=str(self.modelscope_cache_dir),
                local_files_only=False,
            )
        except Exception as exc:
            raise ModelLoadError(f"模型下载失败: {repo_id}") from exc
        logger.info("Model download completed: repository=%s", repo_id)
        logger.debug("Model download path: repository=%s path=%s", repo_id, downloaded)
        return downloaded

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return (
            value.startswith("./")
            or value.startswith("../")
            or value.startswith("/")
            or value.startswith("~/")
            or "\\" in value
            or Path(value).is_absolute()
            or bool(PureWindowsPath(value).drive)
        )

    @staticmethod
    def _validate_existing_path(model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise ModelResolutionError(f"模型路径不存在: {path}。")

    @staticmethod
    def _is_valid_model_dir(path: Path) -> bool:
        markers = ["model.pt", "configuration.json", "config.yaml", "pytorch_model.bin"]
        if any((path / marker).exists() for marker in markers):
            return True
        return any(path.glob("*.onnx"))

    @staticmethod
    def _validate_required_files(model_path: str, required_files: Optional[Iterable[str]]) -> None:
        if not required_files:
            return
        path = Path(model_path)
        if not path.exists():
            return
        missing = [name for name in required_files if not (path / name).exists()]
        if missing:
            raise ModelResolutionError(
                f"模型目录不完整: {path}，缺少: {', '.join(missing)}。"
            )
