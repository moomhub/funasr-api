"""Runtime directory resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict


class RuntimePathResolver:
    def __init__(self, config: Dict[str, Any], get_env: Callable[..., Any]):
        self.config = config
        self.get_env = get_env

    def root_dir(self) -> str:
        runtime_cfg = self.config.get("runtime", {}) or {}
        return self.get_env("runtime.root_dir", "RUNTIME_ROOT_DIR", runtime_cfg.get("root_dir", "./data"))

    def resolve(self, configured_path: Any, default_relative: str, env_var: str = None) -> str:
        value = self.get_env("runtime.path", env_var, configured_path) if env_var else configured_path
        path = Path(str(value or default_relative).strip())
        return str(path) if path.is_absolute() else str(Path(self.root_dir()) / path)

    def all_paths(self) -> Dict[str, str]:
        runtime_cfg = self.config.get("runtime", {}) or {}
        return {
            "root_dir": self.root_dir(),
            "models_dir": self.resolve(runtime_cfg.get("models_dir"), "models", "RUNTIME_MODELS_DIR"),
            "sqlite_path": self.resolve(runtime_cfg.get("sqlite_path"), "sqlite/funasr_tasks.db", "RUNTIME_SQLITE_PATH"),
            "temp_dir": self.resolve(runtime_cfg.get("temp_dir"), "temp", "RUNTIME_TEMP_DIR"),
            "offline_result_dir": self.resolve(runtime_cfg.get("offline_result_dir"), "results/offline", "RUNTIME_OFFLINE_RESULT_DIR"),
            "local_files_dir": self.resolve(runtime_cfg.get("local_files_dir"), "files", "RUNTIME_LOCAL_FILES_DIR"),
        }


__all__ = ["RuntimePathResolver"]
