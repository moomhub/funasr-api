"""Local temporary upload storage adapter."""

from __future__ import annotations

import asyncio
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Tuple

from .status import component_status


class LocalTempFileStore:
    name = "local"
    enabled = True

    def __init__(self, temp_dir: str):
        self.root = Path(temp_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.available = True
        self.last_error = None

    async def save_upload(
        self,
        upload_file: Any,
        task_id: str,
        max_size: int,
    ) -> Tuple[Path, int]:
        filename = Path(upload_file.filename or "audio").name
        task_dir = self.root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        file_path = task_dir / filename
        file_size = 0
        chunk_size = 1024 * 1024

        try:
            with file_path.open("wb") as target:
                while True:
                    chunk = await upload_file.read(chunk_size)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > max_size:
                        self._remove_partial_upload(file_path, task_dir)
                        raise ValueError(
                            f"文件过大: 超过限制 {max_size / 1024 / 1024:.0f} MB"
                        )
                    await asyncio.to_thread(target.write, chunk)
            self.available = True
            self.last_error = None
            return file_path, file_size
        except Exception as exc:
            self.last_error = str(exc)
            raise

    def resolve(self, task_id: str, filename: str) -> Path:
        return self.root / task_id / Path(filename).name

    def exists(self, task_id: str, filename: str) -> bool:
        return self.resolve(task_id, filename).exists()

    def cleanup(self, task_id: str) -> None:
        shutil.rmtree(self.root / task_id, ignore_errors=True)

    def status(self) -> Dict[str, Any]:
        return component_status(self, {"root": str(self.root)})

    @staticmethod
    def _remove_partial_upload(file_path: Path, task_dir: Path) -> None:
        with suppress(OSError):
            file_path.unlink(missing_ok=True)
            task_dir.rmdir()


__all__ = ["LocalTempFileStore"]
