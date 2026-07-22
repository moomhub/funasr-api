"""Components used by the upload completion pipeline."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from src.core.debug_logging import log_exception

from .postprocess import StoredFileResult, calculate_file_hash

logger = logging.getLogger(__name__)


class FileHasher:
    async def sha256(self, path: Path) -> str:
        return await asyncio.to_thread(calculate_file_hash, str(path))


class FileIndex:
    def __init__(self, repository: Any = None):
        self.repository = repository

    def find(self, file_sha256: str) -> Any:
        if self.repository is None:
            return None
        try:
            return self.repository.get_by_hash(file_sha256)
        except Exception as exc:
            log_exception(
                logger,
                logging.DEBUG,
                "File hash deduplication lookup",
                exc,
                context={"file_sha256": file_sha256},
            )
            return None

    def record(self, task_key: str, source: str, filename: str, stored: StoredFileResult) -> bool:
        if self.repository is None:
            logger.info(
                "File index skipped: task_id=%s backend=%s reason=repository_disabled",
                task_key,
                stored.storage_backend,
            )
            return False
        try:
            self.repository.create(
                task_key=task_key,
                task_type=source,
                storage_backend=stored.storage_backend,
                s3_key=stored.s3_key,
                stored_filename=stored.stored_filename,
                original_filename=filename,
                file_sha256=stored.file_sha256,
                bucket_name=stored.bucket_name,
                file_size=stored.file_size,
                local_path=stored.local_path,
                upload_status=stored.upload_status,
                is_reused=stored.is_reused,
            )
            logger.info(
                "File index saved: task_id=%s backend=%s",
                task_key,
                stored.storage_backend,
            )
            return True
        except Exception as exc:
            log_exception(
                logger,
                logging.WARNING,
                "File index persistence",
                exc,
                context={"task_id": task_key, "source": source, "filename": filename},
            )
            return False


class ArchiveStorage:
    def __init__(self, config: Any, audio_backup_store: Any = None):
        self.config = config
        self.audio_backup_store = audio_backup_store

    async def store(
        self,
        path: Path,
        task_key: str,
        filename: str,
        source: str,
        file_sha256: str,
        file_size: int,
    ) -> Optional[StoredFileResult]:
        store = self.audio_backup_store
        if store is None or not store.enabled:
            raise RuntimeError("Configured archive storage is unavailable")
        backend = str(store.name)
        logger.info(
            "File archive started: backend=%s task_id=%s source=%s size_bytes=%s",
            backend,
            task_key,
            source,
            file_size,
        )
        stored_key = await store.backup_original(str(path), task_key, filename)
        if not stored_key:
            raise RuntimeError("Configured archive storage returned no key")
        local_path = store.get_local_path(stored_key)
        logger.info("File archive completed: backend=%s task_id=%s", backend, task_key)
        logger.debug(
            "File archive details: backend=%s task_id=%s source=%s key=%s local_path=%s size=%s",
            backend,
            task_key,
            source,
            stored_key,
            local_path,
            file_size,
        )
        return StoredFileResult(
            s3_key=stored_key,
            storage_backend=backend,
            bucket_name=getattr(store, "bucket", None),
            stored_filename=Path(stored_key).name,
            file_sha256=file_sha256,
            file_size=file_size,
            local_path=local_path,
        )


class TempFileCleaner:
    @staticmethod
    def cleanup(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()
            except OSError:
                pass
            logger.info("✅ 临时文件已清理")
            logger.debug("临时文件清理路径: %s", path)
        except Exception as exc:
            log_exception(
                logger,
                logging.WARNING,
                "Temporary file cleanup",
                exc,
                context={"path": str(path)},
            )


__all__ = ["ArchiveStorage", "FileHasher", "FileIndex", "TempFileCleaner"]
