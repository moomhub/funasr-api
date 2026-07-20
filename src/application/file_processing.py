"""Components used by the upload completion pipeline."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
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

    def record(self, task_key: str, source: str, filename: str, stored: StoredFileResult) -> None:
        if self.repository is None:
            return
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
        except Exception as exc:
            log_exception(
                logger,
                logging.WARNING,
                "File index persistence",
                exc,
                context={"task_id": task_key, "source": source, "filename": filename},
            )


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
        s3_config = self.config.get("storage.s3", {}) or {}
        prefix = str(
            self.config.get_env(
                "storage.s3.prefix",
                "S3_PREFIX",
                s3_config.get("prefix", "audio"),
            )
            or ""
        ).strip("/")
        stored_filename = f"{task_key}_{uuid.uuid4().hex}{path.suffix}"
        s3_key = f"{prefix}/{stored_filename}" if prefix else stored_filename

        if self.audio_backup_store is not None and self.audio_backup_store.enabled:
            try:
                uploaded_key = await self.audio_backup_store.backup_original(
                    str(path),
                    task_key,
                    filename,
                )
                if uploaded_key:
                    logger.info("文件归档完成: backend=s3 task_id=%s", task_key)
                    logger.debug(
                        "文件归档详情: backend=s3 task_id=%s source=%s key=%s size=%s",
                        task_key,
                        source,
                        uploaded_key,
                        file_size,
                    )
                    return StoredFileResult(
                        s3_key=uploaded_key,
                        storage_backend="s3",
                        bucket_name=self.audio_backup_store.bucket,
                        stored_filename=Path(uploaded_key).name,
                        file_sha256=file_sha256,
                        file_size=file_size,
                    )
                logger.debug(
                    "S3 归档未返回 key，使用本地兜底: task_id=%s source=%s",
                    task_key,
                    source,
                )
            except Exception as exc:
                log_exception(
                    logger,
                    logging.WARNING,
                    "S3 archive upload",
                    exc,
                    context={"task_id": task_key, "source": source, "path": str(path)},
                )

        local_root = Path(self.config.get_runtime_paths()["local_files_dir"])
        local_dir = local_root / prefix if prefix else local_root
        local_dir.mkdir(parents=True, exist_ok=True)
        destination = local_dir / stored_filename
        await asyncio.to_thread(shutil.copy2, str(path), str(destination))
        logger.info("文件归档完成: backend=local task_id=%s", task_key)
        logger.debug(
            "文件归档详情: backend=local task_id=%s source=%s path=%s key=%s size=%s",
            task_key,
            source,
            destination,
            s3_key,
            file_size,
        )
        return StoredFileResult(
            s3_key=s3_key,
            storage_backend="local",
            bucket_name=None,
            stored_filename=stored_filename,
            file_sha256=file_sha256,
            file_size=file_size,
            local_path=str(destination),
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
