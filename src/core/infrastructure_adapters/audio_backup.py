"""Original audio backup adapters."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.core.debug_logging import json_for_log

from .status import component_status

logger = logging.getLogger(__name__)


async def _restore_atomically(
    destination: Path,
    restore_to_path: Callable[[str], None],
) -> str:
    """Restore into a temporary sibling before exposing the completed file."""
    part_path = destination.with_name(f"{destination.name}.part")
    await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(part_path.unlink, missing_ok=True)
    try:
        await asyncio.to_thread(restore_to_path, str(part_path))
        await asyncio.to_thread(part_path.replace, destination)
        return str(destination)
    except Exception:
        await asyncio.to_thread(part_path.unlink, missing_ok=True)
        await asyncio.to_thread(destination.unlink, missing_ok=True)
        raise


class NoopAudioBackupStore:
    name = "noop"
    enabled = False
    available = True
    last_error = None
    bucket = None

    async def backup_original(
        self,
        local_path: str,
        task_id: str,
        filename: str,
    ) -> Optional[str]:
        return None

    def status(self) -> Dict[str, Any]:
        return component_status(self)

    def check_connection(self) -> None:
        return None

    def get_local_path(self, _key: str) -> Optional[str]:
        return None

    async def restore_original(
        self,
        archive_key: str,
        local_path: str,
        task_id: str,
    ) -> str:
        raise RuntimeError("Archive storage is disabled")


class LocalAudioBackupStore:
    name = "local"
    enabled = True
    bucket = None

    def __init__(self, root: str, prefix: str = "audio"):
        self.root = Path(root)
        self.prefix = str(prefix or "").strip("/\\")
        self.available = True
        self.last_error = None

    async def backup_original(
        self,
        local_path: str,
        task_id: str,
        filename: str,
    ) -> str:
        suffix = Path(filename).suffix or Path(local_path).suffix
        stored_filename = f"{task_id}_{uuid.uuid4().hex}{suffix}"
        key = f"{self.prefix}/{stored_filename}" if self.prefix else stored_filename
        destination = self.root.joinpath(*key.split("/"))
        try:
            await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, str(local_path), str(destination))
            self.available = True
            self.last_error = None
            return key
        except Exception as exc:
            self.available = False
            self.last_error = type(exc).__name__
            logger.warning(
                "Local audio backup failed: task_id=%s error_type=%s",
                task_id,
                type(exc).__name__,
            )
            logger.debug(
                "Local audio backup failure details: task_id=%s",
                task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise

    def check_connection(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".funasr-write-check-", delete=True):
                pass
            self.available = True
            self.last_error = None
        except Exception as exc:
            self.available = False
            self.last_error = type(exc).__name__
            raise

    def get_local_path(self, key: str) -> str:
        return str(self._resolve_key(key))

    async def restore_original(
        self,
        archive_key: str,
        local_path: str,
        task_id: str,
    ) -> str:
        destination = Path(local_path)
        try:
            source = self._resolve_key(archive_key)
            if not source.is_file():
                raise FileNotFoundError("Archived audio is unavailable")
            restored = await _restore_atomically(
                destination,
                lambda part_path: shutil.copy2(str(source), part_path),
            )
            self.available = True
            self.last_error = None
            logger.info(
                "Archived audio restored: backend=local task_id=%s",
                task_id,
            )
            logger.debug(
                "Local archive restore details: %s",
                json_for_log({
                    "task_id": task_id,
                    "archive_key": archive_key,
                    "source_path": str(source),
                    "destination_path": restored,
                }),
            )
            return restored
        except Exception as exc:
            self.available = False
            self.last_error = type(exc).__name__
            logger.warning(
                "Archived audio restore failed: backend=local task_id=%s error_type=%s",
                task_id,
                type(exc).__name__,
            )
            logger.debug(
                "Local archive restore failure details: task_id=%s",
                task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise

    def _resolve_key(self, key: str) -> Path:
        normalized = str(key or "").replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("Archive key is empty")
        root = self.root.resolve()
        candidate = root.joinpath(*normalized.split("/")).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Archive key resolves outside the configured root") from exc
        return candidate

    def status(self) -> Dict[str, Any]:
        return component_status(
            self,
            {"root": str(self.root), "prefix": self.prefix},
        )


class S3AudioBackupStore:
    name = "s3"
    enabled = True

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1",
        prefix: str = "audio",
    ):
        import boto3

        self.bucket = bucket
        self.prefix = str(prefix or "").strip("/")
        self.available = True
        self.last_error = None
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    async def backup_original(
        self,
        local_path: str,
        task_id: str,
        filename: str,
    ) -> str:
        try:
            suffix = Path(filename).suffix or Path(local_path).suffix
            stored_filename = f"{task_id}_{uuid.uuid4().hex}{suffix}"
            s3_key = (
                f"{self.prefix}/{stored_filename}"
                if self.prefix
                else stored_filename
            )
            await asyncio.to_thread(
                self.s3_client.upload_file,
                str(local_path),
                self.bucket,
                s3_key,
                ExtraArgs={
                    "Metadata": {
                        "task_id": task_id,
                        "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
            )
            self.available = True
            self.last_error = None
            return s3_key
        except Exception as exc:
            self.available = False
            self.last_error = type(exc).__name__
            logger.warning(
                "S3 audio backup failed: task_id=%s error_type=%s",
                task_id,
                type(exc).__name__,
            )
            logger.debug(
                "S3 audio backup failure details: task_id=%s",
                task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise

    def check_connection(self) -> None:
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            self.available = True
            self.last_error = None
        except Exception as exc:
            self.available = False
            self.last_error = type(exc).__name__
            raise

    def get_local_path(self, _key: str) -> Optional[str]:
        return None

    async def restore_original(
        self,
        archive_key: str,
        local_path: str,
        task_id: str,
    ) -> str:
        if not archive_key:
            raise ValueError("Archive key is empty")
        destination = Path(local_path)
        try:
            restored = await _restore_atomically(
                destination,
                lambda part_path: self.s3_client.download_file(
                    self.bucket,
                    archive_key,
                    part_path,
                ),
            )
            self.available = True
            self.last_error = None
            logger.info(
                "Archived audio restored: backend=s3 task_id=%s",
                task_id,
            )
            logger.debug(
                "S3 archive restore details: %s",
                json_for_log({
                    "task_id": task_id,
                    "archive_key": archive_key,
                    "destination_path": restored,
                    "bucket": self.bucket,
                }),
            )
            return restored
        except Exception as exc:
            self.available = False
            self.last_error = type(exc).__name__
            logger.warning(
                "Archived audio restore failed: backend=s3 task_id=%s error_type=%s",
                task_id,
                type(exc).__name__,
            )
            logger.debug(
                "S3 archive restore failure details: task_id=%s",
                task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise

    def status(self) -> Dict[str, Any]:
        return component_status(
            self,
            {"bucket": self.bucket, "prefix": self.prefix},
        )


__all__ = [
    "LocalAudioBackupStore",
    "NoopAudioBackupStore",
    "S3AudioBackupStore",
]
