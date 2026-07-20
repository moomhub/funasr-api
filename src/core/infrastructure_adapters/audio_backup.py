"""Original audio backup adapters."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .status import component_status

logger = logging.getLogger(__name__)


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
    ) -> Optional[str]:
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
            self.last_error = str(exc)
            logger.warning(
                "S3 audio backup skipped: error_type=%s",
                type(exc).__name__,
            )
            logger.debug(
                "S3 audio backup failure details: task_id=%s",
                task_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            return None

    def status(self) -> Dict[str, Any]:
        return component_status(
            self,
            {"bucket": self.bucket, "prefix": self.prefix},
        )


__all__ = ["NoopAudioBackupStore", "S3AudioBackupStore"]
