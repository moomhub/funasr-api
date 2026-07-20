"""Post-processing hooks for completed offline/SPK uploads."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.core.debug_logging import log_exception

logger = logging.getLogger(__name__)


@dataclass
class StoredFileResult:
    s3_key: str
    storage_backend: str
    bucket_name: Optional[str]
    stored_filename: str
    file_sha256: str
    file_size: int
    local_path: Optional[str] = None
    upload_status: str = "uploaded"
    is_reused: bool = False


def calculate_file_hash(path: str, algorithm: str = "sha256") -> str:
    """Calculate a streaming file hash."""
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class RocketMQCompletionPublisher:
    """Optional RocketMQ publisher; unavailable configuration is a no-op."""

    def __init__(self, config: Any):
        self.config = config
        notifications = self.config.get("notifications", None)
        notifications = notifications or {}
        rocket = notifications.get("rocketmq", {}) or {}
        self.enabled = bool(notifications.get("enabled", False)) and notifications.get("type") == "rocketmq"
        self.namesrv = self.config.get_env("notifications.rocketmq.namesrv", "ROCKETMQ_NAMESRV", rocket.get("namesrv"))
        self.topic = self.config.get_env("notifications.rocketmq.topic", "ROCKETMQ_TOPIC", rocket.get("topic"))
        self.group = self.config.get_env("notifications.rocketmq.group", "ROCKETMQ_GROUP", rocket.get("group", "funasr-producer"))

    async def publish_completion(self, *, task_key: str, email: str, s3_key: str, source: str) -> bool:
        if not email:
            return False
        if not self.enabled or not self.namesrv or not self.topic:
            logger.info("RocketMQ 未配置，跳过完成消息: %s", task_key)
            return False

        payload = {
            "task_key": task_key,
            "email": email,
            "s3_key": s3_key,
            "source": source,
        }
        try:
            await asyncio.to_thread(self._publish_sync, payload)
            logger.info("✅ RocketMQ 完成消息已发送: %s", task_key)
            return True
        except Exception as exc:
            log_exception(
                logger,
                logging.WARNING,
                "RocketMQ completion publish",
                exc,
                context={"task_id": task_key},
            )
            return False

    def _publish_sync(self, payload: dict) -> None:
        from rocketmq.client import Message, Producer

        producer = Producer(self.group)
        producer.set_namesrv_addr(self.namesrv)
        producer.start()
        try:
            message = Message(self.topic)
            message.set_body(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            message.set_keys(payload["task_key"])
            producer.send_sync(message)
        finally:
            producer.shutdown()


class FilePostProcessor:
    """Archive original upload, dedupe by hash, notify MQ, and clean temp files."""

    def __init__(
        self,
        config: Any,
        publisher: RocketMQCompletionPublisher = None,
        file_index_repository: Any = None,
        hasher: Any = None,
        archive_storage: Any = None,
        cleaner: Any = None,
        audio_backup_store: Any = None,
    ):
        from .file_processing import ArchiveStorage, FileHasher, FileIndex, TempFileCleaner

        self.config = config
        self.publisher = publisher or RocketMQCompletionPublisher(self.config)
        self.hasher = hasher or FileHasher()
        self.file_index = FileIndex(file_index_repository)
        self.archive_storage = archive_storage or ArchiveStorage(
            config,
            audio_backup_store=audio_backup_store,
        )
        self.cleaner = cleaner or TempFileCleaner()

    async def handle_complete(
        self,
        *,
        local_path: str,
        task_key: str,
        filename: str,
        source: str,
        email: Optional[str] = None,
        delete_local_on_success: bool = True,
    ) -> Optional[StoredFileResult]:
        logger.debug(
            "文件后处理输入: task_id=%s source=%s filename=%s local_path=%s "
            "delete_local_on_success=%s email_present=%s",
            task_key,
            source,
            filename,
            local_path,
            delete_local_on_success,
            bool(email),
        )
        path = Path(local_path)
        if not path.exists():
            logger.warning("后处理跳过，文件不存在: task_id=%s", task_key)
            logger.debug("后处理缺失文件路径: task_id=%s path=%s", task_key, local_path)
            return None

        file_sha256 = await self.hasher.sha256(path)
        file_size = await asyncio.to_thread(lambda: path.stat().st_size)

        existing = await asyncio.to_thread(self.file_index.find, file_sha256)
        if existing is not None:
            stored = StoredFileResult(
                s3_key=existing.s3_key,
                storage_backend=existing.storage_backend,
                bucket_name=existing.bucket_name,
                stored_filename=existing.stored_filename,
                file_sha256=file_sha256,
                file_size=file_size,
                local_path=existing.local_path,
                upload_status="reused",
                is_reused=True,
            )
        else:
            stored = await self._store_new_file(path, task_key, filename, source, file_sha256, file_size)
            if stored is not None:
                await asyncio.to_thread(self.file_index.record, task_key, source, filename, stored)

        if stored is None:
            return None

        await self.publisher.publish_completion(
            task_key=task_key,
            email=email or "",
            s3_key=stored.s3_key,
            source=source,
        )
        if delete_local_on_success:
            await asyncio.to_thread(self.cleaner.cleanup, path)
        return stored

    async def _store_new_file(
        self,
        path: Path,
        task_key: str,
        filename: str,
        source: str,
        file_sha256: str,
        file_size: int,
    ) -> Optional[StoredFileResult]:
        return await self.archive_storage.store(
            path,
            task_key,
            filename,
            source,
            file_sha256,
            file_size,
        )

