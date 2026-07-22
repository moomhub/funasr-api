"""Application ports for optional infrastructure modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class TaskRecord:
    id: str
    filename: str
    file_size: Optional[int] = None
    status: str = "pending"
    full_text: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    s3_key: Optional[str] = None
    file_hash: Optional[str] = None
    vip: bool = False
    word_timestamps: Optional[List[Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    email: Optional[str] = None
    hotwords: Optional[str] = None
    hotword_id: Optional[int] = None
    is_deleted: bool = False
    source_task_id: Optional[str] = None


class TaskRepository(Protocol):
    name: str
    available: bool
    last_error: Optional[str]

    def create_task(self, task_id: str, filename: str, file_size: int, email: str = None, hotwords: str = None, hotword_id: int = None, vip: bool = False, source_task_id: str = None, s3_key: str = None, file_hash: str = None) -> TaskRecord:
        ...

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        ...

    def get_pending_tasks(self, limit: int = 100) -> List[TaskRecord]:
        ...

    def recover_stale_processing(self, timeout_seconds: int) -> int:
        ...

    def update_status(self, task_id: str, status: str) -> Optional[TaskRecord]:
        ...

    def save_result(self, task_id: str, full_text: str, segments: List[Dict[str, Any]], processing_time: float, word_timestamps: List[Any] = None) -> Optional[TaskRecord]:
        ...

    def record_error(self, task_id: str, error_message: str, retry: bool = True) -> Optional[TaskRecord]:
        ...

    def record_file_info(self, task_id: str, s3_key: str = None, file_hash: str = None) -> Optional[TaskRecord]:
        ...

    def close(self) -> None:
        ...

    def status(self) -> Dict[str, Any]:
        ...

    def check_connection(self) -> None:
        ...


class SpeakerTaskRepository(Protocol):
    def create_task(self, **kwargs: Any) -> Any:
        ...

    def get_task(self, task_id: str) -> Any:
        ...

    def get_pending_tasks(self, limit: int = 100) -> List[Any]:
        ...

    def recover_stale_processing(self, timeout_seconds: int) -> int:
        ...

    def update_status(self, task_id: str, status: str) -> Any:
        ...

    def save_result(self, task_id: str, result: dict, processing_time: float, **kwargs: Any) -> Any:
        ...

    def record_error(self, task_id: str, error_message: str, retry: bool = True) -> Any:
        ...


class FileIndexRepository(Protocol):
    def get_by_hash(self, file_sha256: str) -> Any:
        ...

    def create(self, **kwargs: Any) -> Any:
        ...


class HotwordRepository(Protocol):
    def get_by_id(self, hotword_id: int) -> List[dict]:
        ...

    def get_formatted_list(self) -> List[dict]:
        ...


class TempFileStore(Protocol):
    name: str
    available: bool
    last_error: Optional[str]

    async def save_upload(self, upload_file: Any, task_id: str, max_size: int) -> Tuple[Path, int]:
        ...

    def resolve(self, task_id: str, filename: str) -> Path:
        ...

    def exists(self, task_id: str, filename: str) -> bool:
        ...

    def cleanup(self, task_id: str) -> None:
        ...

    def status(self) -> Dict[str, Any]:
        ...


class AudioBackupStore(Protocol):
    name: str
    enabled: bool
    available: bool
    last_error: Optional[str]
    bucket: Optional[str]

    async def backup_original(self, local_path: str, task_id: str, filename: str) -> Optional[str]:
        ...

    async def restore_original(self, archive_key: str, local_path: str, task_id: str) -> str:
        ...

    def check_connection(self) -> None:
        ...

    def get_local_path(self, key: str) -> Optional[str]:
        ...

    def status(self) -> Dict[str, Any]:
        ...


class HotwordProvider(Protocol):
    name: str
    enabled: bool
    available: bool
    last_error: Optional[str]

    def get_hotwords(self) -> List[Any]:
        ...

    def status(self) -> Dict[str, Any]:
        ...
