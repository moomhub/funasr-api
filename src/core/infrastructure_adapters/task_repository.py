"""In-memory task repository used for isolated runtime execution and tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.core.ports import TaskRecord

from .status import component_status


class MemoryTaskRepository:
    name = "memory"
    enabled = True
    available = True
    last_error = None

    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}

    def create_task(
        self,
        task_id: str,
        filename: str,
        file_size: int,
        email: str = None,
        hotwords: str = None,
        hotword_id: int = None,
        vip: bool = False,
        source_task_id: str = None,
        s3_key: str = None,
        file_hash: str = None,
    ) -> TaskRecord:
        task = TaskRecord(
            id=task_id,
            filename=filename,
            source_task_id=source_task_id,
            file_size=file_size,
            email=email,
            hotwords=hotwords,
            hotword_id=hotword_id,
            vip=bool(vip),
            s3_key=s3_key,
            file_hash=file_hash,
            created_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def get_pending_tasks(self, limit: int = 100) -> List[TaskRecord]:
        tasks = [
            task
            for task in self._tasks.values()
            if task.status == "pending" and task.retry_count < task.max_retries
        ]
        tasks.sort(
            key=lambda item: (
                not item.vip,
                item.created_at or datetime.now(timezone.utc),
            )
        )
        return tasks[:limit]

    def recover_stale_processing(self, timeout_seconds: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=max(0, timeout_seconds)
        )
        recovered = 0
        for task in self._tasks.values():
            if (
                task.status != "processing"
                or not task.started_at
                or task.started_at > cutoff
            ):
                continue
            task.status = "pending"
            recovered += 1
        return recovered

    def update_status(
        self,
        task_id: str,
        status: str,
    ) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        task.status = status
        if status == "processing":
            task.started_at = datetime.now(timezone.utc)
        if status in {"completed", "failed"}:
            task.completed_at = datetime.now(timezone.utc)
        return task

    def save_result(
        self,
        task_id: str,
        full_text: str,
        segments: List[Dict[str, Any]],
        processing_time: float,
        word_timestamps: List[Any] = None,
    ) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        task.full_text = full_text
        task.segments = segments
        task.word_timestamps = word_timestamps
        task.processing_time = processing_time
        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        return task

    def record_error(
        self,
        task_id: str,
        error_message: str,
        retry: bool = True,
    ) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        task.error_message = error_message
        task.retry_count += 1
        if retry and task.retry_count < task.max_retries:
            task.status = "pending"
        else:
            task.status = "failed"
            task.completed_at = datetime.now(timezone.utc)
        return task

    def record_file_info(
        self,
        task_id: str,
        s3_key: str = None,
        file_hash: str = None,
    ) -> Optional[TaskRecord]:
        task = self._tasks.get(task_id)
        if task:
            if s3_key is not None:
                task.s3_key = s3_key
            if file_hash is not None:
                task.file_hash = file_hash
        return task

    def close(self) -> None:
        return None

    def status(self) -> Dict[str, Any]:
        return component_status(self, {"count": len(self._tasks)})


__all__ = ["MemoryTaskRepository"]
