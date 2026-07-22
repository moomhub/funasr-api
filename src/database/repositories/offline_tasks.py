"""SQL repository for OFFLINE tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.core.ports import TaskRecord
from src.database.daos.offline_tasks import OfflineTaskDAO
from src.database.repositories._session import flush_and_detach


class SqlTaskRepository:
    name = "sql"
    available = True
    last_error = None

    def __init__(self, db_manager: Any):
        self.db = db_manager

    @staticmethod
    def _to_record(task: Any) -> Optional[TaskRecord]:
        if not task:
            return None
        return TaskRecord(
            id=task.id, filename=task.filename,
            source_task_id=task.source_task_id, file_size=task.file_size,
            status=task.status, full_text=task.full_text, segments=task.segments,
            processing_time=task.processing_time, error_message=task.error_message,
            created_at=task.created_at, started_at=task.started_at,
            completed_at=task.completed_at, s3_key=task.s3_key,
            file_hash=task.file_hash, vip=bool(task.vip),
            is_deleted=bool(task.is_deleted),
            word_timestamps=task.word_timestamps,
            retry_count=task.retry_count or 0, max_retries=task.max_retries or 3,
            email=task.email, hotwords=task.hotwords, hotword_id=task.hotword_id,
        )

    def _call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.db.session_scope() as session:
            result = operation(*args, session=session, **kwargs)
            return flush_and_detach(session, result)

    def create_task(self, task_id: str, filename: str, file_size: int, email: str = None, hotwords: str = None, hotword_id: int = None, vip: bool = False, source_task_id: str = None, s3_key: str = None, file_hash: str = None) -> TaskRecord:
        return self._to_record(self._call(OfflineTaskDAO.create, filename=filename, task_id=task_id, file_size=file_size, email=email, hotwords=hotwords, hotword_id=hotword_id, vip=vip, source_task_id=source_task_id, s3_key=s3_key, file_hash=file_hash))

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._to_record(self._call(OfflineTaskDAO.get_by_id, task_id))

    def get_pending_tasks(self, limit: int = 100) -> List[TaskRecord]:
        return [self._to_record(task) for task in self._call(OfflineTaskDAO.get_pending_tasks, limit=limit)]

    def recover_stale_processing(self, timeout_seconds: int) -> int:
        return self._call(OfflineTaskDAO.recover_stale_processing, timeout_seconds)

    def update_status(self, task_id: str, status: str) -> Optional[TaskRecord]:
        return self._to_record(self._call(OfflineTaskDAO.update_status, task_id, status))

    def save_result(self, task_id: str, full_text: str, segments: List[Dict[str, Any]], processing_time: float, word_timestamps: List[Any] = None) -> Optional[TaskRecord]:
        return self._to_record(self._call(OfflineTaskDAO.update_result, task_id, full_text, segments, processing_time, word_timestamps=word_timestamps))

    def record_error(self, task_id: str, error_message: str, retry: bool = True) -> Optional[TaskRecord]:
        return self._to_record(self._call(OfflineTaskDAO.update_error, task_id, error_message, retry=retry))

    def record_file_info(self, task_id: str, s3_key: str = None, file_hash: str = None) -> Optional[TaskRecord]:
        return self._to_record(self._call(OfflineTaskDAO.update_file_info, task_id, s3_key=s3_key, file_hash=file_hash))

    def close(self) -> None:
        self.db.close()

    def check_connection(self) -> None:
        self.db.init_db()
        self.db.check_connection()

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "enabled": True,
            "available": self.available,
            "last_error": self.last_error,
            "backend": self.db.engine.url.get_backend_name(),
        }


__all__ = ["SqlTaskRepository"]
