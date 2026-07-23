"""SQL repository for SPK tasks."""

from typing import Any, Callable, List

from src.database.daos.speaker_tasks import SpkTaskDAO
from src.database.repositories._session import flush_and_detach


class SqlSpeakerTaskRepository:
    def __init__(self, db_manager: Any):
        self.db = db_manager

    def _call(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self.db.session_scope() as session:
            result = operation(*args, session=session, **kwargs)
            return flush_and_detach(session, result)

    def create_task(self, **kwargs: Any) -> Any:
        return self._call(SpkTaskDAO.create, **kwargs)

    def get_task(self, task_id: str) -> Any:
        return self._call(SpkTaskDAO.get_by_id, task_id)

    def get_pending_tasks(self, limit: int = 100) -> List[Any]:
        return self._call(SpkTaskDAO.get_pending_tasks, limit=limit)

    def recover_stale_processing(self, timeout_seconds: int) -> int:
        return self._call(SpkTaskDAO.recover_stale_processing, timeout_seconds)

    def update_status(self, task_id: str, status: str) -> Any:
        return self._call(SpkTaskDAO.update_status, task_id, status)

    def save_result(self, task_id: str, result: dict, processing_time: float, **kwargs: Any) -> Any:
        return self._call(SpkTaskDAO.update_result, task_id, result, processing_time, **kwargs)

    def record_error(self, task_id: str, error_message: str, retry: bool = False) -> Any:
        return self._call(SpkTaskDAO.update_error, task_id, error_message, retry=retry)


__all__ = ["SqlSpeakerTaskRepository"]
