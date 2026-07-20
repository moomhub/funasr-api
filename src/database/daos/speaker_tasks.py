"""Data access operations for standalone SPK tasks."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from src.core.debug_logging import json_for_log
from src.database.models import SpkTask

logger = logging.getLogger(__name__)


class SpkTaskDAO:
    """Persist standalone SPK task state within a caller-owned session."""

    @staticmethod
    def create(
        filename: str,
        task_id: str = None,
        file_size: int = None,
        email: str = None,
        vip: bool = False,
        *,
        session: Session,
    ) -> SpkTask:
        if task_id is None:
            task_id = str(uuid.uuid4())
        task = SpkTask(
            id=task_id,
            filename=filename,
            file_size=file_size,
            email=email,
            vip=bool(vip),
            status="pending",
        )
        session.add(task)
        session.flush()

        logger.info("SPK 任务已创建: task_id=%s file_size=%s", task_id, file_size or 0)
        logger.debug(
            "SPK 任务创建详情: %s",
            json_for_log({
                "task_id": task_id,
                "filename": filename,
                "file_size": file_size,
                "email": email,
                "vip": bool(vip),
            }),
        )
        return task

    @staticmethod
    def get_by_id(task_id: str, *, session: Session) -> Optional[SpkTask]:
        return session.query(SpkTask).filter(SpkTask.id == task_id).first()

    @staticmethod
    def get_pending_tasks(limit: int = 100, *, session: Session) -> List[SpkTask]:
        return session.query(SpkTask).filter(
            SpkTask.status == "pending",
            SpkTask.retry_count < SpkTask.max_retries,
        ).order_by(
            SpkTask.vip.desc(),
            SpkTask.created_at.asc(),
        ).limit(limit).all()

    @staticmethod
    def recover_stale_processing(timeout_seconds: int, *, session: Session) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(0, timeout_seconds))
        tasks = session.query(SpkTask).filter(
            SpkTask.status == "processing",
            SpkTask.started_at.isnot(None),
            SpkTask.started_at <= cutoff,
        ).all()
        for task in tasks:
            task.status = "pending"
        return len(tasks)

    @staticmethod
    def update_status(
        task_id: str,
        status: str,
        *,
        session: Session,
    ) -> Optional[SpkTask]:
        task = SpkTaskDAO.get_by_id(task_id, session=session)
        if task is None:
            return None
        task.status = status
        if status == "processing":
            task.started_at = datetime.now(timezone.utc)
        elif status == "completed":
            task.completed_at = datetime.now(timezone.utc)
        return task

    @staticmethod
    def update_result(
        task_id: str,
        result: dict,
        processing_time: float,
        s3_key: str = None,
        file_hash: str = None,
        *,
        session: Session,
    ) -> Optional[SpkTask]:
        task = SpkTaskDAO.get_by_id(task_id, session=session)
        if task is None:
            return None
        task.result = result
        task.segments = result.get("segments") if isinstance(result, dict) else None
        task.speaker_ids = result.get("speaker_ids") if isinstance(result, dict) else None
        task.speaker_count = result.get("speaker_count", 0) if isinstance(result, dict) else 0
        task.processing_time = processing_time
        task.s3_key = s3_key or task.s3_key
        task.file_hash = file_hash or task.file_hash
        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        return task

    @staticmethod
    def update_error(
        task_id: str,
        error_message: str,
        retry: bool = True,
        *,
        session: Session,
    ) -> Optional[SpkTask]:
        task = SpkTaskDAO.get_by_id(task_id, session=session)
        if task is None:
            return None
        task.error_message = error_message
        task.retry_count += 1
        if retry and task.retry_count < task.max_retries:
            task.status = "pending"
        else:
            task.status = "failed"
            task.completed_at = datetime.now(timezone.utc)
        return task


__all__ = ["SpkTaskDAO"]
