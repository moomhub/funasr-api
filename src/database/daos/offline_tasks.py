"""Data access operations for OFFLINE recognition tasks."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from src.core.debug_logging import json_for_log
from src.core.ids import new_task_id
from src.database.models import OfflineTask

logger = logging.getLogger(__name__)


class OfflineTaskDAO:
    """Persist OFFLINE task state within a caller-owned session."""

    @staticmethod
    def create(
        filename: str,
        task_id: str = None,
        file_size: int = None,
        email: str = None,
        hotwords: str = None,
        hotword_id: int = None,
        vip: bool = False,
        source_task_id: str = None,
        s3_key: str = None,
        file_hash: str = None,
        *,
        session: Session,
    ) -> OfflineTask:
        if task_id is None:
            task_id = new_task_id()
        task = OfflineTask(
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
            status="pending",
        )
        session.add(task)
        session.flush()

        logger.info("离线任务已创建: task_id=%s file_size=%s", task.id, file_size or 0)
        logger.debug(
            "离线任务创建详情: %s",
            json_for_log({
                "task_id": task.id,
                "filename": filename,
                "source_task_id": source_task_id,
                "file_size": file_size,
                "email": email,
                "hotwords": hotwords,
                "hotword_id": hotword_id,
                "vip": bool(vip),
            }),
        )
        return task

    @staticmethod
    def get_by_id(task_id: str, *, session: Session) -> Optional[OfflineTask]:
        return session.query(OfflineTask).filter(OfflineTask.id == task_id).first()

    @staticmethod
    def get_pending_tasks(limit: int = 100, *, session: Session) -> List[OfflineTask]:
        return session.query(OfflineTask).filter(
            OfflineTask.status == "pending",
        ).order_by(
            OfflineTask.vip.desc(),
            OfflineTask.created_at.asc(),
        ).limit(limit).all()

    @staticmethod
    def recover_stale_processing(timeout_seconds: int, *, session: Session) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(0, timeout_seconds))
        tasks = session.query(OfflineTask).filter(
            OfflineTask.status == "processing",
            OfflineTask.started_at.isnot(None),
            OfflineTask.started_at <= cutoff,
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
    ) -> Optional[OfflineTask]:
        task = OfflineTaskDAO.get_by_id(task_id, session=session)
        if task is None:
            return None
        task.status = status
        if status == "processing":
            task.started_at = datetime.now(timezone.utc)
        elif status == "completed":
            task.completed_at = datetime.now(timezone.utc)
        logger.info("离线任务状态已更新: task_id=%s status=%s", task_id, status)
        return task

    @staticmethod
    def update_result(
        task_id: str,
        full_text: str,
        segments: list,
        processing_time: float,
        word_timestamps: list = None,
        *,
        session: Session,
    ) -> Optional[OfflineTask]:
        task = OfflineTaskDAO.get_by_id(task_id, session=session)
        if task is None:
            return None
        task.full_text = full_text
        task.segments = segments
        task.word_timestamps = word_timestamps
        task.processing_time = processing_time
        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        logger.info("离线任务结果已更新: task_id=%s", task_id)
        return task

    @staticmethod
    def update_file_info(
        task_id: str,
        s3_key: str = None,
        file_hash: str = None,
        *,
        session: Session,
    ) -> Optional[OfflineTask]:
        task = OfflineTaskDAO.get_by_id(task_id, session=session)
        if task is None:
            return None
        if s3_key is not None:
            task.s3_key = s3_key
        if file_hash is not None:
            task.file_hash = file_hash
        return task

    @staticmethod
    def update_error(
        task_id: str,
        error_message: str,
        retry: bool = False,
        *,
        session: Session,
    ) -> Optional[OfflineTask]:
        task = OfflineTaskDAO.get_by_id(task_id, session=session)
        if task is None:
            return None
        task.error_message = error_message
        task.retry_count += 1
        task.status = "failed"
        task.completed_at = datetime.now(timezone.utc)
        if retry:
            logger.info("离线任务重试已改为手动接口: task_id=%s", task_id)
        logger.error("离线任务失败: task_id=%s", task_id)
        return task


__all__ = ["OfflineTaskDAO"]
