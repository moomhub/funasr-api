"""Shared helpers for task-oriented application flows."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class TaskFileRef:
    task_id: str
    task: Any
    filename: str
    audio_path: Path


async def load_task_file_ref(
    *,
    repository: Any,
    temp_file_store: Any,
    task_id: str,
    logger: logging.Logger,
    missing_message: str,
) -> Optional[TaskFileRef]:
    task = await asyncio.to_thread(repository.get_task, task_id)
    if not task:
        logger.error(missing_message, task_id)
        return None

    filename = task.filename
    return TaskFileRef(
        task_id=task_id,
        task=task,
        filename=filename,
        audio_path=Path(temp_file_store.resolve(task_id, filename)),
    )


def require_audio_file(audio_path: Path) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")


async def mark_task_processing(repository: Any, task_id: str) -> None:
    await asyncio.to_thread(repository.update_status, task_id, "processing")


async def record_task_error(
    repository: Any,
    task_id: str,
    error_message: str,
    *,
    retry: bool = False,
) -> None:
    await asyncio.to_thread(repository.record_error, task_id, error_message, retry=retry)


__all__ = [
    "TaskFileRef",
    "load_task_file_ref",
    "mark_task_processing",
    "record_task_error",
    "require_audio_file",
]
