"""Recovery helpers for queue startup."""

from __future__ import annotations

import logging
from typing import Any, Callable


def recover_and_enqueue_tasks(
    *,
    repository: Any,
    timeout_seconds: int,
    enqueue: Callable[..., None],
    task_kind: str,
    logger: logging.Logger,
    pending_limit: int = 1000,
    missing_repository_is_error: bool = True,
) -> None:
    if repository is None:
        if missing_repository_is_error:
            logger.warning(
                "Task recovery skipped: task_kind=%s repository_not_configured=true",
                task_kind,
            )
        return

    try:
        recovered = repository.recover_stale_processing(timeout_seconds)
        if recovered:
            logger.warning(
                "Recovered stale processing tasks: task_kind=%s count=%s",
                task_kind,
                recovered,
            )
    except Exception as exc:
        _log_recovery_failure(
            logger,
            logging.WARNING,
            phase="processing",
            task_kind=task_kind,
            exc=exc,
        )

    try:
        for task in repository.get_pending_tasks(limit=pending_limit):
            enqueue(task.id, vip=bool(task.vip))
    except Exception as exc:
        level = logging.WARNING if missing_repository_is_error else logging.DEBUG
        _log_recovery_failure(
            logger,
            level,
            phase="pending",
            task_kind=task_kind,
            exc=exc,
        )


def _log_recovery_failure(
    logger: logging.Logger,
    level: int,
    *,
    phase: str,
    task_kind: str,
    exc: Exception,
) -> None:
    logger.log(
        level,
        "Task recovery failed: phase=%s task_kind=%s error_type=%s",
        phase,
        task_kind,
        type(exc).__name__,
    )
    logger.debug(
        "Task recovery failure details: phase=%s task_kind=%s",
        phase,
        task_kind,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


__all__ = ["recover_and_enqueue_tasks"]
