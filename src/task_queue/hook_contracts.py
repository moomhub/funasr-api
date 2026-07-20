"""Contracts for OFFLINE task and batch result hooks."""

from __future__ import annotations

from typing import Protocol

from src.application.task_results import OfflineBatchContext, OfflineTaskContext
from src.core.results.types import RecognitionResult


class OfflineTaskResultHook(Protocol):
    name: str
    critical: bool

    async def on_success(
        self,
        context: OfflineTaskContext,
        result: RecognitionResult,
    ) -> None:
        ...

    async def on_failure(
        self,
        context: OfflineTaskContext,
        error_message: str,
    ) -> None:
        ...


class BaseOfflineTaskResultHook:
    name = "base"
    critical = False

    async def on_success(
        self,
        context: OfflineTaskContext,
        result: RecognitionResult,
    ) -> None:
        return None

    async def on_failure(
        self,
        context: OfflineTaskContext,
        error_message: str,
    ) -> None:
        return None


class OfflineBatchResultHook(Protocol):
    name: str
    critical: bool

    async def on_complete(self, context: OfflineBatchContext) -> None:
        ...


__all__ = [
    "BaseOfflineTaskResultHook",
    "OfflineBatchResultHook",
    "OfflineTaskResultHook",
]
