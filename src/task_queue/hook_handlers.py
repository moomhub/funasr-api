"""Result hook composition and execution handlers."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List

from src.application.task_results import OfflineBatchContext, OfflineTaskContext
from src.core.results.types import RecognitionResult
from src.task_queue.hook_contracts import (
    OfflineBatchResultHook,
    OfflineTaskResultHook,
)
from src.task_queue.hook_execution import SequentialHookExecutor
from src.task_queue.result_hooks import (
    AudioBackupHook,
    ResultPersistenceHook,
    TempCleanupHook,
    TextResultFileHook,
)

logger = logging.getLogger(__name__)


class OfflineTaskResultHandler:
    def __init__(
        self,
        hooks: Iterable[OfflineTaskResultHook],
        executor: SequentialHookExecutor | None = None,
    ):
        self.hooks: List[OfflineTaskResultHook] = list(hooks)
        self.executor = executor or SequentialHookExecutor(logger)

    @classmethod
    def from_services(
        cls,
        *,
        task_repository: Any,
        postprocessor: Any,
        result_dir: str,
        temp_file_store: Any = None,
        **_: Any,
    ) -> "OfflineTaskResultHandler":
        return cls([
            AudioBackupHook(task_repository, postprocessor=postprocessor),
            TextResultFileHook(result_dir),
            ResultPersistenceHook(task_repository),
            TempCleanupHook(temp_file_store),
        ])

    async def handle_success(
        self,
        context: OfflineTaskContext,
        result: RecognitionResult,
    ) -> None:
        await self.executor.run(
            self.hooks,
            phase="offline_success",
            invoke=lambda hook: hook.on_success(context, result),
            raise_critical=True,
        )

    async def handle_failure(
        self,
        context: OfflineTaskContext,
        error_message: str,
    ) -> None:
        await self.executor.run(
            self.hooks,
            phase="offline_failure",
            invoke=lambda hook: hook.on_failure(context, error_message),
            raise_critical=False,
        )


class OfflineBatchResultHandler:
    def __init__(
        self,
        hooks: Iterable[OfflineBatchResultHook] = (),
        executor: SequentialHookExecutor | None = None,
    ):
        self.hooks: List[OfflineBatchResultHook] = list(hooks)
        self.executor = executor or SequentialHookExecutor(logger)

    @classmethod
    def from_services(cls, **_: Any) -> "OfflineBatchResultHandler":
        return cls()

    async def handle_complete(self, context: OfflineBatchContext) -> None:
        await self.executor.run(
            self.hooks,
            phase="offline_batch_complete",
            invoke=lambda hook: hook.on_complete(context),
            raise_critical=True,
        )


__all__ = ["OfflineBatchResultHandler", "OfflineTaskResultHandler"]
