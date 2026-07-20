"""Compatibility facade for split OFFLINE result hooks."""

from src.application.task_results import OfflineBatchContext, OfflineTaskContext
from src.task_queue.hook_contracts import (
    BaseOfflineTaskResultHook,
    OfflineBatchResultHook,
    OfflineTaskResultHook,
)
from src.task_queue.hook_handlers import (
    OfflineBatchResultHandler,
    OfflineTaskResultHandler,
)
from src.task_queue.result_hooks import (
    AudioBackupHook,
    ResultPersistenceHook,
    TempCleanupHook,
    TextResultFileHook,
)

__all__ = [
    "AudioBackupHook",
    "BaseOfflineTaskResultHook",
    "OfflineBatchContext",
    "OfflineBatchResultHandler",
    "OfflineBatchResultHook",
    "OfflineTaskContext",
    "OfflineTaskResultHandler",
    "OfflineTaskResultHook",
    "ResultPersistenceHook",
    "TempCleanupHook",
    "TextResultFileHook",
]
