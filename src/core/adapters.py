"""Compatibility facade for split infrastructure adapters."""

from src.core.infrastructure_adapters.audio_backup import (
    NoopAudioBackupStore,
    S3AudioBackupStore,
)
from src.core.infrastructure_adapters.hotwords import (
    DatabaseHotwordProvider,
    EmptyHotwordProvider,
)
from src.core.infrastructure_adapters.task_repository import MemoryTaskRepository
from src.core.infrastructure_adapters.temp_files import LocalTempFileStore

__all__ = [
    "DatabaseHotwordProvider",
    "EmptyHotwordProvider",
    "LocalTempFileStore",
    "MemoryTaskRepository",
    "NoopAudioBackupStore",
    "S3AudioBackupStore",
]
