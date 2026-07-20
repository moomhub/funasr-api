"""Entity-scoped SQL repository adapters."""

from .file_index import SqlFileIndexRepository
from .hotwords import SqlHotwordRepository
from .offline_tasks import SqlTaskRepository
from .speaker_tasks import SqlSpeakerTaskRepository

__all__ = [
    "SqlFileIndexRepository",
    "SqlHotwordRepository",
    "SqlSpeakerTaskRepository",
    "SqlTaskRepository",
]
