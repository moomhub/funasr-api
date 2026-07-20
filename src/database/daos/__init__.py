"""Entity-scoped SQLAlchemy data access objects."""

from .file_index import S3FileDAO
from .hotwords import HotwordDAO
from .offline_tasks import OfflineTaskDAO
from .speaker_tasks import SpkTaskDAO

__all__ = [
    "HotwordDAO",
    "OfflineTaskDAO",
    "S3FileDAO",
    "SpkTaskDAO",
]
