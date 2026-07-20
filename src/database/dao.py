"""Backward-compatible imports for entity-scoped database DAOs."""

from src.database.daos import HotwordDAO, OfflineTaskDAO, S3FileDAO, SpkTaskDAO

__all__ = [
    "HotwordDAO",
    "OfflineTaskDAO",
    "S3FileDAO",
    "SpkTaskDAO",
]
