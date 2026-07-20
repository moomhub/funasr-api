"""SQL repository for archived-file indexes."""

from typing import Any

from src.database.daos.file_index import S3FileDAO
from src.database.repositories._session import flush_and_detach


class SqlFileIndexRepository:
    def __init__(self, db_manager: Any):
        self.db = db_manager

    def get_by_hash(self, file_sha256: str) -> Any:
        with self.db.session_scope() as session:
            item = S3FileDAO.get_by_hash(file_sha256, session=session)
            return flush_and_detach(session, item)

    def create(self, **kwargs: Any) -> Any:
        with self.db.session_scope() as session:
            item = S3FileDAO.create(session=session, **kwargs)
            return flush_and_detach(session, item)


__all__ = ["SqlFileIndexRepository"]
