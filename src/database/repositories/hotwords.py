"""SQL repository for hotwords."""

from typing import Any, List

from src.database.daos.hotwords import HotwordDAO


class SqlHotwordRepository:
    def __init__(self, db_manager: Any):
        self.db = db_manager

    def get_by_id(self, hotword_id: int) -> List[dict]:
        from src.core.hotwords.loader import parse_hotwords
        with self.db.session_scope() as session:
            hotword = HotwordDAO.get_by_id(hotword_id, session=session)
            if hotword is not None:
                return parse_hotwords(hotword.text)
        return []

    def get_formatted_list(self) -> List[dict]:
        with self.db.session_scope() as session:
            return HotwordDAO.get_formatted_list(session=session)


__all__ = ["SqlHotwordRepository"]
