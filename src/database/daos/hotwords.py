"""Data access operations for hotwords."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy.orm import Session

from src.core.debug_logging import json_for_log, log_exception
from src.core.hotwords.loader import parse_hotwords
from src.database.models import Hotword

logger = logging.getLogger(__name__)


class HotwordDAO:
    """Persist and format hotword records within a caller-owned session."""

    @staticmethod
    def add(
        name: str,
        text: Any,
        *,
        session: Session,
    ) -> Hotword:
        text_value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        hotword = Hotword(
            name=name,
            text=text_value,
        )
        session.add(hotword)
        session.flush()

        logger.info("热词已创建: id=%s", hotword.id)
        logger.debug(
            "热词创建详情: %s",
            json_for_log({
                "id": hotword.id,
                "name": name,
                "text": text_value,
            }),
        )
        return hotword

    @staticmethod
    def get_all(*, session: Session) -> List[Hotword]:
        return session.query(Hotword).filter(
            Hotword.enabled.is_(True),
            Hotword.is_deleted.is_(False),
        ).all()

    @staticmethod
    def get_by_id(hotword_id: int, *, session: Session) -> Optional[Hotword]:
        return session.query(Hotword).filter(
            Hotword.id == hotword_id,
            Hotword.enabled.is_(True),
            Hotword.is_deleted.is_(False),
        ).first()

    @staticmethod
    def get_formatted_list(
        *,
        session: Session,
    ) -> List[dict]:
        formatted: List[dict] = []
        for hotword in HotwordDAO.get_all(session=session):
            try:
                formatted.extend(parse_hotwords(hotword.text))
            except Exception as exc:
                log_exception(
                    logger,
                    logging.WARNING,
                    "Hotword record parsing",
                    exc,
                    context={"hotword_id": hotword.id, "text": hotword.text},
                )
        return formatted

    @staticmethod
    def update(
        hotword_id: int,
        name: str = None,
        text: Any = None,
        enabled: bool = None,
        *,
        session: Session,
    ) -> Optional[Hotword]:
        hotword = session.query(Hotword).filter(Hotword.id == hotword_id).first()
        if hotword is None:
            return None
        if name is not None:
            hotword.name = name
        if text is not None:
            hotword.text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        if enabled is not None:
            hotword.enabled = enabled
        hotword.updated_at = datetime.now(timezone.utc)
        logger.info("热词已更新: id=%s", hotword_id)
        return hotword

    @staticmethod
    def delete(
        hotword_id: int,
        soft_delete: bool = True,
        *,
        session: Session,
    ) -> Optional[Hotword]:
        hotword = session.query(Hotword).filter(Hotword.id == hotword_id).first()
        if hotword is None:
            return None
        if soft_delete:
            hotword.is_deleted = True
            hotword.updated_at = datetime.now(timezone.utc)
            logger.info("热词已软删除: id=%s", hotword_id)
        else:
            session.delete(hotword)
            logger.info("热词已硬删除: id=%s", hotword_id)
        return hotword


__all__ = ["HotwordDAO"]
