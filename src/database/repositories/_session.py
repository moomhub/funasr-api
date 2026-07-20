"""Session-boundary helpers for SQL repository adapters."""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect
from sqlalchemy.orm import Session


def flush_and_detach(session: Session, result: Any) -> Any:
    """Flush pending writes and detach mapped results before the session closes."""
    session.flush()
    items = result if isinstance(result, list) else [result]
    for item in items:
        state = inspect(item, raiseerr=False) if item is not None else None
        if state is not None and state.session is session:
            session.expunge(item)
    return result


__all__ = ["flush_and_detach"]
