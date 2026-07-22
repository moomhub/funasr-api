"""Stable identifiers shared by task submission and persistence layers."""

from __future__ import annotations

import uuid


def new_task_id() -> str:
    """Return a compact UUID while preserving UUID collision guarantees."""
    return uuid.uuid4().hex


__all__ = ["new_task_id"]
