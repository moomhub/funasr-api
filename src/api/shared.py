"""Shared API route helpers."""

from dataclasses import asdict, is_dataclass
from typing import Any

from src.application.context import AppServices


def is_engine_enabled(services: AppServices, mode: str) -> bool:
    """Return whether a configured runtime mode is enabled."""
    return services.is_engine_enabled(mode)


def is_task_submission_available(services: AppServices, mode: str) -> bool:
    """Return whether an upload can be consumed by a running task worker."""
    checker = getattr(services, "is_task_submission_available", None)
    if checker is None:
        return is_engine_enabled(services, mode)
    return bool(checker(mode))


def jsonable_payload(value: Any) -> Any:
    """Convert backend/model outputs into a response-safe payload."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: jsonable_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_payload(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value

