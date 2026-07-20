"""Type coercion helpers for YAML and environment values."""

from __future__ import annotations

from typing import Any, Iterable, List


def as_list(value: Any, default: Iterable[Any] | None = None) -> List[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def as_int(value: Any) -> int:
    return int(value)


def as_float(value: Any) -> float:
    return float(value)


__all__ = ["as_bool", "as_float", "as_int", "as_list"]
