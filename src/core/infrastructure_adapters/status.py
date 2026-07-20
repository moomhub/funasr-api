"""Shared health status formatting for infrastructure adapters."""

from __future__ import annotations

from typing import Any, Dict


def component_status(
    component: Any,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data = {
        "name": component.name,
        "type": component.__class__.__name__,
        "enabled": component.enabled,
        "available": component.available,
        "last_error": component.last_error,
    }
    if extra:
        data.update(extra)
    return data


__all__ = ["component_status"]
