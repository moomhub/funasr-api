"""Helpers for safe operational logs and detailed debug payloads."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit


SENSITIVE_KEY_PARTS = (
    "access_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def mask_sensitive(value: Any) -> Any:
    """Recursively mask sensitive mapping values before logging."""
    if isinstance(value, Mapping):
        return {
            key: "***" if is_sensitive_key(key) else mask_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [mask_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_sensitive(item) for item in value)
    return value


def mask_url(url: str) -> str:
    """Hide password-like credentials in database URLs."""
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    if not parts.password:
        return url
    username = parts.username or ""
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = f"{username}:***@{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def json_for_log(payload: Any) -> str:
    """Serialize a debug payload with sensitive values masked."""
    try:
        return json.dumps(mask_sensitive(payload), ensure_ascii=False, default=_json_default)
    except Exception:
        return repr(mask_sensitive(payload))


def log_exception(
    logger: logging.Logger,
    level: int,
    operation: str,
    exc: BaseException,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    """Log a safe operational summary and full DEBUG diagnostics."""
    logger.log(
        level,
        "%s failed: error_type=%s",
        operation,
        type(exc).__name__,
    )
    logger.debug(
        "%s failure details: context=%s",
        operation,
        json_for_log(dict(context or {})),
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def text_preview(text: Any, limit: int = 40) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def sequence_summary(items: Any) -> dict[str, Any]:
    values = list(items or []) if isinstance(items, (list, tuple)) else []
    return {
        "count": len(values),
        "preview": values[:3],
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return value.__dict__
    return repr(value)
