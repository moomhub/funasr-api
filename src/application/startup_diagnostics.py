"""Safe startup configuration reporting and dependency connectivity checks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from src.application.postprocess import RocketMQCompletionPublisher
from src.core.config.coercion import as_bool
from src.core.debug_logging import json_for_log
from src.core.logging_config import get_logging_status

logger = logging.getLogger(__name__)


class StartupDependencyError(RuntimeError):
    """Raised after reporting an unavailable required startup dependency."""


@dataclass
class DependencyCheck:
    component: str
    provider: str
    status: str
    enabled: bool
    required: bool
    available: bool
    reason: str | None = None
    error_type: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    exception: BaseException | None = field(default=None, repr=False)

    def public_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "status": self.status,
            "enabled": self.enabled,
            "required": self.required,
            "available": self.available,
            "has_error": self.status == "failed",
        }


async def run_startup_diagnostics(
    services: Any,
    *,
    config: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Probe selected dependencies, print one table, then enforce failure policy."""
    config = config or services.config
    timeout_seconds = _probe_timeout(config)
    checks = list(
        await asyncio.gather(
            _check_database(services.container.task_repository, timeout_seconds),
            _check_storage(services.container.audio_backup_store, timeout_seconds),
            _check_notifications(config, timeout_seconds),
        )
    )
    checks.extend(_logging_checks())
    checks.extend(_runtime_checks(services))

    logger.info("Startup configuration and dependency status:\n%s", _render_table(checks))
    for check in checks:
        _log_check(check)

    failed_required = [
        check.component
        for check in checks
        if check.required and check.status == "failed"
    ]
    if failed_required:
        raise StartupDependencyError(
            "Required startup dependencies are unavailable: "
            + ", ".join(failed_required)
        )

    summary = {check.component: check.public_summary() for check in checks}
    optional_failures = [
        check.component
        for check in checks
        if not check.required and check.status == "failed"
    ]
    logger.info(
        "Startup dependency checks completed: required_ready=true optional_failures=%s",
        optional_failures,
    )
    return summary


async def _check_database(repository: Any, timeout_seconds: float) -> DependencyCheck:
    provider = _database_backend(repository)
    details: dict[str, Any] = {}
    try:
        status = repository.status()
        details["adapter"] = status.get("type") or status.get("name")
        await _call_probe(repository, timeout_seconds)
    except Exception as exc:
        return _failed("database", provider, True, exc, details)
    return DependencyCheck("database", provider, "connected", True, True, True, details=details)


async def _check_storage(store: Any, timeout_seconds: float) -> DependencyCheck:
    provider = str(getattr(store, "name", "unknown"))
    details: dict[str, Any] = {}
    try:
        status = store.status()
        details["adapter"] = status.get("type") or status.get("name")
        await _call_probe(store, timeout_seconds)
    except Exception as exc:
        return _failed("storage", provider, True, exc, details)
    return DependencyCheck("storage", provider, "connected", True, True, True, details=details)


async def _check_notifications(config: Any, timeout_seconds: float) -> DependencyCheck:
    notifications = config.get("notifications", {}) or {}
    provider = str(notifications.get("type", "rocketmq") or "rocketmq").strip().lower()
    if not as_bool(notifications.get("enabled"), False):
        return DependencyCheck(
            "notifications",
            provider,
            "disabled",
            False,
            False,
            True,
            reason="configuration_disabled",
        )

    publisher = RocketMQCompletionPublisher(config)
    issue = publisher.configuration_issue()
    details = {
        "topic_configured": bool(publisher.topic),
        "namesrv_configured": bool(publisher.namesrv),
    }
    if issue:
        return DependencyCheck(
            "notifications",
            provider,
            "failed",
            True,
            False,
            False,
            reason=issue,
            error_type="ConfigurationError",
            details=details,
        )
    try:
        await _call_probe(publisher, timeout_seconds)
    except Exception as exc:
        return _failed("notifications", provider, False, exc, details)
    return DependencyCheck(
        "notifications",
        provider,
        "connected",
        True,
        False,
        True,
        details=details,
    )


def _logging_checks() -> list[DependencyCheck]:
    checks: list[DependencyCheck] = []
    for name, status in get_logging_status().items():
        enabled = bool(status.get("enabled"))
        ready = status.get("status") == "ready"
        checks.append(
            DependencyCheck(
                component=f"logging_{name}",
                provider=str(status.get("provider", name)),
                status="ready" if ready else "disabled",
                enabled=enabled,
                required=bool(name == "file" and enabled),
                available=ready or not enabled,
            )
        )
    return checks


def _runtime_checks(services: Any) -> list[DependencyCheck]:
    modes = ",".join(services.model_manager.enabled_modes) or "none"
    queue_enabled = bool(services.task_queue.enabled)
    return [
        DependencyCheck("engines", modes, "configured", True, False, True),
        DependencyCheck(
            "task_queue",
            "priority_queue",
            "configured" if queue_enabled else "disabled",
            queue_enabled,
            False,
            True,
        ),
    ]


async def _call_probe(component: Any, timeout_seconds: float) -> None:
    probe = getattr(component, "check_connection", None)
    if not callable(probe):
        raise RuntimeError("connection_probe_unavailable")
    await asyncio.wait_for(asyncio.to_thread(probe), timeout=timeout_seconds)


def _probe_timeout(config: Any) -> float:
    raw_value = config.get("startup.dependency_timeout_seconds", 5.0)
    try:
        return min(60.0, max(0.1, float(raw_value)))
    except (TypeError, ValueError):
        return 5.0


def _database_backend(repository: Any) -> str:
    try:
        return str(repository.db.engine.url.get_backend_name())
    except (AttributeError, TypeError):
        return str(getattr(repository, "name", "unknown"))


def _failed(
    component: str,
    provider: str,
    required: bool,
    exc: BaseException,
    details: dict[str, Any],
) -> DependencyCheck:
    return DependencyCheck(
        component=component,
        provider=provider,
        status="failed",
        enabled=True,
        required=required,
        available=False,
        reason=_classify_failure(exc),
        error_type=type(exc).__name__,
        details=details,
        exception=exc,
    )


def _classify_failure(exc: BaseException) -> str:
    if isinstance(exc, ModuleNotFoundError):
        return "client_dependency_missing"
    error_name = type(exc).__name__.lower()
    if "credential" in error_name or "signature" in error_name:
        return "authentication_failed"
    if "timeout" in error_name:
        return "connection_timeout"
    if "endpointconnection" in error_name or isinstance(exc, ConnectionError):
        return "endpoint_unreachable"
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str((response.get("Error") or {}).get("Code", "")).lower()
        if code in {"accessdenied", "invalidaccesskeyid", "signaturedoesnotmatch"}:
            return "authentication_or_permission_denied"
        if code in {"nosuchbucket", "notfound", "404"}:
            return "bucket_not_found"
    if isinstance(exc, OSError):
        return "network_or_filesystem_error"
    return "connection_check_failed"


def _render_table(checks: list[DependencyCheck]) -> str:
    headers = ("COMPONENT", "PROVIDER", "ENABLED", "REQUIRED", "STATUS")
    rows = [
        (
            check.component,
            check.provider,
            "yes" if check.enabled else "no",
            "yes" if check.required else "no",
            check.status,
        )
        for check in checks
    ]
    widths = [
        max(len(headers[index]), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"

    return "\n".join([separator, format_row(headers), separator, *(format_row(row) for row in rows), separator])


def _log_check(check: DependencyCheck) -> None:
    level = logging.WARNING if check.status == "failed" else logging.INFO
    logger.log(
        level,
        "Startup dependency check: component=%s provider=%s status=%s enabled=%s required=%s reason=%s error_type=%s",
        check.component,
        check.provider,
        check.status,
        check.enabled,
        check.required,
        check.reason or "none",
        check.error_type or "none",
    )
    if check.status == "failed":
        logger.debug(
            "Startup dependency failure details: component=%s context=%s",
            check.component,
            json_for_log(check.details),
            exc_info=(
                (type(check.exception), check.exception, check.exception.__traceback__)
                if check.exception is not None
                else None
            ),
        )


__all__ = [
    "DependencyCheck",
    "StartupDependencyError",
    "run_startup_diagnostics",
]
