import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.application.startup_diagnostics import run_startup_diagnostics


class _Config:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def get_env(self, _key, _env_name, default=None):
        return default


class _Repository:
    name = "sql"

    def __init__(self, error=None, fallback_error=None):
        self.error = error
        self.calls = 0
        self.db = SimpleNamespace(
            engine=SimpleNamespace(
                url=SimpleNamespace(get_backend_name=lambda: "sqlite")
            )
        )
        self.fallback_error = fallback_error

    def check_connection(self):
        self.calls += 1
        if self.error:
            raise self.error

    def status(self):
        return {
            "name": "sql",
            "type": "SqlTaskRepository",
            "enabled": True,
            "available": self.error is None,
            "last_error": self.fallback_error,
            "fallback_error_type": "OperationalError" if self.fallback_error else None,
        }


class _Store:
    def __init__(self, error=None, enabled=True, last_error=None):
        self.error = error
        self.enabled = enabled
        self.last_error = last_error
        self.calls = 0

    def check_connection(self):
        self.calls += 1
        if self.error:
            raise self.error

    def status(self):
        return {
            "name": "s3" if self.enabled else "noop",
            "type": "S3AudioBackupStore" if self.enabled else "NoopAudioBackupStore",
            "enabled": self.enabled,
            "available": self.error is None,
            "last_error": self.last_error,
            "bucket": "private-bucket",
        }


def _services(config, repository=None, store=None):
    return SimpleNamespace(
        config=config,
        container=SimpleNamespace(
            task_repository=repository or _Repository(),
            audio_backup_store=store or _Store(enabled=False),
        ),
    )


@pytest.mark.asyncio
async def test_startup_diagnostics_reports_connected_database_and_disabled_optionals(caplog):
    repository = _Repository()
    services = _services(_Config(), repository=repository)

    with caplog.at_level(logging.INFO, logger="src.application.startup_diagnostics"):
        result = await run_startup_diagnostics(services)

    assert repository.calls == 1
    assert result == {
        "database": {
            "status": "connected",
            "enabled": True,
            "available": True,
            "has_error": False,
        },
        "s3": {
            "status": "disabled",
            "enabled": False,
            "available": True,
            "has_error": False,
        },
        "notifications": {
            "status": "disabled",
            "enabled": False,
            "available": True,
            "has_error": False,
        },
    }
    assert "component=database provider=sqlite status=connected" in caplog.text


@pytest.mark.asyncio
async def test_startup_diagnostics_classifies_s3_failure_without_leaking_details(caplog):
    secret = "secret endpoint and credential payload"
    store = _Store(error=ConnectionError(secret))
    config = _Config({"storage.s3": {"enabled": True}})

    with caplog.at_level(logging.WARNING, logger="src.application.startup_diagnostics"):
        result = await run_startup_diagnostics(_services(config, store=store))

    assert store.calls == 1
    assert result["s3"] == {
        "status": "failed",
        "enabled": True,
        "available": False,
        "has_error": True,
    }
    assert "reason=endpoint_unreachable" in caplog.text
    assert "error_type=ConnectionError" in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_startup_diagnostics_reports_notification_configuration_error(caplog):
    config = _Config(
        {
            "notifications": {
                "enabled": True,
                "type": "rocketmq",
                "rocketmq": {"namesrv": "", "topic": "events"},
            }
        }
    )

    with caplog.at_level(logging.WARNING, logger="src.application.startup_diagnostics"):
        result = await run_startup_diagnostics(_services(config))

    assert result["notifications"]["status"] == "failed"
    assert "component=notifications provider=rocketmq status=failed" in caplog.text
    assert "reason=configuration_incomplete" in caplog.text


@pytest.mark.asyncio
async def test_startup_diagnostics_marks_database_fallback_as_degraded(caplog):
    repository = _Repository(fallback_error="private database connection details")

    with caplog.at_level(logging.WARNING, logger="src.application.startup_diagnostics"):
        result = await run_startup_diagnostics(
            _services(_Config(), repository=repository)
        )

    assert result["database"]["status"] == "degraded"
    assert "reason=configured_database_unavailable" in caplog.text
    assert "private database connection details" not in caplog.text


@pytest.mark.asyncio
async def test_startup_diagnostics_probes_rocketmq_producer(monkeypatch):
    calls = []

    class Producer:
        def __init__(self, group):
            calls.append(("create", group))

        def set_namesrv_addr(self, namesrv):
            calls.append(("namesrv", namesrv))

        def start(self):
            calls.append(("start", None))

        def shutdown(self):
            calls.append(("shutdown", None))

    package = ModuleType("rocketmq")
    client = ModuleType("rocketmq.client")
    client.Producer = Producer
    monkeypatch.setitem(sys.modules, "rocketmq", package)
    monkeypatch.setitem(sys.modules, "rocketmq.client", client)
    config = _Config(
        {
            "notifications": {
                "enabled": True,
                "type": "rocketmq",
                "rocketmq": {
                    "namesrv": "mq.internal:9876",
                    "topic": "events",
                    "group": "api-producer",
                },
            }
        }
    )

    result = await run_startup_diagnostics(_services(config))

    assert result["notifications"]["status"] == "connected"
    assert calls == [
        ("create", "api-producer"),
        ("namesrv", "mq.internal:9876"),
        ("start", None),
        ("shutdown", None),
    ]
