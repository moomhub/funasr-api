import logging
from types import SimpleNamespace

import pytest

from src.application.lifecycle import ApplicationLifecycle


class _Config:
    def get(self, key, default=None):
        if key == "logging.level":
            return "DEBUG"
        return default


class _Container:
    def __init__(self):
        self.shutdown_calls = 0

    def get_status(self):
        return {
            "task_repository": {
                "name": "sql",
                "enabled": True,
                "available": True,
                "path": "private/runtime/path",
            }
        }

    def shutdown(self):
        self.shutdown_calls += 1


class _Queue:
    def __init__(self, enabled=True, stop_error=None):
        self.enabled = enabled
        self.is_running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.stop_error = stop_error

    def supports(self, mode):
        return mode in {"offline", "spk"}

    def start(self):
        self.start_calls += 1
        self.is_running = True

    async def stop(self):
        self.stop_calls += 1
        self.is_running = False
        if self.stop_error is not None:
            raise self.stop_error


class _Runtime:
    def __init__(self, available, preload_summary=None):
        self.available = dict(available)
        self.preload_summary = preload_summary or {"loaded": [], "failed": {}}

    def preload_enabled_models(self):
        return self.preload_summary

    def is_mode_available(self, mode):
        return bool(self.available.get(mode, False))

    def get_enabled_modes(self):
        return [mode for mode, enabled in self.available.items() if enabled]


def _services(*, available, queue=None, preload_summary=None):
    return SimpleNamespace(
        container=_Container(),
        task_queue=queue or _Queue(),
        runtime_application=_Runtime(available, preload_summary),
        model_manager=SimpleNamespace(
            enabled_modes=list(available),
            auto_download=False,
            get_inference_backends=lambda: {mode: "pt" for mode in available},
        ),
    )


@pytest.mark.asyncio
async def test_lifecycle_reuses_loaded_config_and_starts_queue_for_available_offline():
    config = _Config()
    services = _services(
        available={"offline": True, "spk": False},
        preload_summary={"loaded": ["offline"], "failed": {}},
    )
    config_paths = []
    factory_calls = []

    def config_factory(config_path):
        config_paths.append(config_path)
        return config

    def services_factory(**kwargs):
        factory_calls.append(kwargs)
        return services

    lifecycle = ApplicationLifecycle(
        "demo.yaml",
        config_factory=config_factory,
        services_factory=services_factory,
    )
    app = SimpleNamespace(state=SimpleNamespace())

    returned = await lifecycle.start(app)

    assert returned is services
    assert app.state.services is services
    assert config_paths == ["demo.yaml"]
    assert factory_calls == [{"config_loader": config}]
    assert services.task_queue.start_calls == 1
    assert services.task_queue.is_running is True


@pytest.mark.asyncio
async def test_lifecycle_does_not_start_queue_without_available_task_runtime():
    services = _services(
        available={"offline": False, "spk": False},
        preload_summary={
            "loaded": [],
            "failed": {"offline": "private model path and payload"},
        },
    )
    lifecycle = ApplicationLifecycle(
        config_factory=lambda _path: _Config(),
        services_factory=lambda **_kwargs: services,
    )
    app = SimpleNamespace(state=SimpleNamespace())

    await lifecycle.start(app)

    assert services.task_queue.start_calls == 0
    assert services.task_queue.is_running is False


@pytest.mark.asyncio
async def test_lifecycle_shutdown_continues_after_queue_error_and_logs_details_only_at_debug(caplog):
    secret = "private shutdown details"
    queue = _Queue(stop_error=RuntimeError(secret))
    services = _services(available={"offline": True}, queue=queue)
    app = SimpleNamespace(state=SimpleNamespace(services=services))
    lifecycle = ApplicationLifecycle()

    with caplog.at_level(logging.DEBUG, logger="src.application.lifecycle"):
        await lifecycle.stop(app)

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert queue.stop_calls == 1
    assert services.container.shutdown_calls == 1
    assert all(secret not in message for message in warning_messages)
    assert any("Task queue shutdown failure details" in message for message in debug_messages)
