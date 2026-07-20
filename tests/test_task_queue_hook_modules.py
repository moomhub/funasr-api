import ast
import logging
from pathlib import Path

import pytest

from src.application.task_results import OfflineTaskContext
from src.core.results import RecognitionResult
from src.task_queue.hook_execution import SequentialHookExecutor
from src.task_queue.hook_handlers import (
    OfflineBatchResultHandler,
    OfflineTaskResultHandler,
)
from src.task_queue.hooks import (
    OfflineBatchResultHandler as LegacyBatchResultHandler,
    OfflineTaskResultHandler as LegacyTaskResultHandler,
    ResultPersistenceHook as LegacyResultPersistenceHook,
    TempCleanupHook as LegacyTempCleanupHook,
)
from src.task_queue.result_hooks import ResultPersistenceHook, TempCleanupHook


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_hook_module_reexports_split_implementations():
    assert LegacyTaskResultHandler is OfflineTaskResultHandler
    assert LegacyBatchResultHandler is OfflineBatchResultHandler
    assert LegacyResultPersistenceHook is ResultPersistenceHook
    assert LegacyTempCleanupHook is TempCleanupHook
    assert OfflineTaskResultHandler.__module__.endswith(".hook_handlers")
    assert ResultPersistenceHook.__module__.endswith(".result_hooks")


def test_legacy_hook_module_contains_no_implementation_classes():
    path = ROOT / "src" / "task_queue" / "hooks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)


def test_composition_imports_hook_handlers_directly():
    path = ROOT / "src" / "composition.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert "src.task_queue.hooks" not in modules
    assert "src.task_queue.hook_handlers" in modules


class _Hook:
    def __init__(self, name, *, critical=False, error=None, calls=None):
        self.name = name
        self.critical = critical
        self.error = error
        self.calls = calls if calls is not None else []

    async def run(self):
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
async def test_hook_executor_continues_optional_failure_and_redacts_warning(caplog):
    calls = []
    hooks = [
        _Hook(
            "optional",
            error=RuntimeError("private hook payload"),
            calls=calls,
        ),
        _Hook("next", calls=calls),
    ]
    executor = SequentialHookExecutor(logging.getLogger("tests.hooks.optional"))

    with caplog.at_level(logging.DEBUG, logger="tests.hooks.optional"):
        await executor.run(
            hooks,
            phase="test",
            invoke=lambda hook: hook.run(),
            raise_critical=True,
        )

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert calls == ["optional", "next"]
    assert warning_messages == [
        "Hook execution failed: phase=test hook=optional error_type=RuntimeError"
    ]
    assert all("private hook payload" not in message for message in warning_messages)


@pytest.mark.asyncio
async def test_hook_executor_raises_critical_failure_and_stops_following_hooks():
    calls = []
    hooks = [
        _Hook(
            "critical",
            critical=True,
            error=RuntimeError("failure"),
            calls=calls,
        ),
        _Hook("not-run", calls=calls),
    ]
    executor = SequentialHookExecutor(logging.getLogger("tests.hooks.critical"))

    with pytest.raises(RuntimeError, match="failure"):
        await executor.run(
            hooks,
            phase="test",
            invoke=lambda hook: hook.run(),
            raise_critical=True,
        )

    assert calls == ["critical"]


@pytest.mark.asyncio
async def test_failure_phase_does_not_mask_original_error_with_critical_hook():
    calls = []
    hooks = [
        _Hook(
            "critical",
            critical=True,
            error=RuntimeError("secondary failure"),
            calls=calls,
        ),
        _Hook("next", calls=calls),
    ]
    executor = SequentialHookExecutor(logging.getLogger("tests.hooks.failure"))

    await executor.run(
        hooks,
        phase="offline_failure",
        invoke=lambda hook: hook.run(),
        raise_critical=False,
    )

    assert calls == ["critical", "next"]


@pytest.mark.asyncio
async def test_temp_cleanup_hook_delegates_to_temp_file_store():
    class Store:
        def __init__(self):
            self.cleaned = []

        def cleanup(self, task_id):
            self.cleaned.append(task_id)

    store = Store()
    hook = TempCleanupHook(store)
    context = OfflineTaskContext(
        task_id="task-1",
        filename="private.wav",
        audio_path="C:/private/private.wav",
        metadata={"backup_key": "archive/task-1.wav"},
    )

    await hook.on_success(context, RecognitionResult(mode="offline"))

    assert store.cleaned == ["task-1"]


@pytest.mark.asyncio
async def test_persistence_info_log_excludes_filename_and_recognition_text(caplog):
    class Repository:
        def save_result(self, **kwargs):
            return object()

    hook = ResultPersistenceHook(Repository())
    context = OfflineTaskContext(
        task_id="task-1",
        filename="private-recording.wav",
        audio_path="C:/private/private-recording.wav",
    )

    with caplog.at_level(
        logging.INFO,
        logger="src.task_queue.result_hooks",
    ):
        await hook.on_success(
            context,
            RecognitionResult(mode="offline", full_text="private transcript"),
        )

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert all("private-recording.wav" not in message for message in info_messages)
    assert all("private transcript" not in message for message in info_messages)
