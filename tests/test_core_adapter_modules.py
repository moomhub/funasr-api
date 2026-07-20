import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.core.container as container_module
from src.core.adapters import (
    DatabaseHotwordProvider as LegacyDatabaseHotwordProvider,
    EmptyHotwordProvider as LegacyEmptyHotwordProvider,
    LocalTempFileStore as LegacyLocalTempFileStore,
    MemoryTaskRepository as LegacyMemoryTaskRepository,
    NoopAudioBackupStore as LegacyNoopAudioBackupStore,
    S3AudioBackupStore as LegacyS3AudioBackupStore,
)
from src.core.infrastructure_adapters.audio_backup import (
    NoopAudioBackupStore,
    S3AudioBackupStore,
)
from src.core.infrastructure_adapters.hotwords import (
    DatabaseHotwordProvider,
    EmptyHotwordProvider,
)
from src.core.infrastructure_adapters.task_repository import MemoryTaskRepository
from src.core.infrastructure_adapters.temp_files import LocalTempFileStore


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_adapter_module_reexports_responsibility_scoped_classes():
    assert LegacyMemoryTaskRepository is MemoryTaskRepository
    assert LegacyLocalTempFileStore is LocalTempFileStore
    assert LegacyNoopAudioBackupStore is NoopAudioBackupStore
    assert LegacyS3AudioBackupStore is S3AudioBackupStore
    assert LegacyEmptyHotwordProvider is EmptyHotwordProvider
    assert LegacyDatabaseHotwordProvider is DatabaseHotwordProvider
    assert MemoryTaskRepository.__module__.endswith(".task_repository")
    assert LocalTempFileStore.__module__.endswith(".temp_files")
    assert S3AudioBackupStore.__module__.endswith(".audio_backup")
    assert DatabaseHotwordProvider.__module__.endswith(".hotwords")


def test_legacy_adapter_module_contains_no_implementation_classes():
    path = ROOT / "src" / "core" / "adapters.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)


def test_infrastructure_builder_imports_concrete_adapter_modules_directly():
    path = ROOT / "src" / "core" / "infrastructure.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert "src.core.adapters" not in imported_modules
    assert "src.core.infrastructure_adapters.audio_backup" in imported_modules
    assert "src.core.infrastructure_adapters.hotwords" in imported_modules
    assert "src.core.infrastructure_adapters.temp_files" in imported_modules


class _Upload:
    filename = "../../patient-recording.wav"

    def __init__(self):
        self.chunks = [b"audio", b""]

    async def read(self, _size):
        return self.chunks.pop(0)


@pytest.mark.asyncio
async def test_temp_file_adapter_sanitizes_filename_and_resets_health(tmp_path):
    store = LocalTempFileStore(str(tmp_path))
    store.last_error = "previous failure"

    saved_path, file_size = await store.save_upload(_Upload(), "task-1", 1024)

    assert saved_path == tmp_path / "task-1" / "patient-recording.wav"
    assert file_size == 5
    assert store.available is True
    assert store.last_error is None


def test_database_hotword_provider_tracks_failure_and_recovery(caplog):
    class Repository:
        broken = True

        def get_formatted_list(self):
            if self.broken:
                raise RuntimeError("private database response")
            return [{"hotword": "启动", "weight": 80}]

    repository = Repository()
    provider = DatabaseHotwordProvider(repository)
    with caplog.at_level(
        logging.DEBUG,
        logger="src.core.infrastructure_adapters.hotwords",
    ):
        assert provider.get_hotwords() == []

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert provider.available is False
    assert provider.last_error == "private database response"
    assert warning_messages == [
        "Database hotwords unavailable, using empty list: error_type=RuntimeError"
    ]
    assert all("private database response" not in message for message in warning_messages)

    repository.broken = False
    assert provider.get_hotwords() == [{"hotword": "启动", "weight": 80}]
    assert provider.available is True
    assert provider.last_error is None


def test_container_info_log_excludes_detailed_component_status(monkeypatch, caplog):
    class Component:
        name = "component"
        enabled = True
        available = True
        last_error = None

        def status(self):
            return {
                "name": self.name,
                "type": type(self).__name__,
                "enabled": self.enabled,
                "available": self.available,
                "last_error": self.last_error,
                "root": "C:/private/runtime/path",
            }

        def close(self):
            return None

    task_repository = Component()
    repositories = SimpleNamespace(
        task_repository=task_repository,
        speaker_task_repository=object(),
        file_index_repository=object(),
        hotword_repository=object(),
    )
    monkeypatch.setattr(container_module, "build_task_repository", lambda _config: task_repository)
    monkeypatch.setattr(container_module, "build_repository_bundle", lambda _repo: repositories)
    monkeypatch.setattr(container_module, "build_temp_file_store", lambda _config: Component())
    monkeypatch.setattr(container_module, "build_audio_backup_store", lambda _config: Component())
    monkeypatch.setattr(
        container_module,
        "build_hotword_provider",
        lambda _config, _task_repo, _hotword_repo: Component(),
    )

    with caplog.at_level(logging.DEBUG, logger="src.core.container"):
        container_module.build_container(object())

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert all("C:/private/runtime/path" not in message for message in info_messages)
    assert any("C:/private/runtime/path" in message for message in debug_messages)
