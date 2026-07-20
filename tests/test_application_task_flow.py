import logging
from types import SimpleNamespace

import pytest

from src.application.task_flow import (
    load_task_file_ref,
    mark_task_processing,
    record_task_error,
    require_audio_file,
)


class _Repository:
    def __init__(self, task=None):
        self.task = task
        self.statuses = []
        self.errors = []

    def get_task(self, task_id):
        return self.task if self.task and task_id == self.task.id else None

    def update_status(self, task_id, status):
        self.statuses.append((task_id, status))

    def record_error(self, task_id, error_message, retry=True):
        self.errors.append((task_id, error_message, retry))


class _TempStore:
    def __init__(self, path):
        self.path = path

    def resolve(self, task_id, filename):
        return self.path


@pytest.mark.asyncio
async def test_load_task_file_ref_resolves_existing_task_file(tmp_path):
    audio_path = tmp_path / "demo.wav"
    repository = _Repository(SimpleNamespace(id="task-1", filename="demo.wav"))

    task_ref = await load_task_file_ref(
        repository=repository,
        temp_file_store=_TempStore(audio_path),
        task_id="task-1",
        logger=logging.getLogger(__name__),
        missing_message="missing %s",
    )

    assert task_ref.task_id == "task-1"
    assert task_ref.filename == "demo.wav"
    assert task_ref.audio_path == audio_path


@pytest.mark.asyncio
async def test_load_task_file_ref_returns_none_for_missing_task(caplog, tmp_path):
    with caplog.at_level(logging.ERROR):
        task_ref = await load_task_file_ref(
            repository=_Repository(),
            temp_file_store=_TempStore(tmp_path / "missing.wav"),
            task_id="missing",
            logger=logging.getLogger(__name__),
            missing_message="missing task %s",
        )

    assert task_ref is None
    assert "missing task missing" in caplog.text


@pytest.mark.asyncio
async def test_task_flow_marks_processing_and_records_retryable_error():
    repository = _Repository()

    await mark_task_processing(repository, "task-1")
    await record_task_error(repository, "task-1", "failed", retry=False)

    assert repository.statuses == [("task-1", "processing")]
    assert repository.errors == [("task-1", "failed", False)]


def test_require_audio_file_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="音频文件不存在"):
        require_audio_file(tmp_path / "missing.wav")
