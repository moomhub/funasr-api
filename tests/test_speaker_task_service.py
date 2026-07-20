import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.postprocess import StoredFileResult
from src.application.speaker import SpkTaskService
from src.core.results import SpeakerResult, SpeakerSegment


class _Repository:
    def __init__(self):
        self.task = SimpleNamespace(
            id="spk-1",
            filename="demo.wav",
            email="user@example.com",
            vip=False,
        )
        self.statuses = []
        self.saved = None
        self.error = None

    def get_task(self, task_id):
        return self.task if task_id == self.task.id else None

    def update_status(self, task_id, status):
        self.statuses.append((task_id, status))

    def save_result(self, **kwargs):
        self.saved = kwargs

    def record_error(self, task_id, error_message, retry=True):
        self.error = (task_id, error_message, retry)


class _TempStore:
    def __init__(self, path):
        self.path = path
        self.cleaned = []

    def resolve(self, task_id, filename):
        return self.path

    def cleanup(self, task_id):
        self.cleaned.append(task_id)


class _SpeakerService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def diarize(self, audio_path, *, metadata=None, generate_kwargs=None):
        self.calls.append(
            {
                "audio_path": audio_path,
                "metadata": metadata,
                "generate_kwargs": generate_kwargs,
            }
        )
        return self.result


class _Postprocessor:
    def __init__(self, stored=None):
        self.stored = stored
        self.calls = []

    async def handle_complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.stored


@pytest.mark.asyncio
async def test_spk_task_service_processes_task_and_saves_result(tmp_path):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")
    repository = _Repository()
    temp_store = _TempStore(audio_path)
    speaker_service = _SpeakerService(
        SpeakerResult(
            segments=[SpeakerSegment(speaker="A", start=0, end=1000)],
            speaker_ids=["A"],
            speaker_count=1,
        )
    )
    postprocessor = _Postprocessor(
        StoredFileResult("audio/key", "s3", None, "stored.wav", "sha256", 5)
    )
    service = SpkTaskService(
        speaker_task_repository=repository,
        temp_file_store=temp_store,
        speaker_service=speaker_service,
        postprocessor=postprocessor,
    )

    ok = await service.process_task("spk-1")

    assert ok is True
    assert repository.statuses == [("spk-1", "processing")]
    assert speaker_service.calls == [
        {
            "audio_path": str(audio_path),
            "metadata": {"task_id": "spk-1", "filename": "demo.wav"},
            "generate_kwargs": None,
        }
    ]
    assert postprocessor.calls[0]["source"] == "spk"
    assert postprocessor.calls[0]["email"] == "user@example.com"
    assert repository.saved["task_id"] == "spk-1"
    assert repository.saved["result"]["speaker_ids"] == ["A"]
    assert repository.saved["s3_key"] == "audio/key"
    assert repository.saved["file_hash"] == "sha256"
    assert temp_store.cleaned == ["spk-1"]


@pytest.mark.asyncio
async def test_spk_task_service_records_error_when_audio_missing(tmp_path):
    repository = _Repository()
    service = SpkTaskService(
        speaker_task_repository=repository,
        temp_file_store=_TempStore(tmp_path / "missing.wav"),
        speaker_service=_SpeakerService(SpeakerResult()),
        postprocessor=_Postprocessor(),
    )

    ok = await service.process_task("spk-1")

    assert ok is False
    assert repository.error[0] == "spk-1"
    assert "音频文件不存在" in repository.error[1]
    assert repository.error[2] is True
