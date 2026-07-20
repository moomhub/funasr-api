import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.application.file_processing import ArchiveStorage
from src.application.postprocess import FilePostProcessor, StoredFileResult
from src.application.runtime import RuntimeApplication
from src.application.tasks import (
    TaskSubmissionService,
    TaskSubmissionUnavailableError,
    UploadTooLargeError,
    build_submission_response,
)
from src.engine_runtime.services import RuntimeServiceFactory
from src.engine_runtime.services.contracts import ModelPreloadStatus
from src.task_queue.queue import OfflineTaskQueue


class _Config:
    def get(self, key, default=None):
        if key == "limits":
            return {"max_file_size_mb": 1}
        return default

    def get_processing_config(self):
        return SimpleNamespace(
            offline_async_enabled=True,
            offline_async_allow_immediate=True,
            max_concurrent_tasks=1,
            timeout_seconds=60,
        )


class _TempStore:
    def __init__(self, fail=False):
        self.fail = fail
        self.cleaned = []

    async def save_upload(self, upload, task_id, max_size):
        if self.fail:
            raise ValueError("too large")
        return Path(upload.filename), 1024

    def cleanup(self, task_id):
        self.cleaned.append(task_id)


class _Repository:
    def __init__(self):
        self.created = []
        self.errors = []

    def create_task(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id=kwargs["task_id"])

    def record_error(self, task_id, error_message, retry=True):
        self.errors.append((task_id, error_message, retry))
        return SimpleNamespace(id=task_id, status="failed")


class _Scheduler:
    def __init__(self, available=True):
        self.calls = []
        self.available = available

    def can_accept(self, _mode):
        return self.available

    def enqueue_offline(self, task_id, vip=False):
        self.calls.append(("offline", task_id, vip))

    def enqueue_spk(self, task_id, vip=False):
        self.calls.append(("spk", task_id, vip))


def test_task_submission_service_rejects_incomplete_scheduler_contract():
    with pytest.raises(TypeError, match="can_accept"):
        TaskSubmissionService(
            config=_Config(),
            temp_file_store=_TempStore(),
            task_repository=_Repository(),
            speaker_task_repository=_Repository(),
            scheduler=object(),
        )


@pytest.mark.asyncio
async def test_task_submission_service_preserves_offline_response_and_queueing():
    temp = _TempStore()
    repository = _Repository()
    speaker_repository = _Repository()
    scheduler = _Scheduler()
    service = TaskSubmissionService(
        config=_Config(), temp_file_store=temp, task_repository=repository,
        speaker_task_repository=speaker_repository, scheduler=scheduler,
    )

    payload = await service.submit_offline(
        SimpleNamespace(filename="demo.wav"),
        email="user@example.com",
        hotwords='[{"weight":80,"hotword":"测试"}]',
        hotword_id=2,
        vip=True,
    )

    assert payload["status"] == "success"
    assert payload["filename"] == "demo.wav"
    assert payload["file_size"] == 1024
    assert payload["message"] == "任务已加入队列，正在处理"
    assert repository.created[0]["task_id"] == payload["task_id"]
    assert scheduler.calls == [("offline", payload["task_id"], True)]


@pytest.mark.asyncio
async def test_task_submission_debug_logs_input_save_and_output(caplog):
    temp = _TempStore()
    repository = _Repository()
    service = TaskSubmissionService(
        config=_Config(),
        temp_file_store=temp,
        task_repository=repository,
        speaker_task_repository=_Repository(),
        scheduler=_Scheduler(),
    )
    caplog.set_level(logging.DEBUG, logger="src.application.tasks")

    payload = await service.submit_offline(
        SimpleNamespace(filename="debug.wav"),
        email="debug@example.com",
        hotwords=None,
        hotword_id=None,
        vip=False,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Task submission input" in message for message in messages)
    assert any("Task upload saved" in message for message in messages)
    assert any("Task submission output" in message for message in messages)
    assert any(payload["task_id"] in message for message in messages)


@pytest.mark.asyncio
async def test_task_submission_rejects_unavailable_queue_before_file_io():
    temp = _TempStore()
    repository = _Repository()
    service = TaskSubmissionService(
        config=_Config(),
        temp_file_store=temp,
        task_repository=repository,
        speaker_task_repository=_Repository(),
        scheduler=_Scheduler(available=False),
    )

    with pytest.raises(TaskSubmissionUnavailableError):
        await service.submit_offline(
            SimpleNamespace(filename="demo.wav"),
            email=None,
            hotwords=None,
            hotword_id=None,
            vip=False,
        )

    assert repository.created == []
    assert temp.cleaned == []


@pytest.mark.asyncio
async def test_task_submission_marks_created_task_failed_when_enqueue_races_with_shutdown():
    class FailingScheduler(_Scheduler):
        def enqueue_offline(self, task_id, vip=False):
            raise TaskSubmissionUnavailableError("queue stopped")

    temp = _TempStore()
    repository = _Repository()
    service = TaskSubmissionService(
        config=_Config(),
        temp_file_store=temp,
        task_repository=repository,
        speaker_task_repository=_Repository(),
        scheduler=FailingScheduler(),
    )

    with pytest.raises(TaskSubmissionUnavailableError, match="queue stopped"):
        await service.submit_offline(
            SimpleNamespace(filename="demo.wav"),
            email=None,
            hotwords=None,
            hotword_id=None,
            vip=False,
        )

    task_id = repository.created[0]["task_id"]
    assert repository.errors == [
        (task_id, "Task submission failed before queue scheduling", False)
    ]
    assert temp.cleaned == [task_id]


@pytest.mark.asyncio
async def test_task_submission_preserves_file_when_failure_state_cannot_be_persisted():
    class BrokenRepository(_Repository):
        def record_error(self, task_id, error_message, retry=True):
            raise RuntimeError("database unavailable")

    class FailingScheduler(_Scheduler):
        def enqueue_offline(self, task_id, vip=False):
            raise TaskSubmissionUnavailableError("queue stopped")

    temp = _TempStore()
    repository = BrokenRepository()
    service = TaskSubmissionService(
        config=_Config(),
        temp_file_store=temp,
        task_repository=repository,
        speaker_task_repository=_Repository(),
        scheduler=FailingScheduler(),
    )

    with pytest.raises(TaskSubmissionUnavailableError, match="queue stopped"):
        await service.submit_offline(
            SimpleNamespace(filename="demo.wav"),
            email=None,
            hotwords=None,
            hotword_id=None,
            vip=False,
        )

    assert len(repository.created) == 1
    assert temp.cleaned == []


@pytest.mark.asyncio
async def test_task_submission_rejects_large_file_and_cleans_temp_state():
    temp = _TempStore(fail=True)
    service = TaskSubmissionService(
        config=_Config(), temp_file_store=temp, task_repository=_Repository(),
        speaker_task_repository=_Repository(), scheduler=_Scheduler(),
    )

    with pytest.raises(UploadTooLargeError, match="too large"):
        await service.submit_speaker(SimpleNamespace(filename="large.wav"), email=None, vip=False)

    assert len(temp.cleaned) == 1


@pytest.mark.asyncio
async def test_file_postprocessor_keeps_archive_notify_cleanup_order(tmp_path):
    events = []
    audio = tmp_path / "demo.wav"
    audio.write_bytes(b"audio")

    class Hasher:
        async def sha256(self, path):
            events.append("hash")
            return "abc"

    class IndexRepository:
        def get_by_hash(self, value):
            events.append("lookup")
            return None

        def create(self, **kwargs):
            events.append("record")

    class Archive:
        async def store(self, path, task_key, filename, source, file_hash, file_size):
            events.append("archive")
            return StoredFileResult("audio/key", "local", None, "stored.wav", file_hash, file_size)

    class Publisher:
        async def publish_completion(self, **kwargs):
            events.append("notify")
            return True

    class Cleaner:
        def cleanup(self, path):
            events.append("cleanup")

    processor = FilePostProcessor(
        _Config(), publisher=Publisher(), file_index_repository=IndexRepository(),
        hasher=Hasher(), archive_storage=Archive(), cleaner=Cleaner(),
    )
    stored = await processor.handle_complete(
        local_path=str(audio), task_key="task", filename="demo.wav",
        source="offline", email="user@example.com",
    )

    assert stored.s3_key == "audio/key"
    assert events == ["hash", "lookup", "archive", "record", "notify", "cleanup"]


class _ArchiveConfig:
    def __init__(self, local_files_dir, prefix="audio"):
        self.local_files_dir = local_files_dir
        self.prefix = prefix

    def get(self, key, default=None):
        if key == "storage.s3":
            return {"prefix": self.prefix}
        return default

    def get_env(self, _key, _env_name, default=None):
        return default

    def get_runtime_paths(self):
        return {"local_files_dir": str(self.local_files_dir)}


@pytest.mark.asyncio
async def test_archive_storage_delegates_s3_to_injected_store(tmp_path):
    audio = tmp_path / "demo.wav"
    audio.write_bytes(b"audio")

    class BackupStore:
        enabled = True
        bucket = "archive-bucket"

        def __init__(self):
            self.calls = []

        async def backup_original(self, local_path, task_id, filename):
            self.calls.append((local_path, task_id, filename))
            return "archive/task-1_uploaded.wav"

    backup_store = BackupStore()
    storage = ArchiveStorage(
        _ArchiveConfig(tmp_path / "local"),
        audio_backup_store=backup_store,
    )

    stored = await storage.store(
        audio,
        "task-1",
        "demo.wav",
        "offline",
        "abc123",
        audio.stat().st_size,
    )

    assert backup_store.calls == [(str(audio), "task-1", "demo.wav")]
    assert stored.storage_backend == "s3"
    assert stored.s3_key == "archive/task-1_uploaded.wav"
    assert stored.bucket_name == "archive-bucket"
    assert stored.stored_filename == "task-1_uploaded.wav"
    assert not (tmp_path / "local").exists()


@pytest.mark.asyncio
async def test_archive_storage_uses_local_fallback_when_s3_returns_no_key(tmp_path, caplog):
    caplog.set_level("INFO")
    audio = tmp_path / "demo.wav"
    audio.write_bytes(b"audio")

    class BackupStore:
        enabled = True
        bucket = "archive-bucket"

        async def backup_original(self, _local_path, _task_id, _filename):
            return None

    storage = ArchiveStorage(
        _ArchiveConfig(tmp_path / "local", prefix="archive"),
        audio_backup_store=BackupStore(),
    )

    stored = await storage.store(
        audio,
        "task-2",
        "demo.wav",
        "spk",
        "def456",
        audio.stat().st_size,
    )

    assert stored.storage_backend == "local"
    assert stored.bucket_name is None
    assert stored.s3_key.startswith("archive/task-2_")
    assert Path(stored.local_path).read_bytes() == b"audio"
    assert Path(stored.local_path).parent == tmp_path / "local" / "archive"
    assert "demo.wav" not in caplog.text
    assert str(tmp_path) not in caplog.text


def test_runtime_application_reports_one_shared_preload_source():
    manager = SimpleNamespace(
        enabled_modes=["offline"],
        get_backend_for_mode=lambda mode: "pt",
        get_inference_backends=lambda: {"offline": "pytorch"},
        get_loaded_models_count=lambda: 1,
    )

    class Factory:
        def preload_enabled_models(self):
            return [
                ModelPreloadStatus("offline_asr", "offline", "pt", True),
                ModelPreloadStatus("speaker_pt", "spk", "pt", True),
            ]

        def get_model_status(self):
            return {"offline_asr": {"name": "offline_asr", "loaded": True}}

        def required_service_modes(self, mode):
            return ["offline", "spk"] if mode == "offline" else [mode]

    application = RuntimeApplication(manager, Factory())

    assert application.preload_enabled_models() == {"loaded": ["offline"], "failed": {}}
    status = application.get_runtime_status()
    assert status["offline_asr"]["loaded"] is True
    assert "offline_model_runtime" not in status
    assert application.get_inference_backends() == {"offline": "pytorch"}
    assert application.get_loaded_models_count() == 1


def test_runtime_application_marks_offline_unavailable_when_speaker_dependency_fails():
    manager = SimpleNamespace(
        enabled_modes=["offline"],
        get_backend_for_mode=lambda mode: "pt",
        get_inference_backends=lambda: {"offline": "pytorch"},
        get_loaded_models_count=lambda: 1,
    )

    class Factory:
        def preload_enabled_models(self):
            return [
                ModelPreloadStatus("offline_asr", "offline", "pt", True),
                ModelPreloadStatus("speaker_pt", "spk", "pt", False, error="speaker failed"),
            ]

        def get_model_status(self):
            return {}

        def required_service_modes(self, mode):
            return ["offline", "spk"] if mode == "offline" else [mode]

    application = RuntimeApplication(manager, Factory())

    assert application.preload_enabled_models() == {
        "loaded": [],
        "failed": {"offline": "speaker_pt: speaker failed"},
    }
    assert application.is_mode_available("offline") is False
    assert application.get_engine_info()["offline"]["available"] is False


def test_runtime_service_factory_preloads_speaker_as_offline_dependency():
    calls = []

    class Manager:
        enabled_modes = ["offline"]

        def get_backend_for_mode(self, mode):
            return "pt"

        def get_offline_pt_recognizer(self):
            return SimpleNamespace(load_model=lambda: calls.append("offline_asr"))

        def get_spk_pt_recognizer(self):
            return SimpleNamespace(load_model=lambda: calls.append("speaker"))

    statuses = RuntimeServiceFactory(Manager()).preload_enabled_models()

    assert [status.service_name for status in statuses] == ["offline_asr_pt", "speaker_pt"]
    assert [status.loaded for status in statuses] == [True, True]
    assert calls == ["offline_asr", "speaker"]


def test_build_submission_response_formats_common_upload_payload():
    assert build_submission_response(
        task_id="task-1",
        filename="demo.wav",
        file_size=1536,
        email="user@example.com",
        vip=True,
        message="queued",
        extra={"hotword_id": 2},
    ) == {
        "status": "success",
        "task_id": "task-1",
        "filename": "demo.wav",
        "file_size": 1536,
        "file_size_mb": 0.0,
        "email": "user@example.com",
        "vip": True,
        "message": "queued",
        "hotword_id": 2,
    }


@pytest.mark.asyncio
async def test_task_submission_service_preserves_spk_response_and_queueing():
    temp = _TempStore()
    repository = _Repository()
    scheduler = _Scheduler()
    service = TaskSubmissionService(
        config=_Config(), temp_file_store=temp, task_repository=_Repository(),
        speaker_task_repository=repository, scheduler=scheduler,
    )

    payload = await service.submit_speaker(
        SimpleNamespace(filename="speaker.wav"),
        email="speaker@example.com",
        vip=True,
    )

    assert payload["status"] == "success"
    assert payload["filename"] == "speaker.wav"
    assert payload["file_size"] == 1024
    assert payload["message"] == "SPK 任务已加入队列，正在处理"
    assert repository.created[0]["task_id"] == payload["task_id"]
    assert scheduler.calls == [("spk", payload["task_id"], True)]


@pytest.mark.asyncio
async def test_task_queue_keeps_vip_then_fifo_and_deduplicates():
    queue = OfflineTaskQueue(
        config=_Config(), task_repository=object(), batch_result_handler=object(), task_service=object(), speaker_task_repository=None,
    )
    queue._runtime.enqueue("offline", "normal-1")
    queue._runtime.enqueue("offline", "vip", vip=True)
    queue._runtime.enqueue("offline", "normal-2")
    queue._runtime.enqueue("offline", "normal-1")

    items = [await queue._queue.get() for _ in range(3)]
    assert [(item[2], item[3]) for item in items] == [
        ("offline", "vip"),
        ("offline", "normal-1"),
        ("offline", "normal-2"),
    ]


def test_task_queue_rejects_public_enqueue_until_workers_are_running():
    queue = OfflineTaskQueue(
        config=_Config(), task_repository=object(), batch_result_handler=object(), task_service=object(), speaker_task_repository=None,
    )

    assert queue.can_accept("offline") is False
    with pytest.raises(TaskSubmissionUnavailableError):
        queue.enqueue_offline("task-1")


@pytest.mark.asyncio
async def test_task_queue_delegates_spk_work_to_task_service():
    class SpkTaskService:
        def __init__(self):
            self.calls = []

        async def process_task(self, task_id):
            self.calls.append(task_id)
            return True

    spk_task_service = SpkTaskService()
    queue = OfflineTaskQueue(
        config=_Config(), task_repository=object(), batch_result_handler=object(), task_service=object(), speaker_task_repository=None,
        spk_task_service=spk_task_service,
    )

    assert await queue._process_spk_task("spk-1") is True
    assert spk_task_service.calls == ["spk-1"]
