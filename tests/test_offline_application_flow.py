import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.application.offline import OfflineRecognitionService, OfflineTaskService
from src.core.config.loader import ConfigLoader
from src.core.results import RecognitionResult, Segment
from src.engine_runtime.manager import EngineModelManager
from src.engine_runtime.services.contracts import ModelPreloadStatus


class _FakeOfflineRuntimeService:
    backend = "pt"

    def __init__(self):
        self._loaded = False
        self.requests = []

    @property
    def is_loaded(self):
        return self._loaded

    def preload(self):
        self._loaded = True
        return ModelPreloadStatus(
            service_name="offline_asr_pt",
            mode="offline",
            backend="pt",
            loaded=True,
        )

    async def recognize(self, request):
        self.requests.append(request)
        return RecognitionResult(
            mode="offline",
            full_text="hello",
            segments=[Segment(text="hello", start=0, end=100)],
        )


def test_engine_manager_reads_common_offline_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - offline
  device: cpu
  disable_model_update: true
  auto_model_download: false
  models:
    offline:
      enabled: onnx
runtime:
  models_dir: ./damo
""",
        encoding="utf-8",
    )

    manager = EngineModelManager(ConfigLoader(str(config_path)))

    assert manager.get_backend_for_mode("offline") == "onnx"
    assert manager.model_dir == str(Path("data") / "damo")
    assert manager.auto_download is False
    assert manager.pt_device == "cpu"
    assert manager.pt_disable_update is True


def test_engine_manager_reads_offline_onnx_runtime_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - offline
  models:
    offline:
      enabled: onnx
      onnx_runtime:
        asr_workers: 9
        load_workers: 8
        sample_rate: 8000
        vad_padding_ms: 120
""",
        encoding="utf-8",
    )

    manager = EngineModelManager(ConfigLoader(str(config_path)))

    assert manager.offline_onnx.asr_workers == 9
    assert manager.offline_onnx.load_workers == 8
    assert manager.offline_onnx.sample_rate == 8000
    assert manager.offline_onnx.vad_padding_ms == 120


@pytest.mark.asyncio
async def test_offline_recognition_service_uses_runtime_facade():
    runtime = _FakeOfflineRuntimeService()
    service = OfflineRecognitionService(runtime)

    result = await service.recognize("demo.wav", hotwords=["热词"])

    assert runtime.is_loaded is True
    assert result.full_text == "hello"
    assert result.metadata["service"] == "offline_asr"
    assert runtime.requests[0].audio_path == "demo.wav"
    assert runtime.requests[0].hotwords == ["热词"]


@pytest.mark.asyncio
async def test_offline_recognition_service_reuses_preloaded_runtime():
    runtime = _FakeOfflineRuntimeService()
    runtime.preload()
    service = OfflineRecognitionService(runtime)

    first = await service.recognize("first.wav")
    second = await service.recognize("second.wav")

    assert first.error is None
    assert second.error is None
    assert [request.audio_path for request in runtime.requests] == ["first.wav", "second.wav"]


class _FakeRepository:
    def __init__(self):
        self.task = SimpleNamespace(
            id="task-1",
            filename="demo.wav",
            hotwords=["热词"],
            hotword_id=None,
            email=None,
            vip=False,
        )
        self.statuses = []
        self.saved = None
        self.error = None

    def get_task(self, task_id):
        return self.task if task_id == self.task.id else None

    def update_status(self, task_id, status):
        self.statuses.append((task_id, status))

    def save_result(self, task_id, full_text, segments, processing_time):
        self.saved = {
            "task_id": task_id,
            "full_text": full_text,
            "segments": segments,
            "processing_time": processing_time,
        }
        return self.task

    def record_error(self, task_id, error_message, retry=False):
        self.error = (task_id, error_message, retry)


class _FakeTempStore:
    def __init__(self, path):
        self.path = path

    def resolve(self, task_id, filename):
        return self.path


class _FakeResultHandler:
    def __init__(self, repository):
        self.repository = repository

    async def handle_success(self, context, result):
        self.repository.save_result(
            context.task_id,
            result.full_text,
            [segment.to_dict() for segment in result.segments],
            result.processing_time,
        )

    async def handle_failure(self, context, error_message):
        self.repository.record_error(context.task_id, error_message)


class _FakeConfig:
    def get(self, key, default=None):
        return default


def test_offline_task_service_logs_logical_custom_hotword_count(caplog):
    service = OfflineTaskService(
        task_repository=object(),
        temp_file_store=object(),
        result_handler=object(),
        recognition_service=OfflineRecognitionService(_FakeOfflineRuntimeService()),
        config=_FakeConfig(),
    )
    task = SimpleNamespace(
        hotwords='[{"weight":100,"hotword":"篮子"},{"weight":80,"hotword":"直播"}]',
        hotword_id=None,
    )

    with caplog.at_level(logging.INFO):
        hotwords = service._load_hotwords(task)

    assert hotwords == "篮子 直播"
    assert "🔑 加载热词: 2 个" in caplog.messages
    assert "🔑 加载热词: 5 个" not in caplog.messages


@pytest.mark.asyncio
async def test_offline_task_service_processes_task_through_recognition_service(tmp_path):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")
    repository = _FakeRepository()
    runtime = _FakeOfflineRuntimeService()
    task_service = OfflineTaskService(
        task_repository=repository,
        temp_file_store=_FakeTempStore(audio_path),
        result_handler=_FakeResultHandler(repository),
        recognition_service=OfflineRecognitionService(runtime),
        config=_FakeConfig(),
    )

    ok = await task_service.process_task("task-1")

    assert ok is True
    assert repository.statuses == [("task-1", "processing")]
    assert repository.saved["full_text"] == "hello"
    assert runtime.requests[0].hotwords == ["热词"]
