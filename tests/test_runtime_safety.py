import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from src.api import offline as offline_api
from src.api import online as online_api
from src.application.tasks import TaskSubmissionService
from src.core.adapters import MemoryTaskRepository
from src.core.debug_logging import json_for_log, mask_url
from src.engine_runtime.engines.offline.onnx.recognizer import OfflineONNXModelBundle
from src.engine_runtime.engines.online.onnx.adapters import ONNXVADWrapper
import src.task_queue.queue as queue_module


class _StatefulVadModel:
    def __call__(self, _audio, param_dict=None):
        cache = param_dict["in_cache"]
        cache.append(len(cache) + 1)
        return [[100, -1]]


def test_onnx_vad_state_is_isolated_per_websocket_session():
    runtime = ONNXVADWrapper.__new__(ONNXVADWrapper)
    runtime.model = _StatefulVadModel()
    runtime._model_lock = threading.Lock()

    first = runtime.create_session()
    second = runtime.create_session()
    first.feed(np.ones(1600, dtype=np.float32))

    assert first.cache == [1]
    assert first.current_speech_start == 100
    assert second.cache == []
    assert second.current_speech_start is None

    second.reset()
    assert first.cache == [1]
    assert first.current_speech_start == 100


class _TrackedAsrModel:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, _audio, **_kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self._lock:
            self.active -= 1
        return [{"text": "测", "timestamp": [[0, 10]]}]


def test_offline_onnx_asr_model_pool_is_shared_across_requests():
    model = _TrackedAsrModel()
    bundle = OfflineONNXModelBundle.__new__(OfflineONNXModelBundle)
    bundle.sample_rate = 1000
    bundle.vad_padding_ms = 0
    bundle.asr_models = [model]
    bundle._asr_model_pool = Queue()
    bundle._asr_model_pool.put(model)
    audio = np.ones(100, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(bundle._run_asr_segments, audio, [[0, 50]], None)
            for _ in range(2)
        ]
        for future in futures:
            future.result()

    assert model.max_active == 1


@pytest.mark.asyncio
async def test_disabled_offline_mode_rejects_before_reading_upload(monkeypatch):
    services = SimpleNamespace(is_engine_enabled=lambda _mode: False)

    response = await offline_api.upload_offline_task(file=None, services=services)

    assert response.status_code == 503
    assert json.loads(response.body) == {"error": "OFFLINE 模式未启用"}


@pytest.mark.asyncio
async def test_invalid_http_hotwords_return_400_before_saving_upload():
    class TempStore:
        def __init__(self):
            self.save_calls = 0

        async def save_upload(self, *_args):
            self.save_calls += 1
            raise AssertionError("invalid hotwords must be rejected before file IO")

        def cleanup(self, _task_id):
            return None

    temp_store = TempStore()
    submission = TaskSubmissionService(
        config=SimpleNamespace(get=lambda _key, default=None: default),
        temp_file_store=temp_store,
        task_repository=object(),
        speaker_task_repository=object(),
        scheduler=SimpleNamespace(
            can_accept=lambda _mode: True,
            enqueue_offline=lambda *_args, **_kwargs: None,
            enqueue_spk=lambda *_args, **_kwargs: None,
        ),
    )
    services = SimpleNamespace(
        is_engine_enabled=lambda mode: mode == "offline",
        task_submission_service=submission,
    )

    response = await offline_api.upload_offline_task(
        file=SimpleNamespace(filename="demo.wav"),
        email=None,
        hotwords="旧格式:80",
        hotword_id=None,
        vip=False,
        services=services,
    )

    assert response.status_code == 400
    assert "JSON 数组" in json.loads(response.body)["error"]
    assert temp_store.save_calls == 0


@pytest.mark.asyncio
async def test_invalid_websocket_hotwords_send_error_and_close_with_1008():
    class WebSocket:
        def __init__(self):
            self.accepted = False
            self.messages = []
            self.close_code = None

        async def accept(self):
            self.accepted = True

        async def send_json(self, payload):
            self.messages.append(payload)

        async def close(self, code):
            self.close_code = code

    processing = SimpleNamespace(online_queue_max_chunks=2)
    config = SimpleNamespace(
        get=lambda _key, default=None: default,
        get_processing_config=lambda: processing,
    )
    services = SimpleNamespace(
        config=config,
        is_engine_enabled=lambda mode: mode == "online",
        hotword_repository=None,
    )
    websocket = WebSocket()

    await online_api.websocket_stream(
        websocket,
        hotwords="旧格式:80",
        hotword_id=None,
        sample_rate=16000,
        services=services,
    )

    assert websocket.accepted is True
    assert websocket.messages[0]["event"] == "error"
    assert "JSON 数组" in websocket.messages[0]["error"]
    assert websocket.close_code == 1008


class _ProcessingConfig:
    offline_async_enabled = True
    offline_async_allow_immediate = True
    max_concurrent_tasks = 1
    timeout_seconds = 60


class _Config:
    def get_processing_config(self):
        return _ProcessingConfig()


class _Container:
    def __init__(self, repository):
        self.task_repository = repository
        self.temp_file_store = object()
        self.audio_backup_store = object()


class _NoopBatchHandler:
    async def handle_complete(self, _context):
        return None


class _RetryingTaskService:
    def __init__(self, repository):
        self.repository = repository
        self.attempts = 0

    async def process_task(self, task_id):
        self.attempts += 1
        if self.attempts == 1:
            self.repository.record_error(task_id, "retry", retry=True)
            return False
        self.repository.save_result(task_id, "ok", [], 0.01)
        return True


@pytest.mark.asyncio
async def test_failed_pending_task_is_retried_in_current_process():
    repository = MemoryTaskRepository()
    repository.create_task("retry-task", "retry.wav", 1)
    service = _RetryingTaskService(repository)
    task_queue = queue_module.OfflineTaskQueue(
        config=_Config(),
        task_repository=repository,
        batch_result_handler=_NoopBatchHandler(),
        task_service=service,
        speaker_task_repository=object(),
    )
    task_queue.retry_delay_seconds = 0.01
    task_queue.start()
    try:
        for _ in range(100):
            if repository.get_task("retry-task").status == "completed":
                break
            await asyncio.sleep(0.01)
        assert repository.get_task("retry-task").status == "completed"
        assert service.attempts == 2
    finally:
        await task_queue.stop()


def test_memory_repository_recovers_stale_processing_tasks():
    repository = MemoryTaskRepository()
    task = repository.create_task("stale", "stale.wav", 1)
    task.status = "processing"
    task.started_at = datetime.now(timezone.utc) - timedelta(seconds=120)

    assert repository.recover_stale_processing(60) == 1
    assert task.status == "pending"


def test_debug_logging_masks_sensitive_values():
    masked_url = mask_url("mysql+pymysql://user:secret@localhost:3306/demo")
    payload = json_for_log(
        {
            "password": "secret",
            "nested": {"access_key": "abc123", "text": "visible"},
            "url": masked_url,
        }
    )

    assert "secret" not in payload
    assert "abc123" not in payload
    assert "***" in payload
    assert "visible" in payload
