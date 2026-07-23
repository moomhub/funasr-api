import asyncio
import logging
from types import SimpleNamespace

import pytest

from src.task_queue.batch import ImmediateBatchProcessor
from src.task_queue.execution import TaskExecutionRegistry
from src.task_queue.queue import OfflineTaskQueue


class _ProcessingConfig:
    offline_async_enabled = True
    offline_async_allow_immediate = True
    max_concurrent_tasks = 1
    timeout_seconds = 60


class _Config:
    def get_processing_config(self):
        return _ProcessingConfig()


class _Repository:
    def recover_stale_processing(self, _timeout):
        return 0

    def get_pending_tasks(self, limit=1000):
        return []

    def get_task(self, task_id):
        if task_id == "bad":
            raise RuntimeError("private database details")
        return SimpleNamespace(
            id=task_id,
            status="pending",
            retry_count=0,
            max_retries=3,
            vip=False,
        )


class _TaskService:
    def __init__(self):
        self.bad_processed = asyncio.Event()
        self.good_processed = asyncio.Event()
        self.calls = []

    async def process_task(self, task_id):
        self.calls.append(task_id)
        if task_id == "bad":
            self.bad_processed.set()
            return False
        self.good_processed.set()
        return True


class _BatchHandler:
    async def handle_complete(self, _context):
        return None


@pytest.mark.asyncio
async def test_failed_queue_task_does_not_retry_and_worker_continues(caplog):
    service = _TaskService()
    queue = OfflineTaskQueue(
        config=_Config(),
        task_repository=_Repository(),
        batch_result_handler=_BatchHandler(),
        task_service=service,
        speaker_task_repository=None,
    )

    with caplog.at_level(logging.DEBUG, logger="src.task_queue.queue"):
        queue.start()
        try:
            queue.enqueue_offline("bad")
            await asyncio.wait_for(service.bad_processed.wait(), timeout=1)
            queue.enqueue_offline("good")
            await asyncio.wait_for(service.good_processed.wait(), timeout=1)
            await asyncio.sleep(0.02)
        finally:
            await queue.stop()

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert service.calls == ["bad", "good"]
    assert not any("Queue retry lookup failed" in message for message in warning_messages)
    assert all("private database details" not in message for message in warning_messages)


@pytest.mark.asyncio
async def test_immediate_batch_processor_honors_disabled_configuration():
    processor = ImmediateBatchProcessor(
        enabled=False,
        task_repository=object(),
        task_service=object(),
        result_handler=object(),
        registry=TaskExecutionRegistry(),
    )

    with pytest.raises(RuntimeError, match="immediate batch processing is disabled"):
        await processor.process(["task-1"])


@pytest.mark.asyncio
async def test_immediate_batch_continues_after_one_task_raises(caplog):
    class Repository:
        def get_task(self, task_id):
            return SimpleNamespace(id=task_id)

    class Service:
        def __init__(self):
            self.calls = []

        async def process_task(self, task_id):
            self.calls.append(task_id)
            if task_id == "bad":
                raise RuntimeError("private recognition details")
            return True

    class Handler:
        def __init__(self):
            self.context = None

        async def handle_complete(self, context):
            self.context = context

    service = Service()
    handler = Handler()
    processor = ImmediateBatchProcessor(
        enabled=True,
        task_repository=Repository(),
        task_service=service,
        result_handler=handler,
        registry=TaskExecutionRegistry(),
    )

    with caplog.at_level(logging.DEBUG, logger="src.task_queue.batch"):
        result = await processor.process(["bad", "good"])

    error_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
    ]
    assert service.calls == ["bad", "good"]
    assert result["completed"] == 1
    assert result["failed"] == 1
    assert handler.context.completed_tasks == 1
    assert handler.context.failed_tasks == 1
    assert all("private recognition details" not in message for message in error_messages)


@pytest.mark.asyncio
async def test_immediate_batch_isolates_repository_lookup_failure():
    class Repository:
        def get_task(self, task_id):
            if task_id == "lookup-failed":
                raise RuntimeError("database unavailable")
            return SimpleNamespace(id=task_id)

    class Service:
        def __init__(self):
            self.calls = []

        async def process_task(self, task_id):
            self.calls.append(task_id)
            return True

    class Handler:
        async def handle_complete(self, context):
            self.context = context

    service = Service()
    handler = Handler()
    processor = ImmediateBatchProcessor(
        enabled=True,
        task_repository=Repository(),
        task_service=service,
        result_handler=handler,
        registry=TaskExecutionRegistry(),
    )

    result = await processor.process(["lookup-failed", "good"])

    assert service.calls == ["good"]
    assert result["completed"] == 1
    assert result["failed"] == 1
    assert handler.context.failed_tasks == 1
