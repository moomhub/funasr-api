import asyncio
from types import SimpleNamespace

import pytest

from src.core.adapters import MemoryTaskRepository
from src.task_queue.execution import TaskExecutionRegistry
from src.task_queue.queue import OfflineTaskQueue


def test_task_execution_registry_blocks_queued_and_inflight_duplicates():
    registry = TaskExecutionRegistry()
    key = ("offline", "task-1")

    assert registry.try_enqueue(key) is True
    assert registry.try_enqueue(key) is False
    registry.begin_queued(key)
    assert registry.try_enqueue(key) is False
    assert registry.try_begin_external(key) is False
    registry.finish(key)
    assert registry.try_begin_external(key) is True
    registry.finish(key)
    assert registry.try_enqueue(key) is True


class _ProcessingConfig:
    offline_async_enabled = True
    offline_async_allow_immediate = True
    max_concurrent_tasks = 1
    timeout_seconds = 60


class _Config:
    def get_processing_config(self):
        return _ProcessingConfig()


class _BatchHandler:
    async def handle_complete(self, _context):
        return None


class _BlockingTaskService:
    def __init__(self, repository):
        self.repository = repository
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.attempts = 0

    async def process_task(self, task_id):
        self.attempts += 1
        self.repository.update_status(task_id, "processing")
        self.entered.set()
        await self.release.wait()
        self.repository.save_result(task_id, "ok", [], 0.01)
        return True


@pytest.mark.asyncio
async def test_offline_queue_rejects_duplicate_enqueue_while_task_is_inflight():
    repository = MemoryTaskRepository()
    repository.create_task("task-1", "demo.wav", 1)
    service = _BlockingTaskService(repository)
    task_queue = OfflineTaskQueue(
        config=_Config(),
        task_repository=repository,
        batch_result_handler=_BatchHandler(),
        task_service=service,
        speaker_task_repository=None,
    )
    task_queue.start()

    try:
        await asyncio.wait_for(service.entered.wait(), timeout=1)
        task_queue.enqueue_offline("task-1")
        service.release.set()
        for _ in range(50):
            if repository.get_task("task-1").status == "completed":
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.02)
    finally:
        await task_queue.stop()

    assert repository.get_task("task-1").status == "completed"
    assert service.attempts == 1
