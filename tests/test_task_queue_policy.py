from datetime import datetime, timezone

from src.task_queue.policy import build_batch_result, is_task_retriable, queue_priority_for


class _Task:
    def __init__(self, status="pending", retry_count=0, max_retries=3):
        self.status = status
        self.retry_count = retry_count
        self.max_retries = max_retries


def test_queue_priority_for_vip_and_normal():
    assert queue_priority_for(True) == 0
    assert queue_priority_for(False) == 1


def test_is_task_retriable_requires_pending_and_remaining_budget():
    assert is_task_retriable(_Task()) is True
    assert is_task_retriable(_Task(status="processing")) is False
    assert is_task_retriable(_Task(retry_count=3)) is False
    assert is_task_retriable(None) is False


def test_build_batch_result_counts_completed_and_failed_tasks():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)

    result = build_batch_result(["a", "b", "c"], [True, False, True], start, end)

    assert result == {
        "total": 3,
        "completed": 2,
        "failed": 1,
        "processing_time": 5.0,
    }
