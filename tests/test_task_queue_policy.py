from datetime import datetime, timezone

from src.task_queue.policy import build_batch_result, queue_priority_for


def test_queue_priority_for_vip_and_normal():
    assert queue_priority_for(True) == 0
    assert queue_priority_for(False) == 1


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
