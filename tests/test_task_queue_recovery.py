import logging
from types import SimpleNamespace

from src.task_queue.recovery import recover_and_enqueue_tasks


class _Repository:
    def __init__(self, pending=None, recovered=0, fail_pending=False):
        self.pending = list(pending or [])
        self.recovered = recovered
        self.fail_pending = fail_pending
        self.timeouts = []

    def recover_stale_processing(self, timeout_seconds):
        self.timeouts.append(timeout_seconds)
        return self.recovered

    def get_pending_tasks(self, limit=1000):
        if self.fail_pending:
            raise RuntimeError("pending unavailable")
        return self.pending[:limit]


def test_recover_and_enqueue_tasks_recovers_and_enqueues_pending_vip_tasks():
    calls = []
    repository = _Repository(
        pending=[
            SimpleNamespace(id="normal", vip=False),
            SimpleNamespace(id="vip", vip=True),
        ],
        recovered=2,
    )

    recover_and_enqueue_tasks(
        repository=repository,
        timeout_seconds=30,
        enqueue=lambda task_id, vip=False: calls.append((task_id, vip)),
        task_kind="OFFLINE",
        logger=logging.getLogger(__name__),
    )

    assert repository.timeouts == [30]
    assert calls == [("normal", False), ("vip", True)]


def test_recover_and_enqueue_tasks_can_skip_missing_optional_repository(caplog):
    with caplog.at_level(logging.WARNING):
        recover_and_enqueue_tasks(
            repository=None,
            timeout_seconds=30,
            enqueue=lambda *_args, **_kwargs: None,
            task_kind="SPK",
            logger=logging.getLogger(__name__),
            missing_repository_is_error=False,
        )

    assert caplog.text == ""


def test_recover_and_enqueue_tasks_logs_pending_failure_by_repository_strictness(caplog):
    logger = logging.getLogger(__name__)

    with caplog.at_level(logging.DEBUG):
        recover_and_enqueue_tasks(
            repository=_Repository(fail_pending=True),
            timeout_seconds=30,
            enqueue=lambda *_args, **_kwargs: None,
            task_kind="OFFLINE",
            logger=logger,
            missing_repository_is_error=True,
        )
        recover_and_enqueue_tasks(
            repository=_Repository(fail_pending=True),
            timeout_seconds=30,
            enqueue=lambda *_args, **_kwargs: None,
            task_kind="SPK",
            logger=logger,
            missing_repository_is_error=False,
        )

    assert "phase=pending task_kind=OFFLINE" in caplog.text
    assert "phase=pending task_kind=SPK" in caplog.text
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert all("pending unavailable" not in message for message in warning_messages)
