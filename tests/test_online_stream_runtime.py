import asyncio
import logging

import pytest

from src.application.online_stream_runtime import OnlineStreamRuntime


class _WebSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _Session:
    def __init__(self):
        self.is_active = False
        self.reset_count = 0
        self.finish_count = 0
        self.added = []
        self.queue_depths = []
        self.dropped = 0
        self.backpressure = 0
        self.partial_result = {"text": "partial", "sentences": []}
        self.locked_result = {"text": "", "sentences": []}
        self.final_result = {"text": "final", "sentences": [{"text": "final"}]}
        self.decode_enabled = True

    def reset(self):
        self.reset_count += 1

    async def finish(self):
        self.finish_count += 1

    async def add_audio(self, data):
        self.added.append(data)
        return False

    def should_decode(self, now):
        return self.decode_enabled

    async def decode_partial(self):
        return self.partial_result

    def build_offline_event(self, is_final):
        return self.final_result if is_final else self.locked_result

    def note_queue_depth(self, depth):
        self.queue_depths.append(depth)

    def note_dropped_chunk(self):
        self.dropped += 1

    def note_backpressure(self):
        self.backpressure += 1

    def get_metrics(self):
        return {
            "added": len(self.added),
            "dropped": self.dropped,
            "backpressure": self.backpressure,
        }


@pytest.mark.asyncio
async def test_online_stream_runtime_runs_audio_and_finalizes_session():
    websocket = _WebSocket()
    session = _Session()
    runtime = OnlineStreamRuntime(
        websocket,
        session,
        queue_max_chunks=2,
        sample_rate=16000,
        clock=lambda: 10.0,
    )
    runtime.start_worker()

    try:
        await runtime.start_session()
        await runtime.enqueue_audio(b"pcm")
        await runtime.audio_queue.join()
        await runtime.stop_session()
    finally:
        await runtime.close()

    assert session.reset_count == 1
    assert session.finish_count == 1
    assert session.added == [b"pcm"]
    assert websocket.sent == [
        {"event": "started", "mode": "2pass"},
        {"text": "partial", "sentences": []},
        {"text": "final", "sentences": [{"text": "final"}]},
        {
            "event": "stopped",
            "metrics": {"added": 1, "dropped": 0, "backpressure": 0},
        },
    ]


@pytest.mark.asyncio
async def test_online_stream_runtime_reports_inactive_and_backpressure():
    websocket = _WebSocket()
    session = _Session()
    runtime = OnlineStreamRuntime(
        websocket,
        session,
        queue_max_chunks=1,
        sample_rate=16000,
        clock=lambda: 1.0,
    )

    await runtime.enqueue_audio(b"inactive")
    session.is_active = True
    await runtime.enqueue_audio(b"first")
    await runtime.enqueue_audio(b"overflow")

    assert websocket.sent == [
        {"event": "error", "error": "会话未开始，请先发送 START"},
        {
            "event": "backpressure",
            "mode": "2pass",
            "queue_size": 1,
            "queue_max_chunks": 1,
            "metrics": {"added": 0, "dropped": 1, "backpressure": 1},
        },
    ]
    assert session.queue_depths == [1]
    runtime._drain_audio_queue()


class _BlockingSession(_Session):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.locked_result = {"text": "stale", "sentences": [{"text": "stale"}]}

    async def add_audio(self, data):
        self.added.append(data)
        self.entered.set()
        await self.release.wait()
        return True


@pytest.mark.asyncio
async def test_online_stream_runtime_discards_inflight_audio_after_restart():
    websocket = _WebSocket()
    session = _BlockingSession()
    runtime = OnlineStreamRuntime(
        websocket,
        session,
        queue_max_chunks=2,
        sample_rate=16000,
    )
    runtime.start_worker()

    try:
        await runtime.start_session()
        await runtime.enqueue_audio(b"old-session")
        await asyncio.wait_for(session.entered.wait(), timeout=1)

        restart = asyncio.create_task(runtime.start_session())
        while runtime._generation < 2:
            await asyncio.sleep(0)
        session.release.set()
        await asyncio.wait_for(restart, timeout=1)
        await asyncio.wait_for(runtime.audio_queue.join(), timeout=1)
    finally:
        await runtime.close()

    assert session.reset_count == 2
    assert {"text": "stale", "sentences": [{"text": "stale"}]} not in websocket.sent
    assert websocket.sent == [
        {"event": "started", "mode": "2pass"},
        {"event": "started", "mode": "2pass"},
    ]


@pytest.mark.asyncio
async def test_online_stream_runtime_debug_logs_output_payload(caplog):
    websocket = _WebSocket()
    runtime = OnlineStreamRuntime(
        websocket,
        _Session(),
        queue_max_chunks=1,
        sample_rate=16000,
    )
    caplog.set_level(logging.DEBUG, logger="src.application.online_stream_runtime")

    await runtime.send_json({"text": "detailed debug text", "timestamp": [[0, 10]]})

    debug_messages = [record.getMessage() for record in caplog.records]
    assert websocket.sent == [
        {"text": "detailed debug text", "timestamp": [[0, 10]]}
    ]
    assert any("detailed debug text" in message for message in debug_messages)
    assert any('"timestamp": [[0, 10]]' in message for message in debug_messages)
