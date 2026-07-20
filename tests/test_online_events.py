from src.application.online_events import (
    backpressure_event,
    error_event,
    inactive_session_event,
    normalize_stream_command,
    ready_event,
    started_event,
    stopped_event,
    unknown_command_event,
)


def test_online_ready_event_includes_protocol_and_hotword_summary():
    assert ready_event(
        sample_rate=16000,
        decode_interval=0.48,
        queue_max_chunks=4,
        hotword_id=7,
        resolved_hotwords=[[80, "保险"], [60, "理赔"]],
    ) == {
        "event": "ready",
        "mode": "2pass",
        "sample_rate": 16000,
        "decode_interval": 0.48,
        "queue_max_chunks": 4,
        "hotword_id": 7,
        "hotword_count": 2,
    }


def test_online_command_and_basic_events():
    assert normalize_stream_command(" start ") == "START"
    assert started_event() == {"event": "started", "mode": "2pass"}
    assert stopped_event({"chunks": 3}) == {"event": "stopped", "metrics": {"chunks": 3}}
    assert error_event("failed") == {"event": "error", "error": "failed"}
    assert unknown_command_event("PING") == {"event": "error", "error": "未知命令: PING"}
    assert inactive_session_event() == {"event": "error", "error": "会话未开始，请先发送 START"}


def test_online_backpressure_event_contains_queue_metrics():
    assert backpressure_event(
        queue_size=4,
        queue_max_chunks=8,
        metrics={"dropped": 2},
    ) == {
        "event": "backpressure",
        "mode": "2pass",
        "queue_size": 4,
        "queue_max_chunks": 8,
        "metrics": {"dropped": 2},
    }
