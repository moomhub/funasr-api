import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.task_results import (
    OfflineTaskContext,
    build_offline_persistence_payload,
    extract_word_timestamps,
)
from src.core.results import RecognitionResult, Segment
from src.task_queue.hooks import ResultPersistenceHook


def test_build_offline_persistence_payload_keeps_summary_safe_and_extracts_timestamps():
    result = RecognitionResult(
        mode="offline",
        full_text="你好世界",
        processing_time=1.25,
        speaker_ids=["A", "B"],
        speaker_count=2,
        metadata={"request_id": "req-1", "raw_payload": {"text": "secret"}},
        segments=[
            Segment(text="你好", start=0, end=500, speaker="A", timestamp=[["你", 0, 200]]),
            Segment(text="世界", start=500, end=900, speaker="B", timestamp=[["世", 500, 700]]),
        ],
    )

    payload = build_offline_persistence_payload(
        OfflineTaskContext(task_id="task-1", filename="demo.wav", audio_path="demo.wav"),
        result,
    )

    assert payload.full_text == "你好世界"
    assert payload.processing_time == 1.25
    assert payload.word_timestamps == [["你", 0, 200], ["世", 500, 700]]
    assert payload.summary == {
        "task_id": "task-1",
        "text_length": 4,
        "segment_count": 2,
        "word_timestamp_count": 2,
        "speaker_ids": ["A", "B"],
        "speaker_count": 2,
        "metadata_keys": ["raw_payload", "request_id"],
    }
    assert "full_text" not in payload.summary
    assert "filename" not in payload.summary
    assert "raw_payload" not in payload.summary.values()


def test_extract_word_timestamps_ignores_missing_or_non_list_values():
    assert extract_word_timestamps(
        [
            {"timestamp": [["你", 0, 100]]},
            {"timestamp": None},
            {"timestamp": "not-list"},
            {},
        ]
    ) == [["你", 0, 100]]


@pytest.mark.asyncio
async def test_result_persistence_hook_uses_prepared_payload():
    class Repository:
        def __init__(self):
            self.saved = None

        def save_result(self, **kwargs):
            self.saved = kwargs
            return object()

    repository = Repository()
    hook = ResultPersistenceHook(repository)

    await hook.on_success(
        OfflineTaskContext(task_id="task-1", filename="demo.wav", audio_path="demo.wav"),
        RecognitionResult(
            mode="offline",
            full_text="你好",
            processing_time=0.5,
            segments=[Segment(text="你好", start=0, end=300, timestamp=[["你", 0, 100]])],
        ),
    )

    assert repository.saved == {
        "task_id": "task-1",
        "full_text": "你好",
        "segments": [
            {
                "text": "你好",
                "start": 0,
                "end": 300,
                "speaker": 0,
                "is_final": False,
                "confidence": 1.0,
                "timestamp": [["你", 0, 100]],
                "duration": 300,
            }
        ],
        "processing_time": 0.5,
        "word_timestamps": [["你", 0, 100]],
    }
