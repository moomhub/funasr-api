import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.results import SpeakerSegment
from src.engine_runtime.engines.offline.speaker_utils import (
    merge_timed_units_by_speaker,
)


def test_merge_timed_units_by_speaker_groups_text_and_preserves_timestamps():
    merged = merge_timed_units_by_speaker(
        [
            {"text": "你", "start": 0, "end": 300, "timestamp": [["你", 0, 300]]},
            {"text": "好", "start": 300, "end": 600, "timestamp": [["好", 300, 600]]},
            {"text": "世", "start": 600, "end": 900, "timestamp": [["世", 600, 900]]},
        ],
        [
            SpeakerSegment(speaker="A", start=0, end=600),
            SpeakerSegment(speaker="B", start=600, end=900),
        ],
    )

    assert merged == [
        {
            "text": "你好",
            "start": 0,
            "end": 600,
            "spk": "A",
            "timestamp": [["你", 0, 300], ["好", 300, 600]],
        },
        {
            "text": "世",
            "start": 600,
            "end": 900,
            "spk": "B",
            "timestamp": [["世", 600, 900]],
        },
    ]


def test_merge_timed_units_by_speaker_uses_short_boundary_correction():
    merged = merge_timed_units_by_speaker(
        [
            {"text": "机", "start": 13270, "end": 13410},
            {"text": "动", "start": 13410, "end": 13530},
            {"text": "你", "start": 13530, "end": 13650},
        ],
        [
            SpeakerSegment(speaker=0, start=900, end=13600),
            SpeakerSegment(speaker=1, start=13600, end=15180),
        ],
    )

    assert merged == [
        {"text": "机动", "start": 13270, "end": 13530, "spk": 0},
        {"text": "你", "start": 13530, "end": 13650, "spk": 1},
    ]
