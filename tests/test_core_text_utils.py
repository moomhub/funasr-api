import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.text import build_full_text_with_speaker, extract_segments_from_sentence_info


def test_extract_segments_from_sentence_info_supports_speaker_normalizer():
    segments = extract_segments_from_sentence_info(
        [
            {"text": "你好", "start": 0, "end": 300, "spk": "1", "timestamp": [["你", 0, 120]]},
            {"text": "世界", "start": 300, "end": 600, "spk": "speaker-2"},
        ],
        speaker_normalizer=lambda value: int(value) if str(value).isdigit() else value,
    )

    assert segments[0].speaker == 1
    assert segments[0].timestamp == [["你", 0, 120]]
    assert segments[1].speaker == "speaker-2"
    assert build_full_text_with_speaker(segments) == "[说话人 1] 你好\n[说话人 speaker-2] 世界"
