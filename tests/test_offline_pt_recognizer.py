import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.results import SpeakerResult, SpeakerSegment
from src.engine_runtime.engines.offline.base import OfflineRecognitionRequest
from src.engine_runtime.engines.offline.pt.speaker_merge import (
    merge_pt_sentence_info_with_speaker,
)
from src.engine_runtime.engines.offline.pt.recognizer import PTOfflineRecognizer


class _FakeOfflineModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate(self, audio_path, **kwargs):
        self.calls.append({"audio_path": audio_path, "kwargs": kwargs})
        return self.payload


class _FakeSpeakerRecognizer:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    async def recognize(self, request):
        self.requests.append(request)
        return self.payload


class _FakeManager:
    def __init__(self, offline_model, speaker_recognizer, processing_config=None):
        self.offline_model = offline_model
        self.speaker_recognizer = speaker_recognizer
        self.processing_config = processing_config

    def get_offline_model(self):
        return self.offline_model

    def get_spk_recognizer(self):
        return self.speaker_recognizer


class _ProcessingConfig:
    def __init__(self, offline_spk_verification_enabled=True):
        self.offline_spk_verification_enabled = offline_spk_verification_enabled


@pytest.mark.asyncio
async def test_pt_offline_recognizer_runs_standalone_spk_and_rebuilds_segments():
    offline_model = _FakeOfflineModel(
        [
            {
                "sentence_info": [
                    {
                        "text": "你好",
                        "start": 0,
                        "end": 1000,
                        "spk": 0,
                        "timestamp": [["你", 0, 400], ["好", 400, 1000]],
                    },
                    {
                        "text": "世界",
                        "start": 1000,
                        "end": 2000,
                        "spk": 0,
                        "timestamp": [["世", 1000, 1500], ["界", 1500, 2000]],
                    },
                ]
            }
        ]
    )
    speaker_recognizer = _FakeSpeakerRecognizer(
        {
            "segments": [
                {"start": 0, "end": 450, "speaker": "spk-0"},
                {"start": 450, "end": 2000, "speaker": "spk-1"},
            ]
        }
    )
    recognizer = PTOfflineRecognizer(_FakeManager(offline_model, speaker_recognizer))

    result = await recognizer.recognize(OfflineRecognitionRequest(audio_path="demo.wav"))

    assert offline_model.calls[0]["audio_path"] == "demo.wav"
    assert offline_model.calls[0]["kwargs"]["return_spk_res"] is False
    assert speaker_recognizer.requests[0].audio_path == "demo.wav"
    assert [segment.text for segment in result.segments] == ["你", "好世界"]
    assert [segment.speaker for segment in result.segments] == ["spk-0", "spk-1"]
    assert result.speaker_ids == ["spk-0", "spk-1"]
    assert result.full_text == "[说话人 spk-0] 你\n[说话人 spk-1] 好世界"
    assert result.metadata["speaker_result"]["speaker_ids"] == ["spk-0", "spk-1"]


@pytest.mark.asyncio
async def test_pt_offline_recognizer_skips_spk_and_returns_asr_text_when_no_timestamps():
    offline_model = _FakeOfflineModel(
        [
            {
                "text": "第一句第二句",
                "sentence_info": [
                    {"sentence": "第一句", "start": 0, "end": 900, "spk": 0, "timestamp": []},
                    {"sentence": "第二句", "start": 900, "end": 1800, "spk": 0, "timestamp": []},
                ]
            }
        ]
    )
    speaker_recognizer = _FakeSpeakerRecognizer(
        {
            "segments": [
                {"start": 0, "end": 800, "speaker": "A"},
                {"start": 800, "end": 1800, "speaker": "B"},
            ]
        }
    )
    recognizer = PTOfflineRecognizer(_FakeManager(offline_model, speaker_recognizer))

    result = await recognizer.recognize(OfflineRecognitionRequest(audio_path="demo.wav"))

    assert speaker_recognizer.requests == []
    assert [segment.text for segment in result.segments] == ["第一句", "第二句"]
    assert result.full_text == "第一句第二句"
    assert "speaker_result" not in result.metadata


@pytest.mark.asyncio
async def test_pt_offline_recognizer_fails_when_required_speaker_result_is_unusable():
    offline_model = _FakeOfflineModel(
        [
            {
                "sentence_info": [
                    {
                        "text": "原文",
                        "start": 0,
                        "end": 500,
                        "spk": 0,
                        "timestamp": [["原", 0, 250], ["文", 250, 500]],
                    }
                ]
            }
        ]
    )
    recognizer = PTOfflineRecognizer(
        _FakeManager(
            offline_model,
            _FakeSpeakerRecognizer(SpeakerResult(error="spk unavailable")),
        )
    )

    result = await recognizer.recognize(OfflineRecognitionRequest(audio_path="demo.wav"))

    assert result.error == "OFFLINE PT 推理失败: OFFLINE SPK 二次校验失败: spk unavailable"
    assert result.is_final is False


@pytest.mark.asyncio
async def test_pt_offline_recognizer_skips_standalone_spk_when_disabled_by_config():
    offline_model = _FakeOfflineModel(
        [
            {
                "text": "第一句第二句",
                "sentence_info": [
                    {"sentence": "第一句", "start": 0, "end": 900, "spk": 0, "timestamp": []},
                    {"sentence": "第二句", "start": 900, "end": 1800, "spk": 0, "timestamp": []},
                ]
            }
        ]
    )
    speaker_recognizer = _FakeSpeakerRecognizer(
        {
            "segments": [
                {"start": 0, "end": 800, "speaker": "A"},
                {"start": 800, "end": 1800, "speaker": "B"},
            ]
        }
    )
    recognizer = PTOfflineRecognizer(
        _FakeManager(
            offline_model,
            speaker_recognizer,
            _ProcessingConfig(offline_spk_verification_enabled=False),
        )
    )

    result = await recognizer.recognize(OfflineRecognitionRequest(audio_path="demo.wav"))

    assert speaker_recognizer.requests == []
    assert [segment.text for segment in result.segments] == ["第一句", "第二句"]
    assert result.full_text == "第一句第二句"
    assert "speaker_result" not in result.metadata


def test_pt_speaker_merge_splits_timestamp_tokens_at_speaker_boundary():
    sentence_info = [
        {
            "text": "你好世界",
            "start": 0,
            "end": 1600,
            "timestamp": [
                ["你", 0, 400],
                ["好", 400, 800],
                ["世", 800, 1200],
                ["界", 1200, 1600],
            ],
        }
    ]
    speaker_result = SpeakerResult(
        segments=[
            SpeakerSegment(speaker="A", start=0, end=800),
            SpeakerSegment(speaker="B", start=800, end=1600),
        ],
        speaker_ids=["A", "B"],
    )

    merged = merge_pt_sentence_info_with_speaker(sentence_info, speaker_result)

    assert [item["text"] for item in merged] == ["你好", "世界"]
    assert [item["spk"] for item in merged] == ["A", "B"]
    assert merged[0]["timestamp"] == [["你", 0, 400], ["好", 400, 800]]
    assert merged[1]["timestamp"] == [["世", 800, 1200], ["界", 1200, 1600]]


def test_pt_speaker_merge_uses_sentence_level_when_timestamps_missing():
    sentence_info = [
        {"text": "第一句", "start": 0, "end": 900},
        {"text": "第二句", "start": 900, "end": 1800},
    ]
    speaker_result = SpeakerResult(
        segments=[
            SpeakerSegment(speaker="A", start=0, end=800),
            SpeakerSegment(speaker="B", start=800, end=1800),
        ],
        speaker_ids=["A", "B"],
    )

    merged = merge_pt_sentence_info_with_speaker(sentence_info, speaker_result)

    assert [item["text"] for item in merged] == ["第一句", "第二句"]
    assert [item["spk"] for item in merged] == ["A", "B"]


def test_pt_speaker_merge_returns_sentence_copies_when_speaker_unavailable():
    sentence_info = [
        {"text": "原文", "start": 0, "end": 500, "spk": 7},
    ]

    merged = merge_pt_sentence_info_with_speaker(
        sentence_info,
        SpeakerResult(error="spk unavailable"),
    )

    assert merged == sentence_info
    assert merged[0] is not sentence_info[0]


def test_pt_speaker_merge_combines_timestamp_detail_for_same_speaker():
    sentence_info = [
        {
            "text": "你好",
            "start": 0,
            "end": 800,
            "timestamp": [["你", 0, 400], ["好", 400, 800]],
        }
    ]
    speaker_result = SpeakerResult(
        segments=[SpeakerSegment(speaker="A", start=0, end=800)],
        speaker_ids=["A"],
    )

    merged = merge_pt_sentence_info_with_speaker(sentence_info, speaker_result)

    assert merged == [
        {
            "text": "你好",
            "start": 0,
            "end": 800,
            "spk": "A",
            "timestamp": [["你", 0, 400], ["好", 400, 800]],
        }
    ]


def test_pt_speaker_merge_short_token_crossing_boundary_moves_to_next_speaker():
    sentence_info = [
        {
            "text": "机动，你喇个也摘了",
            "start": 13270,
            "end": 14330,
            "timestamp": [
                ["机", 13270, 13410],
                ["动", 13410, 13530],
                ["你", 13530, 13650],
                ["喇", 13650, 13830],
                ["个", 13830, 13950],
                ["也", 13950, 14050],
                ["摘", 14050, 14130],
                ["了", 14130, 14330],
            ],
        }
    ]
    speaker_result = SpeakerResult(
        segments=[
            SpeakerSegment(speaker=0, start=900, end=13600),
            SpeakerSegment(speaker=1, start=13600, end=15180),
        ],
        speaker_ids=[0, 1],
    )

    merged = merge_pt_sentence_info_with_speaker(sentence_info, speaker_result)

    assert [item["text"] for item in merged] == ["机动，", "你喇个也摘了"]
    assert [item["spk"] for item in merged] == [0, 1]
