import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config.errors import ModelLoadError
from src.core.results import SpeakerResult, SpeakerSegment
from src.engine_runtime.engines.offline.base import OfflineRecognitionRequest
from src.engine_runtime.engines.offline.catalog import DEFAULT_MODELS
from src.engine_runtime.engines.offline.onnx.helpers import (
    asr_segment_decode,
    combine_asr_segment_decodes,
    extract_timestamps,
    extract_vad_segments,
    slice_audio_by_ms,
)
from src.engine_runtime.engines.offline.onnx.loader import load_offline_onnx_models
from src.engine_runtime.engines.offline.onnx.recognizer import OfflineONNXModelBundle, OfflineONNXRecognizer
from src.engine_runtime.engines.offline.onnx.speaker_merge import merge_onnx_speaker_result


class _FakeBundle:
    def __init__(self):
        self.generate_calls = []
        self.merge_calls = []

    def generate(self, audio_path, hotwords=None, **kwargs):
        self.generate_calls.append(
            {
                "audio_path": audio_path,
                "hotwords": hotwords,
                "kwargs": kwargs,
            }
        )
        return {
            "text": "你好世界",
            "raw_text": "你好世界",
            "timestamps": [["你", 0, 200], ["好", 200, 400], ["世", 400, 600], ["界", 600, 800]],
            "asr_segments": [{"text": "你好世界", "start": 0, "end": 800, "timestamp": []}],
        }

    def merge_speaker_result(self, payload, speaker):
        self.merge_calls.append({"payload": payload, "speaker": speaker})
        return {
            **payload,
            "sentence_info": [{"text": "你好世界", "start": 0, "end": 800, "spk": "spk-0"}],
            "speaker_result": speaker.to_dict(),
        }


class _FakeSpeakerRecognizer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def recognize(self, request):
        self.calls.append(request)
        return self.result


class _FakeManager:
    def __init__(self, speaker_result=None):
        speaker_result = speaker_result or SpeakerResult()
        self._speaker = _FakeSpeakerRecognizer(speaker_result)

    def get_spk_recognizer(self):
        return self._speaker


def test_offline_onnx_loader_loads_vad_punc_and_asr_workers(monkeypatch):
    created = []

    def factory(label):
        def build(**kwargs):
            created.append((label, kwargs))
            return {"label": label, "kwargs": kwargs}

        return build

    fake_module = types.SimpleNamespace(
        Fsmn_vad=factory("vad"),
        CT_Transformer=factory("punc"),
        Paraformer=factory("asr"),
    )
    monkeypatch.setitem(sys.modules, "funasr_onnx", fake_module)

    loaded = load_offline_onnx_models(
        asr_model_dir="asr-dir",
        vad_model_dir="vad-dir",
        punc_model_dir="punc-dir",
        quantize=True,
        num_threads=2,
        device_id=0,
        asr_workers=2,
        load_workers=3,
    )

    assert loaded.vad_model["label"] == "vad"
    assert loaded.punc_model["label"] == "punc"
    assert [model["label"] for model in loaded.asr_models] == ["asr", "asr"]
    assert ("vad", {"model_dir": "vad-dir", "quantize": True, "intra_op_num_threads": 2, "device_id": 0}) in created
    assert ("punc", {"model_dir": "punc-dir", "quantize": True, "intra_op_num_threads": 2, "device_id": 0}) in created


def test_offline_onnx_loader_wraps_model_load_failures(monkeypatch):
    def failing_asr(**_kwargs):
        raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(
        Fsmn_vad=lambda **_kwargs: object(),
        CT_Transformer=lambda **_kwargs: object(),
        Paraformer=failing_asr,
    )
    monkeypatch.setitem(sys.modules, "funasr_onnx", fake_module)

    with pytest.raises(ModelLoadError, match="OFFLINE ONNX 模型加载失败"):
        load_offline_onnx_models(
            asr_model_dir="asr-dir",
            vad_model_dir="vad-dir",
            punc_model_dir="punc-dir",
            quantize=False,
            num_threads=1,
            device_id=-1,
            asr_workers=1,
            load_workers=2,
        )


def test_offline_onnx_catalog_uses_seaco_model():
    assert DEFAULT_MODELS["onnx"]["asr"] == (
        "marxyz/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
    )


def test_offline_onnx_loader_uses_seaco_and_falls_back_to_non_quantized(monkeypatch, tmp_path):
    created = []
    asr_model_dir = tmp_path / "seaco"
    asr_model_dir.mkdir()
    (asr_model_dir / "config.yaml").write_text("model: SeacoParaformer\n", encoding="utf-8")
    (asr_model_dir / "model.onnx").touch()
    (asr_model_dir / "model_eb.onnx").touch()

    def factory(label):
        def build(**kwargs):
            created.append((label, kwargs))
            return {"label": label, "kwargs": kwargs}

        return build

    fake_module = types.SimpleNamespace(
        Fsmn_vad=factory("vad"),
        CT_Transformer=factory("punc"),
        Paraformer=factory("plain-asr"),
        SeacoParaformer=factory("seaco-asr"),
    )
    monkeypatch.setitem(sys.modules, "funasr_onnx", fake_module)

    loaded = load_offline_onnx_models(
        asr_model_dir=str(asr_model_dir),
        vad_model_dir="vad-dir",
        punc_model_dir="punc-dir",
        quantize=True,
        num_threads=2,
        device_id=0,
        asr_workers=2,
        load_workers=3,
    )

    assert loaded.asr_hotword_mode == "seaco"
    assert [model["label"] for model in loaded.asr_models] == ["seaco-asr", "seaco-asr"]
    seaco_calls = [kwargs for label, kwargs in created if label == "seaco-asr"]
    assert len(seaco_calls) == 2
    assert all(call["quantize"] is False for call in seaco_calls)


def test_offline_onnx_loader_rejects_incomplete_seaco_model(monkeypatch, tmp_path):
    asr_model_dir = tmp_path / "seaco"
    asr_model_dir.mkdir()
    (asr_model_dir / "config.yaml").write_text("model: SeacoParaformer\n", encoding="utf-8")
    (asr_model_dir / "model.onnx").touch()

    fake_module = types.SimpleNamespace(
        Fsmn_vad=lambda **_kwargs: object(),
        CT_Transformer=lambda **_kwargs: object(),
        Paraformer=lambda **_kwargs: object(),
        SeacoParaformer=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "funasr_onnx", fake_module)

    with pytest.raises(ModelLoadError, match="model_eb.onnx"):
        load_offline_onnx_models(
            asr_model_dir=str(asr_model_dir),
            vad_model_dir="vad-dir",
            punc_model_dir="punc-dir",
            quantize=True,
            num_threads=1,
            device_id=-1,
            asr_workers=1,
            load_workers=2,
        )


def test_offline_onnx_seaco_converts_hotword_list_to_space_separated_string():
    calls = []

    def model(audio, **kwargs):
        calls.append((audio, kwargs))
        return [{"preds": "测试"}]

    bundle = OfflineONNXModelBundle.__new__(OfflineONNXModelBundle)
    bundle.asr_hotword_mode = "seaco"

    result = bundle._invoke_asr_model(model, "audio", ["篮子", "直播", "蛇哥"])

    assert result == [{"preds": "测试"}]
    assert calls == [("audio", {"hotwords": "篮子 直播 蛇哥"})]


def test_offline_onnx_seaco_preserves_model_ready_hotword_string():
    calls = []

    def model(audio, **kwargs):
        calls.append((audio, kwargs))
        return [{"preds": "测试"}]

    bundle = OfflineONNXModelBundle.__new__(OfflineONNXModelBundle)
    bundle.asr_hotword_mode = "seaco"

    bundle._invoke_asr_model(model, "audio", "篮子 直播 蛇哥")

    assert calls == [("audio", {"hotwords": "篮子 直播 蛇哥"})]


def test_offline_onnx_seaco_extracts_words_from_weighted_hotwords():
    calls = []

    def model(audio, **kwargs):
        calls.append((audio, kwargs))
        return [{"preds": "测试"}]

    bundle = OfflineONNXModelBundle.__new__(OfflineONNXModelBundle)
    bundle.asr_hotword_mode = "seaco"

    bundle._invoke_asr_model(model, "audio", [[100, "篮子"], [80, "直播"]])

    assert calls == [("audio", {"hotwords": "篮子 直播"})]


def test_offline_onnx_seaco_passes_empty_hotword_string_when_no_hotwords():
    calls = []

    def model(audio, **kwargs):
        calls.append((audio, kwargs))
        return [{"preds": "测试"}]

    bundle = OfflineONNXModelBundle.__new__(OfflineONNXModelBundle)
    bundle.asr_hotword_mode = "seaco"

    bundle._invoke_asr_model(model, "audio", None)

    assert calls == [("audio", {"hotwords": ""})]


@pytest.mark.asyncio
async def test_offline_onnx_recognizer_uses_standalone_speaker_service_for_merge(monkeypatch):
    speaker_result = SpeakerResult(
        segments=[SpeakerSegment(speaker="spk-0", start=0, end=800)],
        speaker_ids=["spk-0"],
        speaker_count=1,
    )
    manager = _FakeManager(speaker_result)
    recognizer = OfflineONNXRecognizer(manager)
    bundle = _FakeBundle()

    payload = await recognizer.run_inference(
        bundle,
        OfflineRecognitionRequest(
            audio_path="demo.wav",
            hotwords=["热词"],
            generate_kwargs={"sample_rate": 16000},
        ),
    )

    assert bundle.generate_calls[0]["audio_path"] == "demo.wav"
    assert bundle.generate_calls[0]["hotwords"] == ["热词"]
    assert manager._speaker.calls[0].audio_path == "demo.wav"
    assert manager._speaker.calls[0].generate_kwargs == {"sample_rate": 16000}
    assert payload["speaker_result"]["speaker_ids"] == ["spk-0"]
    assert payload["sentence_info"][0]["spk"] == "spk-0"


@pytest.mark.asyncio
async def test_offline_onnx_recognizer_fails_when_required_speaker_service_fails(monkeypatch):
    class _BrokenRecognizer:
        async def recognize(self, request):
            raise RuntimeError("speaker service failed")

    manager = _FakeManager(SpeakerResult())
    manager.get_spk_recognizer = lambda: _BrokenRecognizer()
    recognizer = OfflineONNXRecognizer(manager)

    with pytest.raises(
        RuntimeError,
        match="OFFLINE SPK 二次校验失败: speaker service failed",
    ):
        await recognizer._recognize_speaker("demo.wav", {})


@pytest.mark.asyncio
async def test_offline_onnx_recognizer_fails_when_required_speaker_result_is_empty():
    recognizer = OfflineONNXRecognizer(_FakeManager(SpeakerResult()))

    with pytest.raises(
        RuntimeError,
        match="OFFLINE SPK 二次校验失败: 未返回有效说话人分段",
    ):
        await recognizer._recognize_speaker("demo.wav", {})


def test_offline_onnx_parse_result_builds_speaker_full_text():
    recognizer = OfflineONNXRecognizer(_FakeManager())

    result = recognizer.parse_result(
        {
            "text": "你好世界",
            "sentence_info": [
                {"text": "你好", "start": 0, "end": 400, "spk": "spk-0"},
                {"text": "世界", "start": 400, "end": 800, "spk": "spk-1"},
            ],
            "speaker_result": {
                "speaker_ids": ["spk-0", "spk-1"],
            },
        }
    )

    assert [segment.speaker for segment in result.segments] == ["spk-0", "spk-1"]
    assert result.full_text == "[说话人 spk-0] 你好\n[说话人 spk-1] 世界"
    assert result.metadata["speaker_result"]["speaker_ids"] == ["spk-0", "spk-1"]


def test_offline_onnx_fills_empty_timestamp_tokens_from_segment_text():
    timestamps = [["", 100, 200], ["", 200, 300], ["", 300, 500]]

    filled = OfflineONNXModelBundle._fill_timestamp_tokens("你 好 世界", timestamps)

    assert filled == [["你", 100, 200], ["好", 200, 300], ["世界", 300, 500]]


def test_offline_onnx_helpers_extract_nested_vad_and_timestamps():
    vad_payload = {
        "value": [
            {"segments": [[300, 600], [0, 200]]},
            {"sentence_info": [[700, 900]]},
            [-1, 100],
        ]
    }
    timestamp_payload = (
        "ignored",
        [
            {"timestamp": [["你", 10, 80]]},
            {"raw_result": [{"timestamp": [[80, 150]]}]},
        ],
    )

    assert extract_vad_segments(vad_payload) == [[0, 200], [300, 600], [700, 900]]
    assert extract_timestamps(timestamp_payload) == [["你", 10, 80], ["", 80, 150]]


def test_offline_onnx_helpers_slice_audio_and_recombine_segment_decodes():
    audio = list(range(100))
    sliced, offset_ms = slice_audio_by_ms(
        audio,
        start_ms=30,
        end_ms=50,
        padding_ms=10,
        sample_rate=1000,
    )
    decodes = [
        asr_segment_decode(index=1, text="好", start_ms=50, end_ms=100, timestamps=[["好", 50, 80]]),
        asr_segment_decode(index=0, text="你", start_ms=0, end_ms=50, timestamps=[["你", 0, 40]]),
    ]

    assert sliced == list(range(20, 60))
    assert offset_ms == 20
    assert combine_asr_segment_decodes(decodes, 2) == (
        "你好",
        [["你", 0, 40], ["好", 50, 80]],
        [
            {"text": "你", "start": 0, "end": 50, "timestamp": [["你", 0, 40]]},
            {"text": "好", "start": 50, "end": 100, "timestamp": [["好", 50, 80]]},
        ],
    )


def test_offline_onnx_speaker_merge_falls_back_to_asr_sentences_when_spk_unavailable():
    payload = {
        "text": "你好",
        "asr_segments": [
            {"text": "你好", "start": 0, "end": 600, "timestamp": [["你", 0, 200], ["好", 200, 600]]}
        ],
    }

    merged = merge_onnx_speaker_result(payload, SpeakerResult(error="spk failed"))

    assert merged["speaker_error"] == "spk failed"
    assert merged["sentence_info"] == [
        {"text": "你好", "start": 0, "end": 600, "spk": 0, "timestamp": [["你", 0, 200], ["好", 200, 600]]}
    ]


def test_offline_onnx_speaker_merge_assigns_speaker_to_asr_segments_without_timestamps():
    payload = {
        "text": "你好世界",
        "asr_segments": [
            {"text": "你好", "start": 0, "end": 500},
            {"text": "世界", "start": 500, "end": 1000},
        ],
    }
    speaker = SpeakerResult(
        segments=[
            SpeakerSegment(speaker="A", start=0, end=500),
            SpeakerSegment(speaker="B", start=500, end=1000),
        ],
        speaker_ids=["A", "B"],
        speaker_count=2,
    )

    merged = merge_onnx_speaker_result(payload, speaker)

    assert merged["sentence_info"] == [
        {"text": "你好", "start": 0, "end": 500, "spk": "A", "timestamp": None},
        {"text": "世界", "start": 500, "end": 1000, "spk": "B", "timestamp": None},
    ]
    assert merged["speaker_result"]["speaker_ids"] == ["A", "B"]


def test_offline_onnx_short_token_crossing_speaker_boundary_moves_to_next_speaker():
    bundle = OfflineONNXModelBundle.__new__(OfflineONNXModelBundle)
    speaker = SpeakerResult(
        segments=[
            SpeakerSegment(speaker=0, start=900, end=13600),
            SpeakerSegment(speaker=1, start=13600, end=15180),
        ],
        speaker_ids=[0, 1],
        speaker_count=2,
    )

    payload = bundle.merge_speaker_result(
        {
            "text": "机动，你喇个也摘了",
            "raw_text": "机 动 你 喇 个 也 摘 了",
            "timestamps": [
                ["机", 13270, 13410],
                ["动", 13410, 13530],
                ["你", 13530, 13650],
                ["喇", 13650, 13830],
                ["个", 13830, 13950],
                ["也", 13950, 14050],
                ["摘", 14050, 14130],
                ["了", 14130, 14330],
            ],
            "asr_segments": [],
        },
        speaker,
    )

    assert payload["sentence_info"] == [
        {"text": "机动，", "start": 13270, "end": 13530, "spk": 0},
        {"text": "你喇个也摘了", "start": 13530, "end": 14330, "spk": 1},
    ]
