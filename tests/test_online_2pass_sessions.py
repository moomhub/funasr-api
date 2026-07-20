import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine_runtime.engines.online.session_manager import (
    OnlineOnnxRealtimeSession,
    OnlineRealtimeSession,
)
from src.engine_runtime.engines.online.onnx.adapters import ONNXPuncWrapper


def _pcm_bytes(duration_ms=400, sample_rate=16000, amplitude=10000):
    samples = int(sample_rate * duration_ms / 1000)
    return np.full(samples, amplitude, dtype=np.int16).tobytes()


class _FakeStreamingASR:
    def __init__(self, text="实时结果"):
        self.text = text

    def generate(self, *args, **kwargs):
        return [{"text": self.text}]


class _FakeFinalASR:
    def __init__(self, text="最终结果"):
        self.text = text

    def generate(self, *args, **kwargs):
        return [{"text": self.text}]


class _FakeTimestampFinalASR:
    def __init__(self, text="你 好", timestamp=None):
        self.text = text
        self.timestamp = timestamp or [[0, 120], [120, 260]]

    def generate(self, *args, **kwargs):
        return [{"text": self.text, "timestamp": self.timestamp}]


class _FakePunc:
    def generate(self, input=None, *args, **kwargs):
        return [{"text": f"{input}。"}]


class _BrokenOnnxPuncModel:
    def __call__(self, *args, **kwargs):
        raise UnboundLocalError("local variable 'punctuations' referenced before assignment")


class _FakeVad:
    def __init__(self, segments=None, final_segments=None):
        self.segments = list(segments or [])
        self.final_segments = list(final_segments or [])
        self.current_speech_start = None

    def reset(self):
        self.current_speech_start = None

    def feed(self, audio, is_final=False):
        if is_final:
            return self.final_segments
        if not self.segments:
            return []
        return [self.segments.pop(0)]


def _pt_session(vad):
    return OnlineRealtimeSession(
        streaming_asr_model=_FakeStreamingASR(),
        vad_model=vad,
        final_model=_FakeFinalASR(),
        punc_model=_FakePunc(),
        sample_rate=16000,
        decode_interval=0,
        first_decode_ms=10,
        chunk_ms=10,
        vad_post_padding_ms=0,
        vad_merge_gap_ms=0,
        vad_min_final_ms=1,
        vad_max_final_ms=1,
        vad_adapter_factory=lambda model: model,
    )


def _pt_session_with_final(vad, final_model, **kwargs):
    return OnlineRealtimeSession(
        streaming_asr_model=_FakeStreamingASR(),
        vad_model=vad,
        final_model=final_model,
        punc_model=_FakePunc(),
        sample_rate=16000,
        decode_interval=0,
        first_decode_ms=10,
        chunk_ms=10,
        vad_pre_padding_ms=kwargs.pop("vad_pre_padding_ms", 0),
        vad_post_padding_ms=kwargs.pop("vad_post_padding_ms", 0),
        vad_merge_gap_ms=0,
        vad_min_final_ms=1,
        vad_max_final_ms=1,
        vad_adapter_factory=lambda model: model,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_pt_session_outputs_2pass_online_partial():
    session = _pt_session(_FakeVad())
    await session.add_audio(_pcm_bytes())

    event = await session.decode_partial()

    assert event["mode"] == "2pass-online"
    assert event["partial"] == "实时结果"
    assert event["is_final"] is False


@pytest.mark.asyncio
async def test_pt_session_outputs_2pass_offline_after_vad_endpoint():
    session = _pt_session(_FakeVad(segments=[[0, 300]]))

    locked = await session.add_audio(_pcm_bytes())
    event = session.build_offline_event()

    assert locked is True
    assert event["mode"] == "2pass-offline"
    assert event["text"] == "最终结果。"
    assert event["sentences"][0]["is_final"] is True


@pytest.mark.asyncio
async def test_onnx_session_uses_streaming_vad_final_and_punc_wrappers():
    session = OnlineOnnxRealtimeSession(
        streaming_asr_model=_FakeStreamingASR("onnx实时"),
        vad_model=_FakeVad(segments=[[0, 300]]),
        final_model=_FakeFinalASR("onnx最终"),
        punc_model=_FakePunc(),
        sample_rate=16000,
        decode_interval=0,
        first_decode_ms=10,
        chunk_ms=10,
        vad_post_padding_ms=0,
        vad_merge_gap_ms=0,
        vad_min_final_ms=1,
        vad_max_final_ms=1,
    )

    await session.add_audio(_pcm_bytes())
    online_event = await session.decode_partial()
    offline_event = session.build_offline_event()

    assert online_event["mode"] == "2pass-online"
    assert online_event["partial"] == "onnx实时"
    assert offline_event["mode"] == "2pass-offline"
    assert offline_event["text"] == "onnx最终。"


@pytest.mark.asyncio
async def test_stop_finish_flushes_tail_as_final_sentence():
    session = _pt_session(_FakeVad(final_segments=[[0, 300]]))
    await session.add_audio(_pcm_bytes())

    response = await session.finish()
    event = session.build_offline_event(is_final=True)

    assert response is None
    assert event["mode"] == "2pass-offline"
    assert event["is_final"] is True
    assert event["text"] == "最终结果。"


@pytest.mark.asyncio
async def test_pt_final_sentence_includes_word_timestamps():
    session = _pt_session_with_final(
        _FakeVad(segments=[[0, 300]]),
        _FakeTimestampFinalASR(),
    )

    await session.add_audio(_pcm_bytes())
    event = session.build_offline_event()
    sentence = event["sentences"][0]

    assert sentence["text"] == "你好。"
    assert sentence["raw_text"] == "你好"
    assert sentence["timestamp"] == [["你", 0, 120], ["好", 120, 260]]
    assert sentence["tokens"] == [
        {"text": "你", "start": 0, "end": 120},
        {"text": "好", "start": 120, "end": 260},
    ]


@pytest.mark.asyncio
async def test_final_timestamps_are_shifted_from_padded_audio_to_global_time():
    session = _pt_session_with_final(
        _FakeVad(segments=[[1000, 1300]]),
        _FakeTimestampFinalASR(timestamp=[[300, 420], [420, 560]]),
        vad_pre_padding_ms=300,
    )
    await session.add_audio(_pcm_bytes(duration_ms=1800))

    event = session.build_offline_event()
    sentence = event["sentences"][0]

    assert sentence["start"] == 1000
    assert sentence["end"] == 1300
    assert sentence["asr_start"] == 700
    assert sentence["timestamp"] == [["你", 1000, 1120], ["好", 1120, 1260]]


@pytest.mark.asyncio
async def test_onnx_final_sentence_preserves_timestamp_payload():
    session = OnlineOnnxRealtimeSession(
        streaming_asr_model=_FakeStreamingASR("onnx实时"),
        vad_model=_FakeVad(segments=[[0, 300]]),
        final_model=_FakeTimestampFinalASR("开 始", [["开", 10, 80], ["始", 80, 180]]),
        punc_model=_FakePunc(),
        sample_rate=16000,
        decode_interval=0,
        first_decode_ms=10,
        chunk_ms=10,
        vad_post_padding_ms=0,
        vad_merge_gap_ms=0,
        vad_min_final_ms=1,
        vad_max_final_ms=1,
    )

    await session.add_audio(_pcm_bytes())
    event = session.build_offline_event()

    assert event["sentences"][0]["text"] == "开始。"
    assert event["sentences"][0]["timestamp"] == [["开", 10, 80], ["始", 80, 180]]


def test_onnx_punc_wrapper_falls_back_when_model_raises_unboundlocalerror():
    wrapper = ONNXPuncWrapper.__new__(ONNXPuncWrapper)
    wrapper.model = _BrokenOnnxPuncModel()

    result = wrapper.generate("它不会发面蒸好的馒头蓬松暄软里面可细腻了能撕掉一层特别筋道的小皮儿")

    assert result[0]["text"].endswith("。")
    assert "，" not in result[0]["text"]


class _SensitiveFailurePunc:
    def generate(self, *args, **kwargs):
        raise RuntimeError("private recognition text")


def test_realtime_punctuation_warning_redacts_exception_message(caplog):
    session = _pt_session(_FakeVad())
    session.punc_model = _SensitiveFailurePunc()

    with caplog.at_level(logging.DEBUG, logger="src.engine_runtime.engines.online.realtime_session"):
        result = session._apply_punctuation("private input text")

    warning_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    debug_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.DEBUG]
    assert result == "private input text"
    assert warning_messages == [
        "ONLINE punctuation failed, fallback to ASR text: error_type=RuntimeError"
    ]
    assert all("private recognition text" not in message for message in warning_messages)
    assert debug_messages == ["ONLINE punctuation failure details"]
