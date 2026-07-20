import ast
import sys
import threading
import types
from pathlib import Path

import numpy as np

from src.engine_runtime.engines.online.onnx.adapters import (
    ONNXFinalASRWrapper as LegacyFinalASRWrapper,
    ONNXPuncWrapper as LegacyPuncWrapper,
    ONNXRealtimeUnsupportedError as LegacyUnsupportedError,
    ONNXStreamingASRWrapper as LegacyStreamingASRWrapper,
    ONNXVADSession as LegacyVADSession,
    ONNXVADWrapper as LegacyVADWrapper,
)
from src.engine_runtime.engines.online.onnx.common import ONNXRealtimeUnsupportedError
from src.engine_runtime.engines.online.onnx.final_asr import ONNXFinalASRWrapper
from src.engine_runtime.engines.online.onnx.punctuation import ONNXPuncWrapper
from src.engine_runtime.engines.online.onnx.streaming_asr import ONNXStreamingASRWrapper
from src.engine_runtime.engines.online.onnx.vad import ONNXVADSession, ONNXVADWrapper


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_adapter_module_reexports_split_classes():
    assert LegacyFinalASRWrapper is ONNXFinalASRWrapper
    assert LegacyPuncWrapper is ONNXPuncWrapper
    assert LegacyStreamingASRWrapper is ONNXStreamingASRWrapper
    assert LegacyVADSession is ONNXVADSession
    assert LegacyVADWrapper is ONNXVADWrapper
    assert LegacyUnsupportedError is ONNXRealtimeUnsupportedError
    assert ONNXStreamingASRWrapper.__module__.endswith(".streaming_asr")
    assert ONNXVADWrapper.__module__.endswith(".vad")
    assert ONNXFinalASRWrapper.__module__.endswith(".final_asr")
    assert ONNXPuncWrapper.__module__.endswith(".punctuation")


def test_legacy_adapter_module_contains_no_implementation_classes():
    path = ROOT / "src" / "engine_runtime" / "engines" / "online" / "onnx" / "adapters.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)


def test_streaming_adapter_normalizes_audio_and_filters_unsupported_kwargs():
    captured = {}

    class Model:
        def generate(self, audio, param_dict=None, encoder_chunk_look_back=None):
            captured["audio"] = audio
            captured["param_dict"] = param_dict
            captured["encoder_chunk_look_back"] = encoder_chunk_look_back
            return [{"text": "partial"}]

    wrapper = ONNXStreamingASRWrapper.__new__(ONNXStreamingASRWrapper)
    wrapper.model = Model()
    wrapper._model_lock = threading.Lock()
    cache = {"state": 1}

    result = wrapper.generate(
        input=[[1, 2], [3, 4]],
        cache=cache,
        is_final=True,
        encoder_chunk_look_back=6,
        decoder_chunk_look_back=2,
        hotwords=["keyword"],
        unsupported="ignored",
    )

    assert result == [{"text": "partial"}]
    assert captured["audio"].dtype == np.float32
    assert captured["audio"].shape == (4,)
    assert captured["param_dict"] == {"cache": cache, "is_final": True}
    assert captured["encoder_chunk_look_back"] == 6


def test_final_adapter_filters_offline_only_kwargs_and_preserves_result():
    captured = {}

    class Model:
        def __call__(self, audio, hotwords=None, accepted=None):
            captured["audio"] = audio
            captured["hotwords"] = hotwords
            captured["accepted"] = accepted
            return [{"text": "final"}]

    wrapper = ONNXFinalASRWrapper.__new__(ONNXFinalASRWrapper)
    wrapper.asr_model = Model()
    wrapper._model_lock = threading.Lock()

    result = wrapper.generate(
        input=[1, 2, 3],
        hotwords=["keyword"],
        accepted="yes",
        batch_size_s=300,
        return_spk_res=True,
        unsupported="ignored",
    )

    assert result == [{"text": "final"}]
    assert captured["audio"].dtype == np.float32
    assert captured["hotwords"] == ["keyword"]
    assert captured["accepted"] == "yes"


def test_final_adapter_loads_seaco_model_and_falls_back_to_non_quantized(monkeypatch, tmp_path):
    created = []
    model_dir = tmp_path / "seaco"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("model: SeacoParaformer\n", encoding="utf-8")
    (model_dir / "model.onnx").touch()
    (model_dir / "model_eb.onnx").touch()

    def factory(label):
        def build(**kwargs):
            created.append((label, kwargs))

            class Model:
                def __call__(self, audio, hotwords="", **_kwargs):
                    return [{"text": f"{label}:{hotwords}"}]

            return Model()

        return build

    fake_module = types.SimpleNamespace(
        Paraformer=factory("plain"),
        SeacoParaformer=factory("seaco"),
    )
    monkeypatch.setitem(sys.modules, "funasr_onnx", fake_module)

    wrapper = ONNXFinalASRWrapper(str(model_dir), quantize=True)

    assert wrapper.asr_hotword_mode == "seaco"
    assert created == [
        (
            "seaco",
            {
                "model_dir": str(model_dir),
                "batch_size": 1,
                "quantize": False,
                "intra_op_num_threads": 4,
                "device_id": "-1",
            },
        )
    ]


def test_final_adapter_seaco_formats_hotwords_for_model():
    captured = {}

    class Model:
        def __call__(self, audio, hotwords="", accepted=None):
            captured["audio"] = audio
            captured["hotwords"] = hotwords
            captured["accepted"] = accepted
            return [{"text": "final"}]

    wrapper = ONNXFinalASRWrapper.__new__(ONNXFinalASRWrapper)
    wrapper.asr_model = Model()
    wrapper.asr_hotword_mode = "seaco"
    wrapper._model_lock = threading.Lock()

    result = wrapper.generate(
        input=[1, 2, 3],
        hotwords=[[100, "篮子"], [80, "直播"]],
        accepted="yes",
        batch_size_s=300,
    )

    assert result == [{"text": "final"}]
    assert captured["hotwords"] == "篮子 直播"
    assert captured["accepted"] == "yes"


def test_punctuation_helpers_remove_whitespace_before_fallback():
    assert ONNXPuncWrapper._fallback_punctuation("你 好  吗") == "你好吗。"
    assert ONNXPuncWrapper._split_text_for_punc("你 好", max_chars=80) == ["你好"]
