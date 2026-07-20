import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config.loader import ConfigLoader
from src.core.config.errors import EngineConfigurationError, ModelLoadError
from src.core.results import SpeakerResult, SpeakerSegment
from src.engine_runtime.engines.spk.base import SpeakerRecognitionRequest
from src.engine_runtime.engines.spk.normalizers import normalize_speaker_result
from src.engine_runtime.engines.spk.pt.loader import SpeakerPTPipelineLoader
from src.engine_runtime.engines.spk.pt.recognizer import PTSpeakerRecognizer
from src.engine_runtime.engines.spk.runner import StandaloneSpeakerRunner
from src.engine_runtime.manager import EngineModelManager
from src.engine_runtime.services.contracts import SpeakerRequest
from src.engine_runtime.services.speaker.pt_speaker_service import PTSpeakerService
from src.core.results.types import Segment


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return {
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "speaker-0"},
                {"start": 1.0, "end": 2.0, "speaker": "speaker-1"},
            ]
        }


class _FakeSpeakerRecognizer:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    async def recognize(self, request):
        self.requests.append(request)
        return self.payload


class _FakeManager:
    def __init__(self, model):
        self.model = model

    def get_spk_model(self):
        return self.model


@pytest.mark.asyncio
async def test_pt_speaker_recognizer_loads_waveform_before_calling_pipeline(monkeypatch):
    pipeline = _FakePipeline()
    recognizer = PTSpeakerRecognizer(_FakeManager(pipeline))

    fake_librosa = types.SimpleNamespace(
        load=lambda audio_path, sr=16000: ([0.1, 0.2, 0.3], sr)
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    payload = await recognizer.recognize(
        SpeakerRecognitionRequest(audio_path="demo.wav", generate_kwargs={"sample_rate": 8000})
    )

    assert payload["segments"][0]["speaker"] == "speaker-0"
    assert pipeline.calls[0]["args"] == ([0.1, 0.2, 0.3],)
    assert pipeline.calls[0]["kwargs"] == {}


def test_speaker_pt_pipeline_loader_uses_fixed_speaker_diarization_task(monkeypatch, tmp_path):
    model_dir = tmp_path / "speaker-diarization-model"
    model_dir.mkdir()
    (model_dir / "configuration.json").write_text(
        '{"task":"speaker-diarization","pipeline":{"type":"segmentation-clustering"}}',
        encoding="utf-8",
    )

    class _FakeDownloader:
        def ensure_model(self, model_name, prefer_repo_id=True):
            assert model_name == "iic/speech_campplus_speaker-diarization_common"
            return str(model_dir)

    captured = {}

    def fake_pipeline(task, model):
        captured["task"] = task
        captured["model"] = model
        return _FakePipeline()

    fake_modelscope = types.ModuleType("modelscope")
    fake_pipelines = types.ModuleType("modelscope.pipelines")
    fake_pipelines.pipeline = fake_pipeline
    fake_modelscope.pipelines = fake_pipelines
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.pipelines", fake_pipelines)

    loader = SpeakerPTPipelineLoader(_FakeDownloader())
    model = loader.load_model(spk_name="iic/speech_campplus_speaker-diarization_common")

    assert isinstance(model, _FakePipeline)
    assert captured["task"] == "speaker-diarization"
    assert captured["model"] == str(model_dir)


def test_speaker_pt_pipeline_loader_rejects_campplus_verification_model(tmp_path):
    verification_dir = tmp_path / "speaker-verification-model"
    verification_dir.mkdir()
    (verification_dir / "configuration.json").write_text(
        '{"task":"speaker-verification","pipeline":{"type":"speaker-verification"}}',
        encoding="utf-8",
    )
    class _FakeDownloader:
        def ensure_model(self, model_name, prefer_repo_id=True):
            assert model_name == "iic/speech_campplus_sv_zh-cn_16k-common"
            return str(verification_dir)

    loader = SpeakerPTPipelineLoader(_FakeDownloader())

    with pytest.raises(ModelLoadError, match="仅支持 speaker-diarization"):
        loader.load_model(spk_name="iic/speech_campplus_sv_zh-cn_16k-common")


def test_speaker_pt_pipeline_loader_rejects_unknown_verification_model(tmp_path):
    model_dir = tmp_path / "speaker-verification-model"
    model_dir.mkdir()
    (model_dir / "configuration.json").write_text(
        '{"task":"speaker-verification","pipeline":{"type":"speaker-verification"}}',
        encoding="utf-8",
    )

    class _FakeDownloader:
        def ensure_model(self, model_name, prefer_repo_id=True):
            return str(model_dir)

    loader = SpeakerPTPipelineLoader(_FakeDownloader())

    with pytest.raises(ModelLoadError, match="仅支持 speaker-diarization"):
        loader.load_model(spk_name="custom/speaker-verification-model")


def test_engine_model_manager_rejects_legacy_spk_onnx_backend(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - spk
  models:
    spk:
      enabled: onnx
      spk: iic/speech_campplus_speaker-diarization_common
""",
        encoding="utf-8",
    )

    with pytest.raises(EngineConfigurationError, match="engines.models.spk.enabled.*engines.models.spk.spk"):
        ConfigLoader(str(config_path))


def test_engine_model_manager_reads_new_flat_spk_model_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - spk
  models:
    spk:
      spk: custom-speaker-model
""",
        encoding="utf-8",
    )

    manager = EngineModelManager(ConfigLoader(str(config_path)))

    assert manager.get_backend_for_mode("spk") == "pt"
    assert manager.engines_config.models.spk.spk == "custom-speaker-model"


def test_engine_model_manager_allows_spk_runtime_for_offline_only_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - offline
  models:
    spk:
      spk: offline-shared-speaker-model
""",
        encoding="utf-8",
    )
    manager = EngineModelManager(ConfigLoader(str(config_path)))
    calls = []

    class _FakeSpeakerLoader:
        def load_model(self, spk_name, cache_key):
            calls.append({"spk_name": spk_name, "cache_key": cache_key})
            return "speaker-runtime"

    manager.spk_pt_loader = _FakeSpeakerLoader()

    assert "spk" not in manager.enabled_modes
    assert manager.get_spk_model() == "speaker-runtime"
    assert calls == [
        {
            "spk_name": "offline-shared-speaker-model",
            "cache_key": "spk-pt:offline-shared-speaker-model",
        }
    ]


def test_engine_model_manager_rejects_spk_runtime_when_not_required(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - online
  models:
    spk:
      spk: unused-speaker-model
""",
        encoding="utf-8",
    )
    manager = EngineModelManager(ConfigLoader(str(config_path)))

    with pytest.raises(EngineConfigurationError, match="SPK runtime 未启用"):
        manager.get_spk_model()


def test_segment_to_dict_normalizes_numpy_scalars_for_json_storage():
    segment = Segment(
        text="你好",
        start=np.int32(100),
        end=np.int32(300),
        speaker=np.int32(2),
        timestamp=[["你", np.int32(100), np.int32(180)], ["好", np.int32(180), np.int32(300)]],
    )

    payload = segment.to_dict()

    assert payload["start"] == 100
    assert isinstance(payload["start"], int)
    assert payload["end"] == 300
    assert isinstance(payload["speaker"], int)
    assert payload["speaker"] == 2
    assert "spk" not in payload
    assert payload["timestamp"][0][1] == 100
    assert isinstance(payload["timestamp"][0][1], int)


def test_speaker_normalizer_copies_existing_result_metadata():
    original = SpeakerResult(
        segments=[SpeakerSegment(speaker="A", start=0, end=1000)],
        speaker_ids=["A"],
        speaker_count=1,
        metadata={"origin": "payload"},
    )

    normalized = normalize_speaker_result(original)
    normalized.metadata["service"] = "speaker_pt"
    normalized.segments[0].speaker = "B"

    assert original.metadata == {"origin": "payload"}
    assert original.segments[0].speaker == "A"


@pytest.mark.asyncio
async def test_standalone_speaker_runner_uses_shared_recognizer_and_request_objects(caplog):
    caplog.set_level(logging.DEBUG, logger="src.engine_runtime.engines.spk.runner")
    recognizer = _FakeSpeakerRecognizer(
        {
            "segments": [
                {"start": 0, "end": 500, "speaker": "A"},
            ]
        }
    )

    class _Manager:
        def get_spk_recognizer(self):
            return recognizer

    runner = StandaloneSpeakerRunner(_Manager())
    first = await runner.recognize(
        SpeakerRecognitionRequest(audio_path="first.wav", generate_kwargs={"sample_rate": 16000}),
        metadata={"source": "first"},
    )
    second = await runner.recognize(
        SpeakerRecognitionRequest(audio_path="second.wav", generate_kwargs={"sample_rate": 8000}),
        metadata={"source": "second"},
    )

    assert [request.audio_path for request in recognizer.requests] == ["first.wav", "second.wav"]
    assert recognizer.requests[0].generate_kwargs == {"sample_rate": 16000}
    assert recognizer.requests[1].generate_kwargs == {"sample_rate": 8000}
    assert first.speaker_ids == ["A"]
    assert first.metadata["source"] == "first"
    assert second.metadata["source"] == "second"
    debug_messages = [record.getMessage() for record in caplog.records]
    assert any("Standalone SPK input" in message for message in debug_messages)
    assert any("Standalone SPK output" in message for message in debug_messages)
    assert any("first.wav" in message for message in debug_messages)
    assert any('"speaker": "A"' in message for message in debug_messages)


@pytest.mark.asyncio
async def test_pt_speaker_recognizer_raises_backend_failures(monkeypatch):
    class BrokenPipeline:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("speaker pipeline failed")

    recognizer = PTSpeakerRecognizer(_FakeManager(BrokenPipeline()))
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        types.SimpleNamespace(load=lambda _path, sr=16000: ([0.1], sr)),
    )

    with pytest.raises(RuntimeError, match="speaker pipeline failed"):
        await recognizer.recognize(
            SpeakerRecognitionRequest(audio_path="broken.wav")
        )


@pytest.mark.asyncio
async def test_standalone_runner_converts_backend_failure_to_speaker_result(caplog):
    class BrokenRecognizer:
        async def recognize(self, _request):
            raise RuntimeError("speaker pipeline failed")

    class Manager:
        def get_spk_recognizer(self):
            return BrokenRecognizer()

    caplog.set_level(logging.DEBUG, logger="src.engine_runtime.engines.spk.runner")
    result = await StandaloneSpeakerRunner(Manager()).recognize(
        SpeakerRecognitionRequest(audio_path="broken.wav")
    )

    assert result.error == "speaker pipeline failed"
    error_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert all("speaker pipeline failed" not in message for message in error_messages)
    assert any("Standalone SPK recognition failure details" in message for message in debug_messages)


@pytest.mark.asyncio
async def test_pt_speaker_service_uses_runner_without_mutating_payload_metadata():
    payload = SpeakerResult(
        segments=[SpeakerSegment(speaker="A", start=0, end=1000)],
        speaker_ids=["A"],
        speaker_count=1,
        metadata={"origin": "recognizer"},
    )
    recognizer = _FakeSpeakerRecognizer(payload)

    class _Manager:
        def get_spk_pt_recognizer(self):
            return recognizer

        def get_spk_recognizer(self):
            return recognizer

    service = PTSpeakerService(_Manager())
    service._loaded = True

    result = await service.diarize(
        SpeakerRequest(
            audio_path="demo.wav",
            generate_kwargs={"sample_rate": 16000},
            metadata={"request_id": "req-1"},
        )
    )

    assert recognizer.requests[0].audio_path == "demo.wav"
    assert result.metadata["origin"] == "recognizer"
    assert result.metadata["request_id"] == "req-1"
    assert result.metadata["service"] == "speaker_pt"
    assert payload.metadata == {"origin": "recognizer"}

