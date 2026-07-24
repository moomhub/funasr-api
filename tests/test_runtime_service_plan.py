from types import SimpleNamespace

from src.engine_runtime.services.factory import RuntimeServiceFactory
from src.engine_runtime.services.service_plan import enabled_service_keys, required_service_modes


def test_enabled_service_keys_preloads_offline_speaker_dependency_once():
    assert enabled_service_keys(["offline"]) == ["offline_asr", "speaker"]
    assert enabled_service_keys(["offline", "spk"]) == ["offline_asr", "speaker"]
    assert enabled_service_keys(["spk", "offline"]) == ["speaker", "offline_asr"]


def test_enabled_service_keys_skips_shared_speaker_for_pt_when_secondary_verification_disabled():
    assert enabled_service_keys(
        ["offline"],
        offline_backend="pt",
        offline_spk_verification_enabled=False,
    ) == ["offline_asr"]


def test_enabled_service_keys_keeps_shared_speaker_for_onnx_when_secondary_verification_disabled():
    assert enabled_service_keys(
        ["offline"],
        offline_backend="onnx",
        offline_spk_verification_enabled=False,
    ) == ["offline_asr", "speaker"]


def test_enabled_service_keys_keeps_online_independent():
    assert enabled_service_keys(["online"]) == ["online_asr"]
    assert enabled_service_keys(["online", "offline"]) == ["online_asr", "offline_asr", "speaker"]


def test_required_service_modes_documents_offline_spk_runtime_dependency():
    assert required_service_modes("offline") == ["offline", "spk"]
    assert required_service_modes("online") == ["online"]
    assert required_service_modes("spk") == ["spk"]
    assert required_service_modes("unknown") == ["unknown"]
    assert required_service_modes(
        "offline",
        offline_backend="pt",
        offline_spk_verification_enabled=False,
    ) == ["offline"]
    assert required_service_modes(
        "offline",
        offline_backend="onnx",
        offline_spk_verification_enabled=False,
    ) == ["offline", "spk"]


def test_runtime_service_factory_does_not_preload_shared_speaker_when_disabled():
    calls = []

    class Manager:
        enabled_modes = ["offline"]
        processing_config = SimpleNamespace(offline_spk_verification_enabled=False)

        def get_backend_for_mode(self, mode):
            return "pt"

        def get_offline_pt_recognizer(self):
            return SimpleNamespace(load_model=lambda: calls.append("offline_asr"))

        def get_spk_pt_recognizer(self):
            return SimpleNamespace(load_model=lambda: calls.append("speaker"))

    statuses = RuntimeServiceFactory(Manager()).preload_enabled_models()

    assert [status.service_name for status in statuses] == ["offline_asr_pt"]
    assert calls == ["offline_asr"]


def test_runtime_service_factory_keeps_shared_speaker_for_onnx_when_disabled():
    calls = []

    class Manager:
        enabled_modes = ["offline"]
        processing_config = SimpleNamespace(offline_spk_verification_enabled=False)

        def get_backend_for_mode(self, mode):
            return "onnx" if mode == "offline" else "pt"

        def get_offline_onnx_recognizer(self):
            return SimpleNamespace(load_model=lambda: calls.append("offline_asr"))

        def get_spk_pt_recognizer(self):
            return SimpleNamespace(load_model=lambda: calls.append("speaker"))

    statuses = RuntimeServiceFactory(Manager()).preload_enabled_models()

    assert [status.service_name for status in statuses] == ["offline_asr_onnx", "speaker_pt"]
    assert calls == ["offline_asr", "speaker"]
