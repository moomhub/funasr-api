from src.engine_runtime.services.service_plan import enabled_service_keys, required_service_modes


def test_enabled_service_keys_preloads_offline_speaker_dependency_once():
    assert enabled_service_keys(["offline"]) == ["offline_asr", "speaker"]
    assert enabled_service_keys(["offline", "spk"]) == ["offline_asr", "speaker"]
    assert enabled_service_keys(["spk", "offline"]) == ["speaker", "offline_asr"]


def test_enabled_service_keys_keeps_online_independent():
    assert enabled_service_keys(["online"]) == ["online_asr"]
    assert enabled_service_keys(["online", "offline"]) == ["online_asr", "offline_asr", "speaker"]


def test_required_service_modes_documents_offline_spk_runtime_dependency():
    assert required_service_modes("offline") == ["offline", "spk"]
    assert required_service_modes("online") == ["online"]
    assert required_service_modes("spk") == ["spk"]
    assert required_service_modes("unknown") == ["unknown"]
