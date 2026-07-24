from src.core.config.loader import ConfigLoader
from src.engine_runtime.manager import EngineModelManager


def _manager(tmp_path, *, offline_backend="onnx"):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
engines:
  enabled:
    - offline
  auto_model_download: false
  models:
    offline:
      enabled: {offline_backend}
    spk:
      spk: shared-speaker-model
""",
        encoding="utf-8",
    )
    return EngineModelManager(ConfigLoader(str(config_path)))


def test_manager_and_offline_runner_share_one_speaker_recognizer(tmp_path):
    manager = _manager(tmp_path)
    offline = manager.get_offline_onnx_recognizer()

    first = manager.get_spk_recognizer()
    second = manager.get_spk_pt_recognizer()

    assert first is second
    assert offline._speaker_runner.model_manager is manager
    assert first.model_manager is manager


def test_replacing_downloader_keeps_all_runtime_loaders_in_sync(tmp_path):
    manager = _manager(tmp_path)
    replacement = object()

    manager.downloader = replacement

    assert manager.downloader is replacement
    assert manager.pt_loader.downloader is replacement
    assert manager.online_onnx_loader.downloader is replacement
    assert manager.spk_pt_loader.downloader is replacement


def test_unload_all_invalidates_provider_bundle_caches(tmp_path):
    manager = _manager(tmp_path)
    manager.model_cache.offline_onnx_model_bundle = object()
    manager.model_cache.online_pt_model_bundle = object()
    manager.model_cache.online_onnx_model_bundle = object()

    manager.unload_model()

    assert manager.model_cache.offline_onnx_model_bundle is None
    assert manager.model_cache.online_pt_model_bundle is None
    assert manager.model_cache.online_onnx_model_bundle is None


def test_loader_replacement_invalidates_only_its_bundle_cache(tmp_path):
    manager = _manager(tmp_path)
    offline_bundle = object()
    manager.model_cache.offline_onnx_model_bundle = offline_bundle
    manager.model_cache.online_pt_model_bundle = object()
    manager.model_cache.online_onnx_model_bundle = object()

    manager.pt_loader = manager.pt_loader

    assert manager.model_cache.online_pt_model_bundle is None
    assert manager.model_cache.online_onnx_model_bundle is not None
    assert manager.model_cache.offline_onnx_model_bundle is offline_bundle


def test_offline_pt_model_loads_embedded_speaker_model(tmp_path):
    manager = _manager(tmp_path, offline_backend="pt")
    calls = []

    class CapturingLoader:
        def load_model(self, **kwargs):
            calls.append(kwargs)
            return object()

    manager.pt_loader = CapturingLoader()

    manager.get_offline_model()

    assert calls[0]["spk_model"] == "iic/speech_campplus_sv_zh-cn_16k-common"
    assert calls[0]["cache_key"].count(":") == 4
