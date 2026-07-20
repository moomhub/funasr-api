from types import SimpleNamespace

from src.engine_runtime.state import (
    RuntimeModelCache,
    RuntimeRecognizerCache,
    build_runtime_loader_bundle,
)


def test_runtime_loader_bundle_shares_one_downloader(tmp_path):
    loaders = build_runtime_loader_bundle(
        model_dir=str(tmp_path),
        auto_download=False,
        pt_device="cpu",
        pt_disable_update=True,
        onnx_quantize=True,
        onnx_threads=2,
        onnx_device_id=-1,
    )

    assert loaders.pt_loader.downloader is loaders.downloader
    assert loaders.online_onnx_loader.downloader is loaders.downloader
    assert loaders.spk_pt_loader.downloader is loaders.downloader


def test_runtime_model_cache_counts_loaded_onnx_related_models():
    cache = RuntimeModelCache()
    online_loader = SimpleNamespace(get_loaded_models_count=lambda: 2)
    spk_loader = SimpleNamespace(get_loaded_models_count=lambda: 1)

    assert cache.loaded_onnx_model_count(
        online_onnx_loader=online_loader,
        spk_pt_loader=spk_loader,
    ) == 3

    cache.offline_onnx_model_bundle = object()

    assert cache.loaded_onnx_model_count(
        online_onnx_loader=online_loader,
        spk_pt_loader=spk_loader,
    ) == 6


def test_runtime_caches_create_values_once():
    model_cache = RuntimeModelCache()
    recognizer_cache = RuntimeRecognizerCache()
    calls = []

    first_model = model_cache.get_or_create(
        "online_pt_model_bundle",
        lambda: calls.append("model") or object(),
    )
    second_model = model_cache.get_or_create(
        "online_pt_model_bundle",
        lambda: calls.append("model-again") or object(),
    )
    first_recognizer = recognizer_cache.get_or_create(
        "spk_pt_recognizer",
        lambda: calls.append("recognizer") or object(),
    )
    second_recognizer = recognizer_cache.get_or_create(
        "spk_pt_recognizer",
        lambda: calls.append("recognizer-again") or object(),
    )

    assert first_model is second_model
    assert first_recognizer is second_recognizer
    assert calls == ["model", "recognizer"]
