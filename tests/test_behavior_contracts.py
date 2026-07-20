from src.api.offline import router as offline_router
from src.api.online import router as online_router
from src.api.spk import router as spk_router
from src.api.system import router as system_router
from src.core.models.downloader import ModelDownloader


def test_route_table_contains_only_current_public_recognition_paths():
    route_paths = {
        route.path
        for router in (offline_router, online_router, spk_router, system_router)
        for route in router.routes
        if getattr(route, "path", None) is not None
    }

    assert {
        "/offline/recognize",
        "/tasks/{task_id}",
        "/spk/recognize",
        "/spk/tasks/{task_id}",
        "/online/stream",
    }.issubset(route_paths)
    assert "/upload" not in route_paths
    assert "/online/realtime" not in route_paths


def test_model_downloader_prefers_existing_cache_without_download(tmp_path, monkeypatch):
    model_path = tmp_path / "iic" / "demo-model"
    model_path.mkdir(parents=True)
    (model_path / "config.yaml").write_text("model: demo", encoding="utf-8")
    downloader = ModelDownloader(str(tmp_path), auto_download=True)

    def fail_download(_repo_id):
        raise AssertionError("existing models must not be downloaded again")

    monkeypatch.setattr(downloader, "_download", fail_download)

    assert downloader.ensure_model("iic/demo-model") == str(model_path)
