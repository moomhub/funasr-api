import os
import sys
import types
from pathlib import Path

import pytest

from src.core.config.errors import ModelResolutionError
from src.core.models.downloader import ModelDownloader


def _install_fake_modelscope(monkeypatch, snapshot_download):
    modelscope = types.ModuleType("modelscope")
    hub = types.ModuleType("modelscope.hub")
    snapshot = types.ModuleType("modelscope.hub.snapshot_download")
    snapshot.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", snapshot)


def test_modelscope_does_not_append_models_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_CACHE", "test-sentinel")
    configured_root = tmp_path / "data" / "models"
    captured = {}

    def fake_snapshot_download(repo_id, cache_dir, **kwargs):
        captured.update(repo_id=repo_id, cache_dir=cache_dir)
        target = Path(cache_dir) / "models" / "iic" / "demo-model"
        target.mkdir(parents=True)
        (target / "config.yaml").write_text("model: demo", encoding="utf-8")
        return str(target)

    _install_fake_modelscope(monkeypatch, fake_snapshot_download)
    downloader = ModelDownloader(str(configured_root), auto_download=True)

    resolved = downloader.ensure_model("iic/demo-model")

    expected = configured_root / "iic" / "demo-model"
    assert captured["cache_dir"] == str(configured_root.parent)
    assert os.environ["MODELSCOPE_CACHE"] == str(configured_root.parent)
    assert Path(resolved) == expected
    assert expected.is_dir()
    assert not (configured_root / "models").exists()


def test_nonstandard_model_root_keeps_existing_modelscope_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_CACHE", "test-sentinel")
    configured_root = tmp_path / "damo"
    captured = {}

    def fake_snapshot_download(repo_id, cache_dir, **kwargs):
        captured["cache_dir"] = cache_dir
        target = Path(cache_dir) / "models" / "iic" / "demo-model"
        target.mkdir(parents=True)
        (target / "config.yaml").write_text("model: demo", encoding="utf-8")
        return str(target)

    _install_fake_modelscope(monkeypatch, fake_snapshot_download)
    downloader = ModelDownloader(str(configured_root), auto_download=True)

    assert Path(downloader.ensure_model("iic/demo-model")) == (
        configured_root / "models" / "iic" / "demo-model"
    )
    assert captured["cache_dir"] == str(configured_root)


def test_double_models_cache_is_not_scanned(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_CACHE", "test-sentinel")
    legacy = tmp_path / "models" / "models" / "iic" / "legacy-model"
    legacy.mkdir(parents=True)
    (legacy / "config.yaml").write_text("model: legacy", encoding="utf-8")
    downloader = ModelDownloader(str(tmp_path / "models"), auto_download=False)

    with pytest.raises(ModelResolutionError, match="未在本地缓存中找到"):
        downloader.ensure_model("iic/legacy-model")
