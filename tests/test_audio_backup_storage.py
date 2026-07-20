from types import SimpleNamespace

import pytest

from src.core.adapters import S3AudioBackupStore
import src.core.infrastructure as infrastructure


class _UploadClient:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def upload_file(self, local_path, bucket, key, ExtraArgs=None):
        self.calls.append((local_path, bucket, key, ExtraArgs))
        if self.error is not None:
            raise self.error


def _make_store(client, prefix="archive"):
    store = S3AudioBackupStore.__new__(S3AudioBackupStore)
    store.bucket = "audio-bucket"
    store.prefix = prefix
    store.s3_client = client
    store.available = True
    store.last_error = None
    return store


@pytest.mark.asyncio
async def test_s3_audio_backup_uses_configured_prefix_and_generated_filename():
    client = _UploadClient()
    store = _make_store(client, prefix="private-archive")

    key = await store.backup_original(
        "C:/temp/upload.wav",
        "task-1",
        "patient-name.wav",
    )

    assert key.startswith("private-archive/task-1_")
    assert key.endswith(".wav")
    assert "patient-name" not in key
    assert client.calls[0][0:3] == ("C:/temp/upload.wav", "audio-bucket", key)
    assert client.calls[0][3]["Metadata"]["task_id"] == "task-1"
    assert store.available is True
    assert store.last_error is None


@pytest.mark.asyncio
async def test_s3_audio_backup_failure_updates_health_state():
    client = _UploadClient(error=RuntimeError("upload failed"))
    store = _make_store(client)

    key = await store.backup_original("C:/temp/upload.wav", "task-2", "demo.wav")

    assert key is None
    assert store.available is False
    assert store.last_error == "upload failed"


def test_build_audio_backup_store_passes_prefix_to_adapter(monkeypatch):
    captured = {}

    class Config:
        def get(self, key, default=None):
            if key == "storage.s3":
                return {
                    "enabled": True,
                    "save_original": True,
                    "endpoint": "http://s3.local",
                    "access_key": "access",
                    "secret_key": "secret",
                    "bucket": "audio-bucket",
                    "region": "test-region",
                    "prefix": "private-archive",
                }
            return default

        def get_env(self, _key, _env_name, default=None):
            return default

    def fake_store(*args):
        captured["args"] = args
        return SimpleNamespace(name="s3")

    monkeypatch.setattr(infrastructure, "S3AudioBackupStore", fake_store)

    store = infrastructure.build_audio_backup_store(Config())

    assert store.name == "s3"
    assert captured["args"] == (
        "http://s3.local",
        "access",
        "secret",
        "audio-bucket",
        "test-region",
        "private-archive",
    )
