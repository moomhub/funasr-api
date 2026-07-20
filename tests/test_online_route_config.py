import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import online


def test_online_hotwords_use_offline_priority(monkeypatch):
    calls = {}
    config = SimpleNamespace(
        get=lambda key, default=None: [2, 3] if key == "hotwords.default_ids" else default
    )
    services = SimpleNamespace(config=config, hotword_repository=None)

    def fake_load_hotwords_with_priority(custom_hotwords, hotword_id, default_hotword_ids, config):
        calls["custom_hotwords"] = custom_hotwords
        calls["hotword_id"] = hotword_id
        calls["default_hotword_ids"] = default_hotword_ids
        return [[80, "保险"]]

    monkeypatch.setattr(online, "load_hotwords_with_priority", fake_load_hotwords_with_priority)

    payload = '[{"weight":80,"hotword":"保险"}]'
    assert online._load_online_hotwords(services, payload, 1) == [[80, "保险"]]
    assert calls == {
        "custom_hotwords": payload,
        "hotword_id": 1,
        "default_hotword_ids": [2, 3],
    }


def test_online_websocket_params_match_database_hotword_fields():
    params = inspect.signature(online.websocket_stream).parameters

    assert "hotwords" in params
    assert "hotword_id" in params
    assert "sample_rate" in params
    assert "hotword_group_id" not in params
    assert "decode_interval" not in params

