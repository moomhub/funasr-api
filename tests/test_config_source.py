import logging
from pathlib import Path

import pytest

from src.core.config.errors import EngineConfigurationError
from src.core.config.loader import ConfigLoader


def test_invalid_yaml_fails_fast(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("engines: [", encoding="utf-8")

    with pytest.raises(EngineConfigurationError, match="配置文件加载失败"):
        ConfigLoader(str(config_path))


def test_non_mapping_yaml_root_fails_fast(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- offline\n- spk\n", encoding="utf-8")

    with pytest.raises(EngineConfigurationError, match="根节点必须是键值映射"):
        ConfigLoader(str(config_path))


def test_configuration_read_failure_fails_fast(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    original_open = Path.open

    def fail_open(self, *args, **kwargs):
        if self == config_path:
            raise OSError("private filesystem details")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(EngineConfigurationError, match="配置文件加载失败"):
        ConfigLoader(str(config_path))


def test_missing_configuration_keeps_defaults_without_info_path_leak(tmp_path, caplog):
    config_path = tmp_path / "missing-private-name.yaml"

    with caplog.at_level(logging.DEBUG, logger="src.core.config.loader"):
        loader = ConfigLoader(str(config_path))

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert loader.config_dict == {}
    assert warning_messages == ["配置文件不存在，使用默认配置"]
    assert all("missing-private-name.yaml" not in message for message in warning_messages)
    assert any("missing-private-name.yaml" in message for message in debug_messages)
