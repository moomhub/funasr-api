import pytest

from src.core.config.errors import EngineConfigurationError
from src.core.config.loader import ConfigLoader
from src.engine_runtime.configuration import EngineRuntimeConfiguration


def _runtime_config(config_path):
    loader = ConfigLoader(str(config_path))
    return EngineRuntimeConfiguration(loader, loader.get_engines_config())


def test_engine_runtime_configuration_resolves_backends_and_model_names(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - offline
    - spk
  models:
    offline:
      enabled: onnx
      onnx:
        asr: custom-offline-asr
    spk:
      spk: custom-speaker
""",
        encoding="utf-8",
    )

    config = _runtime_config(config_path)

    assert config.enabled_engine_modes() == ["offline", "spk"]
    assert config.backend_for_mode("offline") == "onnx"
    assert config.backend_for_mode("spk") == "pt"
    assert config.model_name("offline", "asr", "onnx") == "custom-offline-asr"
    assert config.model_name("spk", "spk") == "custom-speaker"


def test_engine_runtime_configuration_rejects_unsupported_enabled_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - offline
    - batch
""",
        encoding="utf-8",
    )

    config = _runtime_config(config_path)

    with pytest.raises(EngineConfigurationError, match="engines.enabled"):
        config.enabled_engine_modes()


def test_engine_runtime_configuration_validates_offline_onnx_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
engines:
  enabled:
    - offline
  models:
    offline:
      enabled: onnx
      onnx_runtime:
        asr_workers: 0
""",
        encoding="utf-8",
    )

    config = _runtime_config(config_path)

    with pytest.raises(EngineConfigurationError, match="workers"):
        config.offline_onnx_options()
