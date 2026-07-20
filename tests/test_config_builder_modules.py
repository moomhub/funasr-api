import pytest

from src.core.config.builders import TypedConfigBuilder
from src.core.config.coercion import as_bool, as_float, as_int, as_list
from src.core.config.errors import EngineConfigurationError
from src.core.config.validation import reject_removed_configuration


def test_config_coercion_helpers_preserve_existing_value_rules():
    assert as_bool("yes") is True
    assert as_bool("OFF", True) is False
    assert as_bool("unknown", True) is True
    assert as_list("offline, online") == ["offline", "online"]
    assert as_list(None, ["offline"]) == ["offline"]
    assert as_int("12") == 12
    assert as_float("0.25") == 0.25


def test_typed_config_builder_applies_environment_overrides():
    config = {
        "database": {"mysql": {"host": "yaml-db", "port": 3306}},
        "engines": {
            "enabled": ["offline"],
            "models": {
                "offline": {"enabled": "onnx", "onnx": {"asr": "offline-asr"}},
                "spk": {"spk": "speaker-model"},
            },
        },
        "processing": {
            "online": {"chunk_size": [0, 10, 5]},
        },
        "hotwords": {"enabled": True, "default_ids": [1]},
    }
    overrides = {
        "DB_PORT": "4406",
        "ENGINES_ENABLED": "offline,spk",
        "ONLINE_CHUNK_SIZE": "1,2,3",
        "HOTWORDS_DEFAULT_IDS": "7,8",
    }

    def get_env(_key, env_var, default=None):
        return overrides.get(env_var, default)

    builder = TypedConfigBuilder(
        config,
        get_env,
        lambda: {"models_dir": "runtime/models", "temp_dir": "runtime/temp"},
    )

    database = builder.database()
    engines = builder.engines()
    processing = builder.processing()
    hotwords = builder.hotwords()

    assert database.mysql.host == "yaml-db"
    assert database.mysql.port == 4406
    assert engines.enabled == ["offline", "spk"]
    assert engines.models.offline.onnx.asr == "offline-asr"
    assert engines.models.spk.spk == "speaker-model"
    assert engines.model_dir == "runtime/models"
    assert processing.online_chunk_size == [1, 2, 3]
    assert processing.temp_dir == "runtime/temp"
    assert hotwords.default_ids == [7, 8]


def test_typed_config_builder_tolerates_non_mapping_optional_sections():
    builder = TypedConfigBuilder(
        {"database": None, "engines": {"models": None}, "processing": None},
        lambda _key, _env_var, default=None: default,
        lambda: {"models_dir": "models", "temp_dir": "temp"},
    )

    assert builder.database().mysql.host == "localhost"
    assert builder.engines().models.spk.spk is None
    assert builder.processing().temp_dir == "temp"


def test_removed_configuration_validation_is_independent_of_loader():
    with pytest.raises(EngineConfigurationError, match="engines.model_dir.*runtime.models_dir"):
        reject_removed_configuration({"engines": {"model_dir": "legacy"}}, {})
