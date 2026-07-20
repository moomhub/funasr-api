import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config.loader import ConfigLoader
from src.core.config.errors import EngineConfigurationError


def test_runtime_paths_are_resolved_under_root_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
runtime:
  root_dir: ./data
  models_dir: models
  sqlite_path: sqlite/funasr_tasks.db
  temp_dir: temp
  offline_result_dir: results/offline
  local_files_dir: files
""",
        encoding="utf-8",
    )

    config = ConfigLoader(str(config_path))
    paths = config.get_runtime_paths()

    assert paths["models_dir"] == str(Path("data") / "models")
    assert paths["sqlite_path"] == str(Path("data") / "sqlite" / "funasr_tasks.db")
    assert paths["temp_dir"] == str(Path("data") / "temp")
    assert paths["offline_result_dir"] == str(Path("data") / "results" / "offline")
    assert paths["local_files_dir"] == str(Path("data") / "files")


def test_runtime_path_environment_variables_take_priority(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("runtime:\n  root_dir: yaml-root\n  models_dir: yaml-models\n", encoding="utf-8")
    monkeypatch.setenv("RUNTIME_ROOT_DIR", str(tmp_path / "env-root"))
    monkeypatch.setenv("RUNTIME_MODELS_DIR", "env-models")
    monkeypatch.setenv("RUNTIME_SQLITE_PATH", "env/tasks.db")
    monkeypatch.setenv("RUNTIME_TEMP_DIR", "env-temp")
    monkeypatch.setenv("RUNTIME_OFFLINE_RESULT_DIR", "env-results")
    monkeypatch.setenv("RUNTIME_LOCAL_FILES_DIR", "env-files")

    paths = ConfigLoader(str(config_path)).get_runtime_paths()

    root = tmp_path / "env-root"
    assert paths == {
        "root_dir": str(root),
        "models_dir": str(root / "env-models"),
        "sqlite_path": str(root / "env" / "tasks.db"),
        "temp_dir": str(root / "env-temp"),
        "offline_result_dir": str(root / "env-results"),
        "local_files_dir": str(root / "env-files"),
    }


@pytest.mark.parametrize(
    ("yaml_text", "old_key", "new_key"),
    [
        ("engines:\n  model_dir: models\n", "engines.model_dir", "runtime.models_dir"),
        ("database:\n  sqlite:\n    path: old.db\n", "database.sqlite.path", "runtime.sqlite_path"),
        ("processing:\n  temp_dir: temp\n", "processing.temp_dir", "runtime.temp_dir"),
        ("results:\n  offline:\n    dir: out\n", "results.offline.dir", "runtime.offline_result_dir"),
        ("storage:\n  local:\n    dir: files\n", "storage.local.dir", "runtime.local_files_dir"),
        ("messaging:\n  enabled: false\n", "messaging", "notifications"),
        ("engines:\n  models:\n    spk:\n      pt: {}\n", "engines.models.spk.pt", "engines.models.spk.spk"),
        ("engines:\n  models:\n    spk:\n      onnx: {}\n", "engines.models.spk.onnx", "engines.models.spk.spk"),
        ("engines:\n  models:\n    spk:\n      enabled: pt\n", "engines.models.spk.enabled", "engines.models.spk.spk"),
    ],
)
def test_removed_config_keys_fail_with_explicit_mapping(tmp_path, yaml_text, old_key, new_key):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(EngineConfigurationError) as error:
        ConfigLoader(str(config_path))

    assert old_key in str(error.value)
    assert new_key in str(error.value)


@pytest.mark.parametrize(
    ("old_env", "new_env"),
    [
        ("ENGINES_MODEL_DIR", "RUNTIME_MODELS_DIR"),
        ("SQLITE_PATH", "RUNTIME_SQLITE_PATH"),
        ("PROCESSING_TEMP_DIR", "RUNTIME_TEMP_DIR"),
        ("OFFLINE_RESULT_DIR", "RUNTIME_OFFLINE_RESULT_DIR"),
        ("LOCAL_STORAGE_DIR", "RUNTIME_LOCAL_FILES_DIR"),
    ],
)
def test_removed_path_environment_variables_fail_explicitly(tmp_path, monkeypatch, old_env, new_env):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(old_env, "old-value")

    with pytest.raises(EngineConfigurationError) as error:
        ConfigLoader(str(config_path))

    assert old_env in str(error.value)
    assert new_env in str(error.value)

