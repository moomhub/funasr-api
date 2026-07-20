from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ArgumentError, OperationalError

import src.core.infrastructure as infrastructure_module
from src.core.config.loader import ConfigLoader
from src.database.session import DatabaseManager, DatabaseSchemaError


def _write_config(path: Path, runtime_root: Path, database_url: str) -> ConfigLoader:
    path.write_text(
        f"""
runtime:
  root_dir: "{runtime_root.as_posix()}"
  sqlite_path: fallback.db
database:
  url: "{database_url}"
""",
        encoding="utf-8",
    )
    return ConfigLoader(str(path))


def test_schema_error_never_switches_to_sqlite_fallback(tmp_path):
    primary_path = tmp_path / "legacy.db"
    database = DatabaseManager(f"sqlite:///{primary_path}", pool_size=1)
    try:
        with database.engine.begin() as connection:
            connection.execute(text("CREATE TABLE offline_tasks (id VARCHAR(36) PRIMARY KEY)"))
    finally:
        database.close()

    runtime_root = tmp_path / "runtime"
    config = _write_config(
        tmp_path / "config.yaml",
        runtime_root,
        f"sqlite:///{primary_path.as_posix()}",
    )

    with pytest.raises(DatabaseSchemaError, match="offline_tasks.filename"):
        infrastructure_module.build_task_repository(config)

    assert not (runtime_root / "fallback.db").exists()


def test_mysql_operational_error_still_uses_sqlite_fallback(tmp_path, monkeypatch):
    instances = []

    class FakeDatabaseManager:
        def __init__(self, db_url, *_args):
            self.db_url = db_url
            self.closed = False
            instances.append(self)

        def init_db(self):
            if self.db_url.startswith("mysql"):
                raise OperationalError("connect", {}, RuntimeError("database offline"))

        def close(self):
            self.closed = True

    monkeypatch.setattr(infrastructure_module, "DatabaseManager", FakeDatabaseManager)
    runtime_root = tmp_path / "runtime"
    config = _write_config(
        tmp_path / "config.yaml",
        runtime_root,
        "mysql+pymysql://user:password@127.0.0.1:3306/funasr",
    )

    repository = infrastructure_module.build_task_repository(config)

    assert len(instances) == 2
    assert instances[0].closed is True
    assert repository.db is instances[1]
    assert repository.db.db_url == f"sqlite:///{runtime_root / 'fallback.db'}"
    assert "database offline" in repository.last_error


def test_invalid_database_url_does_not_fallback(tmp_path):
    runtime_root = tmp_path / "runtime"
    config = _write_config(
        tmp_path / "config.yaml",
        runtime_root,
        "not-a-database-url",
    )

    with pytest.raises(ArgumentError):
        infrastructure_module.build_task_repository(config)

    assert not (runtime_root / "fallback.db").exists()


def test_repository_bundle_reuses_task_repository_database_manager():
    class TaskRepository:
        db = object()

    bundle = infrastructure_module.build_repository_bundle(TaskRepository())

    assert bundle.task_repository.__class__ is TaskRepository
    assert bundle.speaker_task_repository.db is bundle.task_repository.db
    assert bundle.file_index_repository.db is bundle.task_repository.db
    assert bundle.hotword_repository.db is bundle.task_repository.db
