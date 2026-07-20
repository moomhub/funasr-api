from sqlalchemy import inspect, text
from sqlalchemy.dialects import mysql

import pytest

from src.database.models import Base
from src.database.session import DatabaseManager, DatabaseSchemaError


def _database(tmp_path, name="tasks.db"):
    return DatabaseManager(f"sqlite:///{tmp_path / name}", pool_size=1)


def test_empty_database_creates_current_schema(tmp_path):
    database = _database(tmp_path)
    try:
        database.init_db()
        inspector = inspect(database.engine)
        assert set(Base.metadata.tables).issubset(inspector.get_table_names())
        for table in Base.metadata.sorted_tables:
            actual = {column["name"] for column in inspector.get_columns(table.name)}
            assert {column.name for column in table.columns}.issubset(actual)
    finally:
        database.close()


def test_task_handle_status_uses_mysql_tinyint_with_default_two(tmp_path):
    database = _database(tmp_path)
    try:
        database.init_db()
        inspector = inspect(database.engine)

        for table_name in ("offline_tasks", "spk_tasks"):
            columns = {
                column["name"]: column
                for column in inspector.get_columns(table_name)
            }
            handle_status = columns["handle_status"]
            assert str(handle_status["type"]).upper() == "SMALLINT"
            assert str(handle_status["default"]).strip("()'\"") == "2"

            model_column = Base.metadata.tables[table_name].c.handle_status
            assert model_column.type.compile(dialect=mysql.dialect()) == "TINYINT"
    finally:
        database.close()


def test_hotword_weight_is_stored_only_in_text(tmp_path):
    database = _database(tmp_path)
    try:
        database.init_db()
        columns = {
            column["name"]
            for column in inspect(database.engine).get_columns("hotwords")
        }
        assert "text" in columns
        assert "frequency" not in columns
        assert "created_by" not in columns
        assert "updated_by" not in columns
    finally:
        database.close()


def test_non_hotword_tables_support_soft_delete(tmp_path):
    database = _database(tmp_path)
    try:
        database.init_db()
        inspector = inspect(database.engine)

        for table_name in ("offline_tasks", "spk_tasks", "s3_files"):
            columns = {
                column["name"]: column
                for column in inspector.get_columns(table_name)
            }
            is_deleted = columns["is_deleted"]
            assert is_deleted["nullable"] is False
            assert str(is_deleted["default"]).strip("()'\"") == "0"
    finally:
        database.close()


def test_current_schema_can_be_initialized_repeatedly(tmp_path):
    database = _database(tmp_path)
    try:
        database.init_db()
        database.init_db()
    finally:
        database.close()


def test_missing_table_is_created_without_migrating_existing_tables(tmp_path):
    database = _database(tmp_path)
    try:
        tables_without_s3 = [
            table for table in Base.metadata.sorted_tables if table.name != "s3_files"
        ]
        Base.metadata.create_all(database.engine, tables=tables_without_s3)
        assert "s3_files" not in inspect(database.engine).get_table_names()

        database.init_db()

        assert "s3_files" in inspect(database.engine).get_table_names()
    finally:
        database.close()


def test_existing_table_missing_required_column_fails_without_alter(tmp_path):
    database = _database(tmp_path)
    try:
        with database.engine.begin() as connection:
            connection.execute(text("CREATE TABLE offline_tasks (id VARCHAR(36) PRIMARY KEY)"))

        with pytest.raises(DatabaseSchemaError, match="offline_tasks.filename"):
            database.init_db()

        actual = {column["name"] for column in inspect(database.engine).get_columns("offline_tasks")}
        assert actual == {"id"}
    finally:
        database.close()


def test_extra_database_columns_are_allowed(tmp_path):
    database = _database(tmp_path)
    try:
        database.init_db()
        with database.engine.begin() as connection:
            connection.execute(text("ALTER TABLE offline_tasks ADD COLUMN deployment_note TEXT"))

        database.init_db()

        actual = {column["name"] for column in inspect(database.engine).get_columns("offline_tasks")}
        assert "deployment_note" in actual
    finally:
        database.close()
