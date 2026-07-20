import ast
from pathlib import Path

from sqlalchemy import inspect as sqlalchemy_inspect

from src.database.dao import (
    HotwordDAO as LegacyHotwordDAO,
    OfflineTaskDAO as LegacyOfflineTaskDAO,
    S3FileDAO as LegacyS3FileDAO,
    SpkTaskDAO as LegacySpkTaskDAO,
)
from src.database.daos import HotwordDAO, OfflineTaskDAO, S3FileDAO, SpkTaskDAO
from src.database.repositories import (
    SqlFileIndexRepository,
    SqlHotwordRepository,
    SqlSpeakerTaskRepository,
    SqlTaskRepository,
)
from src.database.session import DatabaseManager


ROOT = Path(__file__).resolve().parents[1]


def _database(tmp_path):
    database = DatabaseManager(f"sqlite:///{tmp_path / 'dao.db'}", pool_size=1)
    database.init_db()
    return database


def test_legacy_dao_module_reexports_entity_scoped_classes():
    assert LegacyHotwordDAO is HotwordDAO
    assert LegacyOfflineTaskDAO is OfflineTaskDAO
    assert LegacySpkTaskDAO is SpkTaskDAO
    assert LegacyS3FileDAO is S3FileDAO
    assert HotwordDAO.__module__ == "src.database.daos.hotwords"
    assert OfflineTaskDAO.__module__ == "src.database.daos.offline_tasks"
    assert SpkTaskDAO.__module__ == "src.database.daos.speaker_tasks"
    assert S3FileDAO.__module__ == "src.database.daos.file_index"


def test_legacy_dao_module_contains_no_implementation_classes():
    path = ROOT / "src" / "database" / "dao.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)


def test_sql_repositories_do_not_use_string_dao_dispatch():
    repository_dir = ROOT / "src" / "database" / "repositories"
    violations = []
    for path in repository_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id == "getattr":
                violations.append(f"{path.name}:{node.lineno}")
    assert violations == []


def test_offline_repository_preserves_task_lifecycle_and_priority(tmp_path):
    database = _database(tmp_path)
    repository = SqlTaskRepository(database)
    try:
        normal = repository.create_task("normal", "normal.wav", 10, vip=False)
        vip = repository.create_task("vip", "vip.wav", 20, vip=True)

        assert normal.id == "normal"
        assert normal.is_deleted is False
        assert vip.vip is True
        assert [task.id for task in repository.get_pending_tasks()] == ["vip", "normal"]

        processing = repository.update_status("normal", "processing")
        assert processing.status == "processing"
        assert processing.started_at is not None

        completed = repository.save_result(
            "normal",
            "recognized",
            [{"text": "recognized"}],
            0.25,
            word_timestamps=[["recognized", 0, 250]],
        )
        assert completed.status == "completed"
        assert completed.full_text == "recognized"
        assert completed.word_timestamps == [["recognized", 0, 250]]

        archived = repository.record_file_info(
            "normal",
            s3_key="audio/normal.wav",
            file_hash="hash-normal",
        )
        assert archived.s3_key == "audio/normal.wav"
        assert archived.file_hash == "hash-normal"

        repository.create_task("retry", "retry.wav", 30)
        assert repository.record_error("retry", "first").status == "pending"
        assert repository.record_error("retry", "second").status == "pending"
        failed = repository.record_error("retry", "third")
        assert failed.status == "failed"
        assert failed.retry_count == 3
        assert failed.completed_at is not None
    finally:
        database.close()


def test_speaker_repository_returns_safe_detached_results(tmp_path):
    database = _database(tmp_path)
    repository = SqlSpeakerTaskRepository(database)
    try:
        created = repository.create_task(
            task_id="spk-1",
            filename="speaker.wav",
            file_size=40,
            email="speaker@example.com",
            vip=True,
        )
        assert created.id == "spk-1"
        assert created.is_deleted is False
        assert sqlalchemy_inspect(created).detached is True

        processing = repository.update_status("spk-1", "processing")
        assert processing.started_at is not None
        assert sqlalchemy_inspect(processing).detached is True

        completed = repository.save_result(
            "spk-1",
            {
                "segments": [{"speaker_id": "speaker_0"}],
                "speaker_ids": ["speaker_0"],
                "speaker_count": 1,
            },
            0.5,
            s3_key="audio/spk-1.wav",
            file_hash="hash-spk",
        )
        assert completed.status == "completed"
        assert completed.speaker_count == 1
        assert completed.s3_key == "audio/spk-1.wav"
        assert sqlalchemy_inspect(completed).detached is True

        loaded = repository.get_task("spk-1")
        assert loaded.file_hash == "hash-spk"
        assert sqlalchemy_inspect(loaded).detached is True
    finally:
        database.close()


def test_hotword_and_file_index_repositories_use_split_daos(tmp_path):
    database = _database(tmp_path)
    hotwords = SqlHotwordRepository(database)
    files = SqlFileIndexRepository(database)
    try:
        with database.session_scope() as session:
            created_hotword = HotwordDAO.add(
                name="commands",
                text=[{"weight": 80, "hotword": "启动"}],
                session=session,
            )
            hotword_id = created_hotword.id
            disabled_hotword = HotwordDAO.add(
                name="disabled",
                text=[{"weight": 60, "hotword": "停用"}],
                session=session,
            )
            HotwordDAO.update(
                disabled_hotword.id,
                enabled=False,
                session=session,
            )
            deleted_hotword = HotwordDAO.add(
                name="deleted",
                text=[{"weight": 40, "hotword": "删除"}],
                session=session,
            )
            HotwordDAO.delete(deleted_hotword.id, session=session)
            disabled_hotword_id = disabled_hotword.id
            deleted_hotword_id = deleted_hotword.id

        assert hotwords.get_by_id(hotword_id) == [{"weight": 80, "hotword": "启动"}]
        assert hotwords.get_by_id(disabled_hotword_id) == []
        assert hotwords.get_by_id(deleted_hotword_id) == []
        assert hotwords.get_formatted_list() == [{"weight": 80, "hotword": "启动"}]

        created_file = files.create(
            task_key="task-1",
            task_type="offline",
            storage_backend="s3",
            bucket_name="audio-bucket",
            s3_key="audio/task-1.wav",
            stored_filename="task-1.wav",
            original_filename="original.wav",
            file_sha256="hash-file",
            file_size=123,
        )
        assert created_file.s3_key == "audio/task-1.wav"
        assert created_file.is_deleted is False
        assert sqlalchemy_inspect(created_file).detached is True

        loaded_file = files.get_by_hash("hash-file")
        assert loaded_file.task_key == "task-1"
        assert loaded_file.bucket_name == "audio-bucket"
        assert sqlalchemy_inspect(loaded_file).detached is True
    finally:
        database.close()
