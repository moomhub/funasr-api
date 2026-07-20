from src.bootstrap import build_app_services


def _write_config(path, runtime_root):
    path.write_text(
        f"""
runtime:
  root_dir: "{runtime_root.as_posix()}"
engines:
  enabled: []
storage:
  s3:
    enabled: false
""",
        encoding="utf-8",
    )


def test_two_application_compositions_do_not_share_state(tmp_path):
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    _write_config(first_config, tmp_path / "first-runtime")
    _write_config(second_config, tmp_path / "second-runtime")

    first = build_app_services(str(first_config))
    second = build_app_services(str(second_config))
    try:
        assert first.config is not second.config
        assert first.container is not second.container
        assert first.model_manager is not second.model_manager
        assert first.runtime_services is not second.runtime_services
        assert first.task_queue is not second.task_queue
        assert first.task_repository.db is not second.task_repository.db
        assert first.task_repository.db.db_url != second.task_repository.db.db_url
    finally:
        first.container.shutdown()
        second.container.shutdown()
