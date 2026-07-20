import logging
from types import SimpleNamespace

from src.application.context import AppServices


def test_health_status_returns_safe_summary_and_keeps_details_in_debug(caplog):
    private_path = "C:/private/audio/results"
    private_error = "database password and internal path"
    config = SimpleNamespace(
        get_processing_config=lambda: SimpleNamespace(offline_async_enabled=True),
        get_runtime_paths=lambda: {"offline_result_dir": private_path},
    )
    container = SimpleNamespace(
        audio_backup_store=SimpleNamespace(enabled=True),
        get_status=lambda: {
            "task_repository": {
                "name": "sql",
                "type": "SqlTaskRepository",
                "enabled": True,
                "available": False,
                "last_error": private_error,
                "path": private_path,
            }
        },
    )
    runtime_application = SimpleNamespace(
        is_mode_available=lambda mode: mode == "offline",
        get_runtime_status=lambda: {
            "offline_asr": {
                "name": "offline_asr",
                "mode": "offline",
                "backend": "pt",
                "loaded": False,
                "available": False,
                "error": private_error,
                "metadata": {"model_path": private_path},
            }
        },
        get_engine_info=lambda: {
            "offline": {"backend": "pt", "enabled": True, "available": True}
        },
        get_loaded_models_count=lambda: 0,
        get_inference_backends=lambda: {"offline": "pt"},
    )
    services = AppServices(
        config=config,
        container=container,
        hotword_manager=object(),
        model_manager=object(),
        runtime_services=object(),
        runtime_application=runtime_application,
        online_service=None,
        speaker_service=None,
        task_submission_service=SimpleNamespace(can_submit=lambda _mode: False),
        task_queue=SimpleNamespace(is_running=False),
    )
    caplog.set_level(logging.DEBUG, logger="src.application.context")

    payload = services.get_health_status()

    serialized = repr(payload)
    assert "runtime_paths" not in payload
    assert private_path not in serialized
    assert private_error not in serialized
    assert payload["runtime_services"]["offline_asr"] == {
        "name": "offline_asr",
        "mode": "offline",
        "backend": "pt",
        "loaded": False,
        "available": False,
        "has_error": True,
    }
    assert payload["modules"]["task_repository"]["has_error"] is True
    assert any(private_path in record.getMessage() for record in caplog.records)
    assert any(private_error in record.getMessage() for record in caplog.records)
