"""Assemble the application service graph from prepared infrastructure."""

from __future__ import annotations

from typing import Any

from src.application.context import AppServices
from src.application.offline import OfflineRecognitionService, OfflineTaskService
from src.application.online import OnlineAsrService
from src.application.postprocess import FilePostProcessor
from src.application.runtime import RuntimeApplication
from src.application.speaker import SpkAsrService, SpkTaskService
from src.application.tasks import TaskSubmissionService
from src.core.hotwords.manager import HotwordManager
from src.engine_runtime.services import RuntimeServiceFactory
from src.task_queue.hook_handlers import OfflineBatchResultHandler, OfflineTaskResultHandler
from src.task_queue.queue import OfflineTaskQueue


def compose_app_services(
    *,
    config: Any,
    container: Any,
    model_manager: Any,
) -> AppServices:
    """Wire only services required by the configured public modes."""
    enabled_modes = set(model_manager.enabled_modes)
    hotword_manager = HotwordManager(config=config, provider=container.hotword_provider)
    runtime_services = RuntimeServiceFactory(model_manager)
    runtime_application = RuntimeApplication(
        manager=model_manager,
        runtime_services=runtime_services,
    )

    online_service = (
        OnlineAsrService(runtime_services.online_asr())
        if "online" in enabled_modes
        else None
    )
    speaker_service = (
        SpkAsrService(runtime_services.speaker())
        if "spk" in enabled_modes
        else None
    )

    postprocessor = None
    if enabled_modes.intersection({"offline", "spk"}):
        postprocessor = FilePostProcessor(
            config,
            file_index_repository=container.file_index_repository,
            audio_backup_store=container.audio_backup_store,
        )

    result_handler = None
    task_service = None
    if "offline" in enabled_modes:
        recognition_service = OfflineRecognitionService(runtime_services.offline_asr())
        result_handler = OfflineTaskResultHandler.from_services(
            task_repository=container.task_repository,
            postprocessor=postprocessor,
            result_dir=config.get_runtime_paths()["offline_result_dir"],
            temp_file_store=container.temp_file_store,
        )
        task_service = OfflineTaskService(
            task_repository=container.task_repository,
            temp_file_store=container.temp_file_store,
            result_handler=result_handler,
            recognition_service=recognition_service,
            config=config,
            hotword_lookup=(
                container.hotword_repository.get_by_id
                if container.hotword_repository is not None
                else None
            ),
        )

    spk_task_service = None
    if "spk" in enabled_modes:
        spk_task_service = SpkTaskService(
            speaker_task_repository=container.speaker_task_repository,
            temp_file_store=container.temp_file_store,
            speaker_service=speaker_service,
            postprocessor=postprocessor,
        )

    task_queue = OfflineTaskQueue(
        config=config,
        task_repository=container.task_repository,
        batch_result_handler=(
            OfflineBatchResultHandler.from_services()
            if task_service is not None
            else None
        ),
        task_service=task_service,
        speaker_task_repository=container.speaker_task_repository,
        spk_task_service=spk_task_service,
    )
    task_submission_service = TaskSubmissionService(
        config=config,
        temp_file_store=container.temp_file_store,
        task_repository=container.task_repository,
        speaker_task_repository=container.speaker_task_repository,
        scheduler=task_queue,
        audio_backup_store=container.audio_backup_store,
    )
    return AppServices(
        config=config,
        container=container,
        hotword_manager=hotword_manager,
        model_manager=model_manager,
        runtime_services=runtime_services,
        runtime_application=runtime_application,
        online_service=online_service,
        speaker_service=speaker_service,
        task_submission_service=task_submission_service,
        task_queue=task_queue,
    )


__all__ = ["compose_app_services"]
