from types import SimpleNamespace

from src.composition import compose_app_services


class _Config:
    def get_hotword_config(self):
        return SimpleNamespace(enabled=True, source="database")

    def get_processing_config(self):
        return SimpleNamespace(
            offline_async_enabled=True,
            offline_async_allow_immediate=True,
            max_concurrent_tasks=2,
            timeout_seconds=30,
        )

    def get_runtime_paths(self):
        return {"offline_result_dir": "/tmp/offline-results"}


class _Container:
    def __init__(self):
        self.task_repository = SimpleNamespace(name="sql")
        self.temp_file_store = SimpleNamespace(name="temp-store")
        self.audio_backup_store = SimpleNamespace(enabled=False)
        self.hotword_provider = SimpleNamespace(name="provider")
        self.speaker_task_repository = SimpleNamespace(name="speaker-task-repo")
        self.file_index_repository = SimpleNamespace(name="file-index-repo")
        self.hotword_repository = SimpleNamespace(get_by_id=lambda *_: None)


class _RuntimeFactory:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.calls = []

    def offline_asr(self):
        self.calls.append("offline")
        return SimpleNamespace(name="offline-runtime")

    def online_asr(self):
        self.calls.append("online")
        return SimpleNamespace(name="online-runtime")

    def speaker(self):
        self.calls.append("speaker")
        return SimpleNamespace(name="speaker-runtime")


class _Queue:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Service:
    def __init__(self, value):
        self.value = value


class _ResultHandler:
    @classmethod
    def from_services(cls, **kwargs):
        return SimpleNamespace(kind="result-handler", kwargs=kwargs)


class _BatchHandler:
    @classmethod
    def from_services(cls, **kwargs):
        return SimpleNamespace(kind="batch-handler", kwargs=kwargs)


def test_compose_app_services_wires_shared_container_and_queue(monkeypatch):
    import src.composition as composition

    runtime_factory = _RuntimeFactory(model_manager=object())

    monkeypatch.setattr(composition, "HotwordManager", lambda config, provider: SimpleNamespace(config=config, provider=provider))
    monkeypatch.setattr(composition, "RuntimeServiceFactory", lambda model_manager: runtime_factory)
    monkeypatch.setattr(composition, "RuntimeApplication", lambda manager, runtime_services: SimpleNamespace(manager=manager, runtime_services=runtime_services))
    monkeypatch.setattr(composition, "OnlineAsrService", lambda runtime: _Service(runtime))
    monkeypatch.setattr(composition, "SpkAsrService", lambda runtime: _Service(runtime))
    monkeypatch.setattr(composition, "OfflineRecognitionService", lambda runtime: _Service(runtime))
    monkeypatch.setattr(
        composition,
        "FilePostProcessor",
        lambda config, file_index_repository=None, audio_backup_store=None: SimpleNamespace(
            config=config,
            file_index_repository=file_index_repository,
            audio_backup_store=audio_backup_store,
        ),
    )
    monkeypatch.setattr(composition, "OfflineTaskResultHandler", _ResultHandler)
    monkeypatch.setattr(composition, "OfflineBatchResultHandler", _BatchHandler)
    monkeypatch.setattr(composition, "OfflineTaskQueue", _Queue)
    monkeypatch.setattr(composition, "TaskSubmissionService", lambda **kwargs: SimpleNamespace(**kwargs))

    config = _Config()
    container = _Container()
    model_manager = SimpleNamespace(enabled_modes=["offline", "online", "spk"])

    services = compose_app_services(config=config, container=container, model_manager=model_manager)

    assert services.config is config
    assert services.container is container
    assert services.model_manager is model_manager
    assert services.hotword_manager.provider is container.hotword_provider
    assert services.runtime_services is runtime_factory
    assert services.online_service.value.name == "online-runtime"
    assert services.speaker_service.value.name == "speaker-runtime"
    assert (
        services.task_queue.kwargs["task_service"].result_handler.kwargs[
            "postprocessor"
        ].audio_backup_store
        is container.audio_backup_store
    )
    assert (
        services.task_queue.kwargs["spk_task_service"].postprocessor.audio_backup_store
        is container.audio_backup_store
    )
    assert services.task_queue.kwargs["task_repository"] is container.task_repository
    assert services.task_queue.kwargs["speaker_task_repository"] is container.speaker_task_repository
    assert services.task_submission_service.scheduler is services.task_queue
    assert runtime_factory.calls == ["online", "speaker", "offline"]


def test_compose_offline_only_does_not_build_or_queue_standalone_spk(monkeypatch):
    import src.composition as composition

    runtime_factory = _RuntimeFactory(model_manager=object())
    monkeypatch.setattr(
        composition,
        "HotwordManager",
        lambda config, provider: SimpleNamespace(config=config, provider=provider),
    )
    monkeypatch.setattr(
        composition,
        "RuntimeServiceFactory",
        lambda model_manager: runtime_factory,
    )
    monkeypatch.setattr(
        composition,
        "RuntimeApplication",
        lambda manager, runtime_services: SimpleNamespace(
            manager=manager,
            runtime_services=runtime_services,
        ),
    )
    monkeypatch.setattr(
        composition,
        "OnlineAsrService",
        lambda _runtime: (_ for _ in ()).throw(
            AssertionError("ONLINE service must not be built")
        ),
    )
    monkeypatch.setattr(
        composition,
        "SpkAsrService",
        lambda _runtime: (_ for _ in ()).throw(
            AssertionError("standalone SPK service must not be built")
        ),
    )
    monkeypatch.setattr(
        composition,
        "OfflineRecognitionService",
        lambda runtime: _Service(runtime),
    )
    monkeypatch.setattr(
        composition,
        "FilePostProcessor",
        lambda config, **kwargs: SimpleNamespace(config=config, **kwargs),
    )
    monkeypatch.setattr(composition, "OfflineTaskResultHandler", _ResultHandler)
    monkeypatch.setattr(composition, "OfflineBatchResultHandler", _BatchHandler)
    monkeypatch.setattr(composition, "OfflineTaskQueue", _Queue)
    monkeypatch.setattr(
        composition,
        "TaskSubmissionService",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    services = compose_app_services(
        config=_Config(),
        container=_Container(),
        model_manager=SimpleNamespace(enabled_modes=["offline"]),
    )

    assert services.online_service is None
    assert services.speaker_service is None
    assert services.task_queue.kwargs["task_service"] is not None
    assert services.task_queue.kwargs["spk_task_service"] is None
    assert runtime_factory.calls == ["offline"]


def test_bootstrap_delegates_service_graph_to_composition(monkeypatch):
    import src.bootstrap as bootstrap

    captured = {}
    config = object()
    container = object()
    model_manager = object()
    expected_services = object()

    monkeypatch.setattr(bootstrap, "ConfigLoader", lambda config_path: config)
    monkeypatch.setattr(bootstrap, "build_container", lambda value: container)
    monkeypatch.setattr(bootstrap, "EngineModelManager", lambda config_loader: model_manager)

    def fake_compose_app_services(**kwargs):
        captured.update(kwargs)
        return expected_services

    monkeypatch.setattr(bootstrap, "compose_app_services", fake_compose_app_services)

    services = bootstrap.build_app_services("demo.yaml")

    assert services is expected_services
    assert captured == {
        "config": config,
        "container": container,
        "model_manager": model_manager,
    }
