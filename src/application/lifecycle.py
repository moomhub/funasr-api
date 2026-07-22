"""FastAPI application startup and shutdown orchestration."""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import FastAPI

from src.bootstrap import build_app_services
from src.core.config.loader import ConfigLoader
from src.core.debug_logging import json_for_log, log_exception
from src.core.logging_config import configure_logging
from src.application.startup_diagnostics import run_startup_diagnostics

logger = logging.getLogger(__name__)


class ApplicationLifecycle:
    """Own startup ordering without coupling it to the HTTP entrypoint."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        *,
        config_factory: Callable[[str], Any] = ConfigLoader,
        services_factory: Callable[..., Any] = build_app_services,
    ) -> None:
        self.config_path = config_path
        self.config_factory = config_factory
        self.services_factory = services_factory

    async def start(self, app: FastAPI) -> Any:
        try:
            config = self.config_factory(self.config_path)
            configure_logging(config)
            logger.info("Application startup started")
            services = self.services_factory(config_loader=config)
            app.state.services = services

            app.state.startup_diagnostics = await run_startup_diagnostics(
                services,
                config=config,
            )
            self._log_composition(services)
            preload_summary = services.runtime_application.preload_enabled_models()
            self._log_preload(preload_summary)
            self._start_queue_if_ready(services)

            logger.info(
                "Application startup completed: enabled_modes=%s queue_running=%s",
                services.runtime_application.get_enabled_modes(),
                services.task_queue.is_running,
            )
            return services
        except Exception as exc:
            log_exception(logger, logging.ERROR, "Application startup", exc)
            raise

    async def stop(self, app: FastAPI) -> None:
        logger.info("Application shutdown started")
        services = getattr(app.state, "services", None)
        if services is None:
            logger.info("Application shutdown completed: services_initialized=false")
            return

        try:
            await services.task_queue.stop()
        except Exception as exc:
            log_exception(logger, logging.WARNING, "Task queue shutdown", exc)

        try:
            services.container.shutdown()
        except Exception as exc:
            log_exception(logger, logging.WARNING, "Service container shutdown", exc)

        logger.info("Application shutdown completed")

    @staticmethod
    def _log_composition(services: Any) -> None:
        status = services.container.get_status()
        summary = {
            name: {
                "enabled": details.get("enabled"),
                "available": details.get("available"),
            }
            for name, details in status.items()
        }
        logger.info("Application services composed: components=%s", summary)
        logger.debug("Application composition details: %s", json_for_log(status))
        logger.info(
            "Runtime configured: enabled_modes=%s backends=%s auto_download=%s",
            services.model_manager.enabled_modes,
            services.model_manager.get_inference_backends(),
            services.model_manager.auto_download,
        )

    @staticmethod
    def _log_preload(summary: dict) -> None:
        loaded = list(summary.get("loaded", []))
        failed = dict(summary.get("failed", {}))
        logger.info(
            "Runtime preload completed: loaded_count=%s failed_count=%s loaded_modes=%s",
            len(loaded),
            len(failed),
            loaded,
        )
        if failed:
            logger.warning("Runtime preload unavailable modes: modes=%s", list(failed))
        logger.debug("Runtime preload details: %s", json_for_log(summary))

    @staticmethod
    def _start_queue_if_ready(services: Any) -> None:
        queue = services.task_queue
        ready_modes = [
            mode
            for mode in ("offline", "spk")
            if (
                services.runtime_application.is_mode_available(mode)
                and queue.supports(mode)
            )
        ]
        if not queue.enabled:
            logger.info("Task queue startup skipped: configured=false")
            return
        if not ready_modes:
            logger.warning("Task queue startup skipped: no_available_task_runtime=true")
            return

        queue.start()
        logger.info("Task queue startup completed: accepted_modes=%s", ready_modes)


__all__ = ["ApplicationLifecycle"]
