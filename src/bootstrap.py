"""Single composition root for application, runtime, and infrastructure services."""

from __future__ import annotations

from src.application.context import AppServices
from src.composition import compose_app_services
from src.core.config.loader import ConfigLoader
from src.core.container import build_container
from src.engine_runtime.manager import EngineModelManager


def build_app_services(
    config_path: str = "config.yaml",
    *,
    config_loader: ConfigLoader | None = None,
) -> AppServices:
    """Build one isolated object graph for a FastAPI application instance."""
    config = config_loader or ConfigLoader(config_path)
    container = build_container(config)
    model_manager = EngineModelManager(config_loader=config)
    return compose_app_services(
        config=config,
        container=container,
        model_manager=model_manager,
    )


__all__ = ["build_app_services"]
