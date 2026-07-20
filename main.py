"""FunASR HTTP application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.application.lifecycle import ApplicationLifecycle
from src.core.debug_logging import log_exception
from src.core.logging_config import configure_logging, initialize_logging

initialize_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    lifecycle_manager = app.state.lifecycle_manager
    try:
        await lifecycle_manager.start(app)
        yield
    finally:
        await lifecycle_manager.stop(app)


def create_app(
    lifecycle_manager: Optional[ApplicationLifecycle] = None,
) -> FastAPI:
    """Create one FastAPI application and its isolated lifecycle manager."""
    app = FastAPI(
        title="FunASR v4.0",
        description="Multi-mode speech recognition service",
        version="4.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.lifecycle_manager = lifecycle_manager or ApplicationLifecycle()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        services = getattr(app.state, "services", None)
        modes = (
            services.runtime_application.get_enabled_modes()
            if services is not None
            else []
        )
        return {
            "name": "FunASR v4.0",
            "version": "4.0.0",
            "description": "Multi-mode speech recognition service",
            "modes": modes,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    from src.api import router

    app.include_router(router)

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        log_exception(
            logger,
            logging.ERROR,
            "Unhandled HTTP request",
            exc,
            context={"method": request.method, "path": request.url.path},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    from src.core.config.loader import ConfigLoader

    config = ConfigLoader("config.yaml")
    log_level = str(config.get("logging.level", "INFO")).lower()
    configure_logging(log_level)
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)

    logger.info("Starting Uvicorn server: host=%s port=%s", host, port)
    runtime_app = create_app(
        ApplicationLifecycle(config_factory=lambda _path: config),
    )
    uvicorn.run(
        runtime_app,
        host=host,
        port=port,
        workers=1,
        reload=False,
        log_level=log_level,
    )
