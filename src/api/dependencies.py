"""FastAPI dependency adapters for the application service context."""

from fastapi import HTTPException
from starlette.requests import HTTPConnection

from src.application.context import AppServices


def get_app_services(connection: HTTPConnection) -> AppServices:
    services = getattr(connection.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Application services are not ready")
    return services


__all__ = ["get_app_services"]
