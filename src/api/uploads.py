"""Shared upload-route helpers."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Tuple, Type

from fastapi.responses import JSONResponse

from src.application.context import AppServices
from src.application.tasks import (
    ArchiveRestoreError,
    RerecognitionConflictError,
    RerecognitionSourceNotFoundError,
    RerecognitionValidationError,
    TaskSubmissionUnavailableError,
    UploadTooLargeError,
)
from src.core.debug_logging import log_exception
from src.core.hotwords import InvalidHotwordFormatError

from .shared import is_task_submission_available


async def submit_upload_or_error(
    *,
    services: AppServices,
    mode: str,
    disabled_error: str,
    submit: Callable[[], Awaitable[object]],
    logger: logging.Logger,
    bad_request_errors: Tuple[Type[Exception], ...] = (),
):
    if not is_task_submission_available(services, mode):
        return JSONResponse(status_code=503, content={"error": disabled_error})

    try:
        return await submit()
    except TaskSubmissionUnavailableError as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Task submission availability check",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=503, content={"error": disabled_error})
    except UploadTooLargeError as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Upload size validation",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=413, content={"error": str(exc)})
    except bad_request_errors as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Upload request validation",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        log_exception(
            logger,
            logging.ERROR,
            "Upload submission",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )


async def submit_rerecognition_or_error(
    *,
    services: AppServices,
    mode: str,
    disabled_error: str,
    submit: Callable[[], Awaitable[object]],
    logger: logging.Logger,
):
    if not is_task_submission_available(services, mode):
        return JSONResponse(status_code=503, content={"error": disabled_error})

    try:
        return await submit()
    except TaskSubmissionUnavailableError as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Rerecognition availability check",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=503, content={"error": disabled_error})
    except RerecognitionSourceNotFoundError as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Rerecognition source lookup",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=404, content={"error": "来源任务不存在"})
    except (RerecognitionValidationError, InvalidHotwordFormatError) as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Rerecognition request validation",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except RerecognitionConflictError as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Rerecognition source validation",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except UploadTooLargeError as exc:
        log_exception(
            logger,
            logging.WARNING,
            "Rerecognition restored file size validation",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(status_code=413, content={"error": str(exc)})
    except ArchiveRestoreError as exc:
        log_exception(
            logger,
            logging.ERROR,
            "Rerecognition archive recovery",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(
            status_code=502,
            content={"error": "归档文件恢复失败"},
        )
    except Exception as exc:
        log_exception(
            logger,
            logging.ERROR,
            "Rerecognition submission",
            exc,
            context={"mode": mode},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )


__all__ = ["submit_rerecognition_or_error", "submit_upload_or_error"]
