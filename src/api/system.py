"""System, task, and health routes."""

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.application.context import AppServices
from src.core.debug_logging import log_exception

from .dependencies import get_app_services

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tasks/{task_id}")
async def get_task_result(task_id: str, services: AppServices = Depends(get_app_services)):
    """查询任务结果。"""
    try:
        task = await services.task_submission_service.get_offline_task(task_id)

        if not task:
            return JSONResponse(
                status_code=404,
                content={"error": f"任务不存在: {task_id}"},
            )

        result = {
            "task_id": task.id,
            "source_task_id": task.source_task_id,
            "filename": task.filename,
            "status": task.status,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "processing_time": task.processing_time,
            "error": task.error_message,
            "s3_key": task.s3_key,
            "file_hash": task.file_hash,
            "vip": task.vip,
        }

        if task.status == "completed":
            result["result"] = {
                "full_text": task.full_text,
                "segments": task.segments,
                "word_timestamps": task.word_timestamps,
            }

        return result

    except Exception as exc:
        log_exception(
            logger,
            logging.ERROR,
            "OFFLINE task query",
            exc,
            context={"task_id": task_id},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )


@router.get("/health")
async def health_check(services: AppServices = Depends(get_app_services)):
    """健康检查端点 - 返回系统状态和已加载的引擎信息。"""
    try:
        return services.get_health_status()
    except Exception as exc:
        log_exception(logger, logging.ERROR, "Health status query", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )
