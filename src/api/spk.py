"""Standalone speaker recognition routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.application.context import AppServices
from src.core.debug_logging import log_exception

from .dependencies import get_app_services
from .uploads import submit_rerecognition_or_error, submit_upload_or_error

logger = logging.getLogger(__name__)

router = APIRouter()


class SpkRerecognitionRequest(BaseModel):
    email: Optional[str] = None
    vip: Optional[bool] = None


@router.post("/spk/recognize")
async def recognize_speaker(
    file: UploadFile = File(...),
    email: Optional[str] = Form(None, description="用户邮箱（非必填）"),
    vip: bool = Form(False, description="是否 VIP 优先任务"),
    services: AppServices = Depends(get_app_services),
):
    """SPK 独立说话人识别/分离端点。"""
    return await submit_upload_or_error(
        services=services,
        mode="spk",
        disabled_error="SPK 模式未启用",
        submit=lambda: services.task_submission_service.submit_speaker(
            file,
            email=email,
            vip=vip,
        ),
        logger=logger,
    )


@router.post("/spk/tasks/{task_id}/rerecognize")
async def rerecognize_speaker_task(
    task_id: str,
    request: Optional[SpkRerecognitionRequest] = None,
    services: AppServices = Depends(get_app_services),
):
    """Restore archived audio and submit a new standalone SPK task."""
    payload = request or SpkRerecognitionRequest()
    return await submit_rerecognition_or_error(
        services=services,
        mode="spk",
        disabled_error="SPK 模式未启用或任务队列不可用",
        submit=lambda: services.task_submission_service.rerecognize_speaker(
            task_id,
            email=payload.email,
            vip=payload.vip,
        ),
        logger=logger,
    )


@router.get("/spk/tasks/{task_id}")
async def get_spk_task(
    task_id: str,
    services: AppServices = Depends(get_app_services),
):
    """查询 SPK 任务结果。"""
    if not services.is_engine_enabled("spk"):
        return JSONResponse(
            status_code=503,
            content={"error": "SPK 模式未启用"},
        )
    try:
        task = await services.task_submission_service.get_speaker_task(task_id)
        if not task:
            return JSONResponse(status_code=404, content={"error": f"SPK 任务不存在: {task_id}"})
        return jsonable_encoder(task.to_dict())
    except Exception as exc:
        log_exception(
            logger,
            logging.ERROR,
            "SPK task query",
            exc,
            context={"task_id": task_id},
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )
