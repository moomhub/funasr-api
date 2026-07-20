"""OFFLINE upload routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.application.context import AppServices
from src.core.hotwords import InvalidHotwordFormatError

from .dependencies import get_app_services
from .uploads import submit_upload_or_error

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/offline/recognize")
async def upload_offline_task(
    file: UploadFile = File(...),
    email: Optional[str] = Form(None, description="用户邮箱（非必填）"),
    hotwords: Optional[str] = Form(None, description='热词 JSON，如 [{"weight":100,"hotword":"篮子"}]（非必填）'),
    hotword_id: Optional[int] = Form(None, description="热词ID，用于从数据库查询热词（非必填）"),
    vip: bool = Form(False, description="是否 VIP 优先任务"),
    services: AppServices = Depends(get_app_services),
):
    """
    上传离线任务文件。

    流程：
    1. 同步流式写入文件（逐块读取+写入，不大块缓冲）
    2. 创建数据库记录（包含邮箱、热词等信息）
    3. 立即返回上传结果
    4. 提交独立异步任务处理音频，不等待识别完成
    """
    return await submit_upload_or_error(
        services=services,
        mode="offline",
        disabled_error="OFFLINE 模式未启用",
        submit=lambda: services.task_submission_service.submit_offline(
            file,
            email=email,
            hotwords=hotwords,
            hotword_id=hotword_id,
            vip=vip,
        ),
        logger=logger,
        bad_request_errors=(InvalidHotwordFormatError,),
    )

