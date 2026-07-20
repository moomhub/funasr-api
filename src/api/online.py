"""ONLINE websocket routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from src.application.context import AppServices
from src.application.online_stream import OnlineStreamController
from src.core.debug_logging import log_exception
from src.core.hotwords import InvalidHotwordFormatError, load_hotwords_with_priority

from .dependencies import get_app_services
from .shared import is_engine_enabled

logger = logging.getLogger(__name__)

router = APIRouter()


def _load_online_hotwords(services: AppServices, hotwords: Optional[str], hotword_id: Optional[int]):
    """Use the same hotword priority as offline tasks: custom text, id, defaults."""
    config = services.config
    default_hotword_ids = config.get("hotwords.default_ids", [])
    lookup_repository = services.hotword_repository
    hotword_options = {}
    if lookup_repository is not None:
        hotword_options["hotword_lookup"] = lookup_repository.get_by_id
    resolved = load_hotwords_with_priority(
        custom_hotwords=hotwords,
        hotword_id=hotword_id,
        default_hotword_ids=default_hotword_ids,
        config=config,
        **hotword_options,
    )
    if resolved:
        logger.info("🔑 ONLINE 加载热词: %s 个", len(resolved))
    return resolved or None


@router.websocket("/online/stream")
async def websocket_stream(
    websocket: WebSocket,
    hotwords: Optional[str] = Query(None, description='严格热词 JSON，如 [{"weight":80,"hotword":"达摩院"}]'),
    hotword_id: Optional[int] = Query(None, description="热词ID，用于从数据库查询热词"),
    sample_rate: int = Query(16000, description="客户端 PCM 采样率，默认 16kHz"),
    services: AppServices = Depends(get_app_services),
):
    """
    ONLINE 实时录音转文字 WebSocket。

    协议：
    - 文本 START：开始/重置会话
    - 文本 STOP：结束会话并返回最终结果
    - 二进制：16kHz mono int16 PCM 音频分片

    参数：
    - hotwords/hotword_id：与 offline 上传接口保持一致，连接建立时一次性加载。
    - sample_rate：必须与客户端发送的 PCM 采样率一致。
    """
    await websocket.accept()
    if not is_engine_enabled(services, "online"):
        await websocket.send_json({"event": "error", "error": "ONLINE 模式未启用"})
        await websocket.close(code=1013)
        return

    try:
        try:
            resolved_hotwords = _load_online_hotwords(services, hotwords, hotword_id)
        except InvalidHotwordFormatError as exc:
            await websocket.send_json({"event": "error", "error": str(exc)})
            await websocket.close(code=1008)
            return
        await OnlineStreamController(services).run(
            websocket,
            resolved_hotwords=resolved_hotwords,
            hotword_id=hotword_id,
            sample_rate=sample_rate,
        )

    except WebSocketDisconnect:
        logger.info("ONLINE WebSocket 客户端断开")
    except Exception as exc:
        log_exception(logger, logging.ERROR, "ONLINE WebSocket handling", exc)
        try:
            await websocket.send_json(
                {"event": "error", "error": "Internal Server Error"}
            )
        except Exception:
            pass

