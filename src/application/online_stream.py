"""Application controller for one ONLINE WebSocket recognition stream."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.application.online_events import (
    normalize_stream_command,
    ready_event,
    unknown_command_event,
)
from src.application.online_stream_runtime import OnlineStreamRuntime

logger = logging.getLogger(__name__)


class OnlineStreamController:
    """Create the stream runtime and dispatch WebSocket protocol messages."""

    def __init__(self, services: Any):
        self.services = services

    async def run(
        self,
        websocket: Any,
        *,
        resolved_hotwords: Optional[List[Any]],
        hotword_id: Optional[int],
        sample_rate: int,
    ) -> None:
        processing_config = self.services.config.get_processing_config()
        session = self.services.online_service.create_realtime_session(
            hotwords=resolved_hotwords,
            sample_rate=sample_rate,
            decode_interval=processing_config.online_decode_interval,
            vad_merge_gap_ms=processing_config.online_vad_merge_gap_ms,
            vad_min_final_ms=processing_config.online_vad_min_final_ms,
            vad_max_final_ms=processing_config.online_vad_max_final_ms,
        )
        runtime = OnlineStreamRuntime(
            websocket,
            session,
            queue_max_chunks=processing_config.online_queue_max_chunks,
            sample_rate=sample_rate,
        )
        runtime.start_worker()

        try:
            await runtime.send_json(
                ready_event(
                    sample_rate=sample_rate,
                    decode_interval=processing_config.online_decode_interval,
                    queue_max_chunks=processing_config.online_queue_max_chunks,
                    hotword_id=hotword_id,
                    resolved_hotwords=resolved_hotwords,
                )
            )

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    logger.info("ONLINE WebSocket 客户端断开")
                    break

                text = message.get("text")
                data = message.get("bytes")
                logger.debug(
                    "ONLINE websocket input: type=%s text=%s audio_bytes=%s",
                    message.get("type"),
                    text,
                    len(data) if data is not None else 0,
                )
                if text is not None:
                    command = text.strip()
                    normalized = normalize_stream_command(command)
                    if normalized == "START":
                        await runtime.start_session()
                    elif normalized == "STOP":
                        await runtime.stop_session()
                    else:
                        await runtime.send_json(unknown_command_event(command))
                    continue

                if data is not None:
                    await runtime.enqueue_audio(data)
        finally:
            await runtime.close()


__all__ = ["OnlineStreamController"]
