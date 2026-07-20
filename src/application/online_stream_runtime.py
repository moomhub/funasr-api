"""Concurrent runtime for one ONLINE recognition stream."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Callable, Optional

from src.application.online_events import (
    backpressure_event,
    error_event,
    inactive_session_event,
    started_event,
    stopped_event,
)
from src.core.debug_logging import json_for_log, log_exception

logger = logging.getLogger(__name__)


class OnlineStreamRuntime:
    """Own audio queue, recognition worker, and session state transitions."""

    def __init__(
        self,
        websocket: Any,
        session: Any,
        *,
        queue_max_chunks: int,
        sample_rate: int,
        backpressure_interval: float = 0.5,
        clock: Callable[[], float] = time.time,
    ):
        self.websocket = websocket
        self.session = session
        self.queue_max_chunks = max(1, int(queue_max_chunks))
        self.sample_rate = sample_rate
        self.backpressure_interval = backpressure_interval
        self.clock = clock

        self.audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(
            maxsize=self.queue_max_chunks
        )
        self._send_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._stop_worker = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None
        self._generation = 0
        self._last_backpressure_notice = 0.0

    def start_worker(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._stop_worker.clear()
        self._worker_task = asyncio.create_task(
            self._recognition_worker(),
            name="online-recognition-worker",
        )

    async def close(self) -> None:
        self._generation += 1
        self._stop_worker.set()
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._worker_task
        self._worker_task = None

    async def send_json(self, payload: dict) -> None:
        logger.debug("ONLINE websocket output: %s", json_for_log(payload))
        async with self._send_lock:
            await self.websocket.send_json(payload)

    async def start_session(self) -> None:
        self._generation += 1
        generation = self._generation
        self._drain_audio_queue()

        async with self._session_lock:
            if generation != self._generation:
                return
            self.session.reset()
            self.session.is_active = True

        logger.debug("ONLINE session started: sample_rate=%s", self.sample_rate)
        await self.send_json(started_event())

    async def stop_session(self) -> None:
        await self.audio_queue.join()
        final_result = None

        async with self._session_lock:
            if self.session.is_active:
                await self.session.finish()
                candidate = self.session.build_offline_event(is_final=True)
                if candidate["sentences"] or candidate["text"]:
                    final_result = candidate
                self.session.is_active = False
            metrics = self.session.get_metrics()

        if final_result is not None:
            await self.send_json(final_result)
        logger.debug("ONLINE session stopped: metrics=%s", metrics)
        await self.send_json(stopped_event(metrics))

    async def enqueue_audio(self, data: bytes) -> None:
        if not self.session.is_active:
            await self.send_json(inactive_session_event())
            return

        try:
            self.audio_queue.put_nowait((self._generation, data))
            queue_size = self.audio_queue.qsize()
            self.session.note_queue_depth(queue_size)
            logger.debug(
                "ONLINE audio queued: bytes=%s queue_size=%s",
                len(data),
                queue_size,
            )
        except asyncio.QueueFull:
            await self._handle_backpressure()

    async def _recognition_worker(self) -> None:
        while not self._stop_worker.is_set():
            try:
                generation, data = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=0.2,
                )
            except asyncio.TimeoutError:
                continue

            try:
                payload = await self._process_audio(generation, data)
                if payload is not None:
                    await self._send_if_current(generation, payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_exception(
                    logger,
                    logging.ERROR,
                    "ONLINE recognition worker",
                    exc,
                    context={"generation": generation, "audio_bytes": len(data)},
                )
                await self.send_json(error_event("Internal Server Error"))
            finally:
                self.audio_queue.task_done()

    async def _process_audio(self, generation: int, data: bytes) -> Optional[dict]:
        async with self._session_lock:
            if generation != self._generation or not self.session.is_active:
                return None

            locked = await self.session.add_audio(data)
            if generation != self._generation:
                return None

            if locked:
                result = self.session.build_offline_event(is_final=False)
                return result if result["sentences"] or result["text"] else None

            now = self.clock()
            if not self.session.should_decode(now):
                return None
            result = await self.session.decode_partial()
            return result if result["text"] else None

    async def _send_if_current(self, generation: int, payload: dict) -> None:
        async with self._session_lock:
            if generation != self._generation:
                return
            await self.send_json(payload)

    async def _handle_backpressure(self) -> None:
        self.session.note_dropped_chunk()
        self.session.note_backpressure()
        now = self.clock()
        if now - self._last_backpressure_notice < self.backpressure_interval:
            return
        self._last_backpressure_notice = now
        await self.send_json(
            backpressure_event(
                queue_size=self.audio_queue.qsize(),
                queue_max_chunks=self.queue_max_chunks,
                metrics=self.session.get_metrics(),
            )
        )

    def _drain_audio_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                return


__all__ = ["OnlineStreamRuntime"]
