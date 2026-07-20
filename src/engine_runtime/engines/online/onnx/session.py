"""ONNX-specific ONLINE realtime session."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import numpy as np

from src.engine_runtime.engines.online.realtime_session import OnlineRealtimeSession


class OnlineOnnxRealtimeSession(OnlineRealtimeSession):
    """
    ONLINE realtime session backed by funasr-onnx models.

    This keeps the WebSocket contract identical to OnlineRealtimeSession while
    avoiding the PyTorch DynamicStreamingVAD wrapper.
    """

    def __init__(
        self,
        streaming_asr_model: Any,
        vad_model: Any,
        final_model: Any,
        punc_model: Any = None,
        hotwords: Optional[List[Any]] = None,
        sample_rate: int = 16000,
        decode_interval: float = 0.48,
        first_decode_ms: int = 600,
        chunk_ms: int = 600,
        chunk_size: Optional[List[int]] = None,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        vad_pre_padding_ms: int = 350,
        vad_post_padding_ms: int = 800,
        vad_merge_gap_ms: int = 1200,
        vad_min_final_ms: int = 2500,
        vad_max_final_ms: int = 12000,
    ):
        def create_vad_adapter(model: Any) -> Any:
            create_session = getattr(model, "create_session", None)
            return create_session() if callable(create_session) else model

        super().__init__(
            streaming_asr_model=streaming_asr_model,
            vad_model=vad_model,
            final_model=final_model,
            punc_model=punc_model,
            hotwords=hotwords,
            sample_rate=sample_rate,
            decode_interval=decode_interval,
            first_decode_ms=first_decode_ms,
            chunk_ms=chunk_ms,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
            vad_pre_padding_ms=vad_pre_padding_ms,
            vad_post_padding_ms=vad_post_padding_ms,
            vad_merge_gap_ms=vad_merge_gap_ms,
            vad_min_final_ms=vad_min_final_ms,
            vad_max_final_ms=vad_max_final_ms,
            vad_adapter_factory=create_vad_adapter,
        )
        self.vad_chunk_samples = max(int(sample_rate * 100 / 1000), 1)

    async def add_audio(self, pcm_bytes: bytes) -> bool:
        """Append 16-bit PCM audio and lock any ONNX VAD-completed segments."""
        if not pcm_bytes:
            return False

        self.metrics["received_chunks"] += 1
        self.metrics["received_bytes"] += len(pcm_bytes)

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return False

        audio_float = audio_int16.astype(np.float32) / 32768.0
        self._append_audio_chunk(audio_float)

        if self._total_samples - self.vad_fed_samples < self.vad_chunk_samples:
            return await self._flush_ready_pending_segments()

        confirmed_segments: List[List[float]] = []
        while self._total_samples - self.vad_fed_samples >= self.vad_chunk_samples:
            chunk_end = self.vad_fed_samples + self.vad_chunk_samples
            vad_chunk = self._slice_audio(self.vad_fed_samples, chunk_end)
            chunk_segments = await asyncio.to_thread(
                self.vad.feed,
                vad_chunk,
                False,
            )
            confirmed_segments.extend(chunk_segments)
            self.vad_fed_samples = chunk_end

        return await self._lock_confirmed_segments(confirmed_segments)

    async def finish(self) -> None:
        """Flush ONNX VAD and lock the remaining speech as final text."""
        locked_count_before_finish = len(self.locked_sentences)
        remaining = self._slice_audio(self.vad_fed_samples, self._total_samples)
        confirmed_segments: List[List[float]] = []
        if remaining.size > 0:
            confirmed_segments.extend(await asyncio.to_thread(
                self.vad.feed,
                remaining,
                True,
            ))
            self.vad_fed_samples = self._total_samples

        confirmed_segments.extend(await asyncio.to_thread(
            self.vad.feed,
            np.array([], dtype=np.float32),
            True,
        ))
        await self._lock_confirmed_segments(confirmed_segments)

        if getattr(self.vad, "current_speech_start", None) is not None:
            end_ms = int(self._total_samples * 1000 / self.sample_rate)
            await self._lock_confirmed_segments([[self.vad.current_speech_start, end_ms]])
            self.vad.current_speech_start = None

        await self._flush_pending_final_segments(force=True)

        if len(self.locked_sentences) == locked_count_before_finish:
            await self._force_lock_tail()

        self.partial = ""
        self.partial_start_ms = 0
