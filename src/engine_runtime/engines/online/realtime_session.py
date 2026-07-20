"""Shared ONLINE realtime session state machine."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.core.debug_logging import text_preview, log_exception
from src.core.text import extract_online_text, merge_online_partial_text
from src.core.text_utils import compact_token_text as _compact_token_text
from .audio_buffer import ChunkedAudioBuffer
from .audio_quality import has_acceptable_final_text, has_effective_speech_audio
from .events import offline_event, online_event
from .final_results import (
    build_final_result,
    build_locked_sentence,
    empty_final_result,
    final_text_from_result,
)
from .metrics import build_metrics, new_metrics, record_decode_metrics
from .session_policy import (
    audio_trim_keep_from_sample,
    effective_segment_merge_gap_ms,
    should_flush_pending_segment,
)
from .text_boundary import (
    repair_previous_boundary_overlap,
)

logger = logging.getLogger(__name__)

class OnlineRealtimeSession:
    """
    Per-WebSocket realtime recognition session.

    Flow:
    - paraformer-zh-streaming produces low-latency partial text.
    - fsmn-vad decides completed sentence boundaries.
    - offline Paraformer + ct_punc re-recognizes each completed VAD segment and
      appends it to locked sentences.
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
        vad_adapter_factory: Any = None,
    ):
        if vad_adapter_factory is None:
            from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

            vad_adapter_factory = DynamicStreamingVAD

        self.streaming_asr_model = streaming_asr_model
        self.vad = vad_adapter_factory(vad_model)
        self.final_model = final_model
        self.punc_model = punc_model
        self.hotwords = hotwords
        self.sample_rate = sample_rate
        self.decode_interval = decode_interval
        self.first_decode_samples = int(sample_rate * first_decode_ms / 1000)
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.chunk_size = chunk_size or [0, 10, 5]
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.vad_pre_padding_ms = vad_pre_padding_ms
        self.vad_post_padding_ms = vad_post_padding_ms
        self.vad_merge_gap_ms = max(0, int(vad_merge_gap_ms))
        self.vad_min_final_ms = max(1, int(vad_min_final_ms))
        self.vad_max_final_ms = max(self.vad_min_final_ms, int(vad_max_final_ms))
        self._audio = ChunkedAudioBuffer()
        self.reset()

    def reset(self) -> None:
        self._audio.reset()
        self.vad_fed_samples = 0
        self.last_stream_samples = 0
        self.last_decode_time = 0.0
        self.streaming_cache: Dict[str, Any] = {}
        self.partial = ""
        self.partial_start_ms = 0
        self.first_decode_done = False
        self.locked_sentences: List[Dict[str, Any]] = []
        self.pending_final_segments: List[Dict[str, int]] = []
        self.emitted_locked_count = 0
        self.metrics = new_metrics()
        self.is_active = False
        self.vad.reset()

    @property
    def audio_buffer(self) -> np.ndarray:
        return self._audio.slice(self._audio.start_sample, self._audio.total_samples)

    @audio_buffer.setter
    def audio_buffer(self, audio: np.ndarray) -> None:
        self._audio.replace(audio)

    @property
    def _buffer_start_sample(self) -> int:
        return self._audio.start_sample

    @property
    def _total_samples(self) -> int:
        return self._audio.total_samples

    async def add_audio(self, pcm_bytes: bytes) -> bool:
        """Append 16-bit PCM audio and lock any VAD-completed segments."""
        if not pcm_bytes:
            return False

        self.metrics["received_chunks"] += 1
        self.metrics["received_bytes"] += len(pcm_bytes)

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return False

        audio_float = audio_int16.astype(np.float32) / 32768.0
        self._append_audio_chunk(audio_float)

        new_audio = self._slice_audio(self.vad_fed_samples, self._total_samples)
        if new_audio.size == 0:
            return False

        confirmed_segments = self.vad.feed(
            torch.from_numpy(new_audio).float(),
            is_final=False,
        )
        self.vad_fed_samples = self._total_samples

        return await self._lock_confirmed_segments(confirmed_segments)

    def should_decode(self, now: float) -> bool:
        if now - self.last_decode_time < self.decode_interval:
            return False

        threshold = self.first_decode_samples if not self.first_decode_done else self.chunk_samples
        return (self._total_samples - self.last_stream_samples) >= threshold

    async def decode_partial(self) -> Dict[str, Any]:
        """Run streaming ASR on new audio and return the current response."""
        threshold = self.first_decode_samples if not self.first_decode_done else self.chunk_samples
        chunk_end = min(self.last_stream_samples + threshold, self._total_samples)
        new_audio = self._slice_audio(self.last_stream_samples, chunk_end)
        if new_audio.size < threshold:
            return self.build_online_event()

        _result, text = await self._decode_streaming_audio(new_audio, is_final=False)
        if text:
            self.partial = merge_online_partial_text(self.partial, text)
            self.first_decode_done = True

        self.last_stream_samples = chunk_end
        self.last_decode_time = datetime.now(timezone.utc).timestamp()
        return self.build_online_event()

    async def finish(self) -> None:
        """Flush VAD and lock the remaining speech as final text."""
        locked_count_before_finish = len(self.locked_sentences)
        remaining = self._slice_audio(self.vad_fed_samples, self._total_samples)
        confirmed_segments: List[List[float]] = []
        if remaining.size > 0:
            confirmed_segments.extend(self.vad.feed(
                torch.from_numpy(remaining).float(),
                is_final=True,
            ))
            self.vad_fed_samples = self._total_samples

        confirmed_segments.extend(self.vad.feed(
            torch.from_numpy(np.array([], dtype=np.float32)).float(),
            is_final=True,
        ))
        await self._lock_confirmed_segments(confirmed_segments)

        if self.vad.current_speech_start is not None:
            end_ms = int(self._total_samples * 1000 / self.sample_rate)
            await self._lock_confirmed_segments([[self.vad.current_speech_start, end_ms]])
            self.vad.current_speech_start = None

        await self._flush_pending_final_segments(force=True)

        flush_text = await self._flush_streaming_tail()
        if flush_text and len(self.locked_sentences) == locked_count_before_finish:
            duration_ms = int(len(self.audio_buffer) * 1000 / self.sample_rate)
            start_ms = min(self.partial_start_ms, duration_ms)
            self._append_locked_sentence(
                flush_text,
                start_ms,
                duration_ms,
                source="streaming_final_flush",
            )
        elif len(self.locked_sentences) == locked_count_before_finish:
            await self._force_lock_tail()

        self.partial = ""
        self.partial_start_ms = 0

    def note_queue_depth(self, depth: int) -> None:
        self.metrics["queue_high_watermark"] = max(
            self.metrics["queue_high_watermark"],
            depth,
        )

    def note_dropped_chunk(self) -> None:
        self.metrics["dropped_chunks"] += 1

    def note_backpressure(self) -> None:
        self.metrics["backpressure_events"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        return self._build_metrics(self._current_duration_ms())

    def consume_locked_sentences(self, *, flush_all: bool = False) -> List[Dict[str, Any]]:
        emit_upto = len(self.locked_sentences)
        new_sentences = list(self.locked_sentences[self.emitted_locked_count:emit_upto])
        self.emitted_locked_count = emit_upto
        return new_sentences

    def build_online_event(self) -> Dict[str, Any]:
        duration_ms = self._current_duration_ms()
        return online_event(
            partial=self.partial,
            partial_start_ms=self.partial_start_ms,
            duration_ms=duration_ms,
            metrics=self._build_metrics(duration_ms),
        )

    def build_offline_event(
        self,
        *,
        is_final: bool = False,
        sentences: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload_sentences = list(
            self.consume_locked_sentences(flush_all=is_final)
            if sentences is None
            else sentences
        )
        duration_ms = self._current_duration_ms()
        return offline_event(
            sentences=payload_sentences,
            duration_ms=duration_ms,
            is_final=is_final,
            metrics=self._build_metrics(duration_ms),
        )

    def _build_metrics(self, duration_ms: int) -> Dict[str, Any]:
        return build_metrics(self.metrics, duration_ms, self.pending_final_segments)

    async def _lock_confirmed_segments(self, segments: List[List[float]]) -> bool:
        locked_any = False
        for seg in segments or []:
            normalized = self._normalize_vad_segment(seg)
            if normalized is None:
                continue
            start_ms, end_ms = normalized
            locked_any = await self._enqueue_pending_final_segment(start_ms, end_ms) or locked_any

        locked_any = await self._flush_ready_pending_segments() or locked_any
        return locked_any

    def _normalize_vad_segment(self, segment: List[float]) -> Optional[List[int]]:
        if len(segment) != 2:
            return None
        start_ms = int(segment[0])
        end_ms = int(segment[1])
        if end_ms <= start_ms:
            return None
        return [start_ms, end_ms]

    async def _enqueue_pending_final_segment(self, start_ms: int, end_ms: int) -> bool:
        if not self.pending_final_segments:
            self.pending_final_segments.append({"start": start_ms, "end": end_ms})
            return False

        current = self.pending_final_segments[-1]
        gap_ms = start_ms - current["end"]
        if gap_ms <= self._effective_segment_merge_gap_ms():
            current["start"] = min(current["start"], start_ms)
            current["end"] = max(current["end"], end_ms)
            return False

        self.pending_final_segments.append({"start": start_ms, "end": end_ms})
        return False

    async def _flush_ready_pending_segments(self) -> bool:
        if not self.pending_final_segments:
            return False
        if not self._should_flush_pending_segment(self.pending_final_segments[0], force=False):
            return False
        return await self._flush_pending_final_segments(force=False)

    async def _flush_pending_final_segments(self, force: bool = False) -> bool:
        locked_any = False
        while self.pending_final_segments:
            segment = self.pending_final_segments[0]
            if not self._should_flush_pending_segment(segment, force=force):
                break
            self.pending_final_segments.pop(0)
            locked_any = await self._finalize_pending_segment(segment) or locked_any
        return locked_any

    def _should_flush_pending_segment(self, segment: Dict[str, int], force: bool = False) -> bool:
        return should_flush_pending_segment(
            segment,
            force=force,
            current_duration_ms=self._current_duration_ms(),
            vad_max_final_ms=self.vad_max_final_ms,
            vad_min_final_ms=self.vad_min_final_ms,
            segment_merge_gap_ms=self._effective_segment_merge_gap_ms(),
            active_speech_start=getattr(self.vad, "current_speech_start", None),
        )

    def _effective_segment_merge_gap_ms(self) -> int:
        return effective_segment_merge_gap_ms(
            self.vad_merge_gap_ms,
            self.vad_post_padding_ms,
        )

    async def _finalize_pending_segment(self, segment: Dict[str, int]) -> bool:
        start_ms = segment["start"]
        end_ms = segment["end"]
        final_result = await asyncio.to_thread(self._decode_final_segment, start_ms, end_ms)
        text = final_text_from_result(final_result)

        if text:
            self._append_locked_sentence(final_result, start_ms, end_ms)
            logger.info(
                "ONLINE final segment locked: start=%sms end=%sms text_length=%s",
                start_ms,
                end_ms,
                len(text),
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "ONLINE final segment detail: start=%sms end=%sms text=%s result=%s",
                    start_ms,
                    end_ms,
                    text,
                    final_result,
                )

        end_sample = min(int(end_ms * self.sample_rate / 1000), self._total_samples)
        self.last_stream_samples = end_sample
        self.partial_start_ms = end_ms
        self.partial = ""
        self.streaming_cache = {}
        self.first_decode_done = False
        self._trim_audio_buffer()
        return bool(text)

    def _current_duration_ms(self) -> int:
        return int(self._total_samples * 1000 / self.sample_rate)

    def _decode_final_segment(self, start_ms: int, end_ms: int) -> Dict[str, Any]:
        padded_start_ms = max(0, start_ms - self.vad_pre_padding_ms)
        padded_end_ms = min(
            int(self._total_samples * 1000 / self.sample_rate),
            end_ms + self.vad_post_padding_ms,
        )
        start_sample = int(padded_start_ms * self.sample_rate / 1000)
        end_sample = min(int(padded_end_ms * self.sample_rate / 1000), self._total_samples)
        segment_audio = self._slice_audio(start_sample, end_sample)

        if segment_audio.size < int(self.sample_rate * 0.1):
            return empty_final_result(start_ms, end_ms, padded_start_ms, padded_end_ms)
        if not has_effective_speech_audio(segment_audio):
            self.metrics["rejected_final_decodes"] += 1
            return empty_final_result(start_ms, end_ms, padded_start_ms, padded_end_ms)

        try:
            start_time = time.perf_counter()
            result = self.final_model.generate(
                input=segment_audio,
                batch_size_s=60,
                hotwords=self.hotwords,
                sentence_timestamp=True,
            )
            decode_time_ms = int((time.perf_counter() - start_time) * 1000)
            record_decode_metrics(self.metrics, "final", decode_time_ms)
            raw_text = _compact_token_text(extract_online_text(result))
            final_text = self._apply_punctuation(raw_text)
            if final_text and not has_acceptable_final_text(final_text):
                self.metrics["rejected_final_decodes"] += 1
                logger.warning(
                    "ONLINE rejected suspicious final text: length=%s",
                    len(final_text),
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ONLINE rejected final payload: preview=%s result=%s",
                        text_preview(final_text),
                        result,
                    )
                return empty_final_result(start_ms, end_ms, padded_start_ms, padded_end_ms)
            return build_final_result(
                final_text=final_text,
                raw_text=raw_text,
                raw_payload=result,
                start_ms=start_ms,
                end_ms=end_ms,
                padded_start_ms=padded_start_ms,
                padded_end_ms=padded_end_ms,
            )
        except Exception as exc:
            log_exception(
                logger,
                logging.ERROR,
                "ONLINE final segment decode",
                exc,
                context={"start_ms": start_ms, "end_ms": end_ms},
            )
            return empty_final_result(start_ms, end_ms, padded_start_ms, padded_end_ms)

    def _apply_punctuation(self, text: str) -> str:
        if self.punc_model is None or not text.strip():
            return text
        try:
            result = self.punc_model.generate(input=text)
        except TypeError:
            try:
                result = self.punc_model.generate(text)
            except Exception as exc:
                self._log_punctuation_failure(exc)
                return text
        except Exception as exc:
            self._log_punctuation_failure(exc)
            return text
        return extract_online_text(result) or text

    @staticmethod
    def _log_punctuation_failure(exc: Exception) -> None:
        logger.warning(
            "ONLINE punctuation failed, fallback to ASR text: error_type=%s",
            type(exc).__name__,
        )
        logger.debug(
            "ONLINE punctuation failure details",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    async def _flush_streaming_tail(self) -> str:
        remaining = self._slice_audio(self.last_stream_samples, self._total_samples)
        if remaining.size == 0:
            return ""
        if not has_effective_speech_audio(remaining):
            self.metrics["rejected_streaming_tails"] += 1
            self.last_stream_samples = self._total_samples
            return ""

        try:
            result, text = await self._decode_streaming_audio(remaining, is_final=True)
            if text and not has_acceptable_final_text(text):
                self.metrics["rejected_streaming_tails"] += 1
                logger.warning(
                    "ONLINE rejected suspicious streaming tail: length=%s",
                    len(text),
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "ONLINE rejected streaming tail payload: preview=%s result=%s",
                        text_preview(text),
                        result,
                    )
                return ""
            self.metrics["final_flush_text"] = text
            self.last_stream_samples = self._total_samples
            return text
        except Exception as exc:
            log_exception(
                logger,
                logging.ERROR,
                "ONLINE streaming final flush",
                exc,
                context={"remaining_samples": int(remaining.size)},
            )
            return ""

    async def _decode_streaming_audio(self, audio: np.ndarray, *, is_final: bool) -> tuple[Any, str]:
        start_time = time.perf_counter()
        result = await asyncio.to_thread(
            self.streaming_asr_model.generate,
            input=audio,
            cache=self.streaming_cache,
            is_final=is_final,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back,
            hotwords=self.hotwords,
        )
        decode_time_ms = int((time.perf_counter() - start_time) * 1000)
        record_decode_metrics(self.metrics, "partial", decode_time_ms)
        return result, extract_online_text(result)

    async def _force_lock_tail(self) -> bool:
        """Fallback finalization when VAD and streaming flush fail to yield a sentence."""
        duration_ms = int(self._total_samples * 1000 / self.sample_rate)
        start_ms = min(self.partial_start_ms, duration_ms)
        if duration_ms <= start_ms:
            return False

        final_result = await asyncio.to_thread(self._decode_final_segment, start_ms, duration_ms)
        text = final_text_from_result(final_result)
        if not text:
            return False

        self._append_locked_sentence(
            final_result,
            start_ms,
            duration_ms,
            source="forced_final_tail",
        )
        logger.warning(
            "ONLINE forced final tail lock: start=%sms end=%sms text_length=%s",
            start_ms,
            duration_ms,
            len(text),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ONLINE forced final tail detail: start=%sms end=%sms text=%s result=%s",
                start_ms,
                duration_ms,
                text,
                final_result,
            )
        return True

    def _append_locked_sentence(
        self,
        final_result: Any,
        start_ms: int,
        end_ms: int,
        source: Optional[str] = None,
    ) -> bool:
        sentence = build_locked_sentence(final_result, start_ms, end_ms, source)
        if sentence is None:
            return False

        corrected_text = self._repair_previous_boundary_overlap(sentence["text"], start_ms)
        if not corrected_text:
            return False
        sentence["text"] = corrected_text
        self.locked_sentences.append(sentence)
        return True

    def _repair_previous_boundary_overlap(self, current_text: str, start_ms: int) -> str:
        current_text = current_text.strip()
        if not current_text or not self.locked_sentences:
            return current_text

        previous = self.locked_sentences[-1]
        previous_text = str(previous.get("text", ""))
        repaired_previous, current_text = repair_previous_boundary_overlap(
            previous_text,
            previous.get("end", 0),
            current_text,
            start_ms,
            self._effective_segment_merge_gap_ms(),
        )
        if repaired_previous and repaired_previous != previous_text:
            previous["text"] = repaired_previous
        return current_text

    def _append_audio_chunk(self, audio: np.ndarray) -> None:
        self._audio.append(audio)

    def _slice_audio(self, start_sample: int, end_sample: int) -> np.ndarray:
        return self._audio.slice(start_sample, end_sample)

    def _drop_audio_before(self, sample_index: int) -> None:
        self._audio.drop_before(sample_index)

    def _trim_audio_buffer(self) -> None:
        keep_from_sample = audio_trim_keep_from_sample(
            last_stream_samples=self.last_stream_samples,
            vad_fed_samples=self.vad_fed_samples,
            pending_final_segments=self.pending_final_segments,
            vad_pre_padding_ms=self.vad_pre_padding_ms,
            sample_rate=self.sample_rate,
            active_speech_start=getattr(self.vad, "current_speech_start", None),
        )
        self._drop_audio_before(keep_from_sample)


__all__ = ["OnlineRealtimeSession"]
