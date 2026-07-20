"""Offline ONNX recognition with funasr_onnx ASR/VAD/PUNC plus CAM++ SPK."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Any, Dict, List, Optional

from src.core.debug_logging import json_for_log, text_preview
from src.core.results import SpeakerResult
from src.core.results.builders import build_recognition_result
from src.core.results.normalizers import normalize_recognition_result
from src.core.results.types import RecognitionResult
from src.core.text import (
    build_full_text_with_speaker,
    extract_model_text,
    extract_segments_from_sentence_info,
)
from src.engine_runtime.engines.offline.base import BaseOfflineRecognizer, OfflineRecognitionRequest
from src.engine_runtime.engines.offline.sentence_builder import (
    normalize_speaker_id,
)
from src.engine_runtime.engines.offline.timestamp_utils import (
    fill_timestamp_tokens,
    offset_timestamps,
)
from src.engine_runtime.engines.offline.onnx.helpers import (
    ASRSegmentDecode,
    asr_segment_decode,
    combine_asr_segment_decodes,
    empty_asr_segment_decode,
    extract_timestamps,
    extract_vad_segments,
    slice_audio_by_ms,
)
from src.engine_runtime.engines.offline.onnx.loader import (
    ASR_HOTWORD_MODE_PLAIN,
    ASR_HOTWORD_MODE_SEACO,
    load_offline_onnx_models,
)
from src.engine_runtime.engines.offline.onnx.speaker_merge import (
    merge_onnx_speaker_result,
)

logger = logging.getLogger(__name__)


class OfflineONNXModelBundle:
    """Loaded funasr_onnx ASR/VAD/PUNC models for OFFLINE recognition."""

    def __init__(
        self,
        *,
        asr_model_dir: str,
        vad_model_dir: str,
        punc_model_dir: str,
        quantize: bool,
        num_threads: int,
        device_id: int,
        sample_rate: int = 16000,
        vad_padding_ms: int = 300,
        asr_workers: int = 2,
        load_workers: int = 4,
    ):
        self.sample_rate = int(sample_rate)
        self.vad_padding_ms = int(vad_padding_ms)
        self.asr_workers = max(1, int(asr_workers))
        self.load_workers = max(1, int(load_workers))

        loaded_models = load_offline_onnx_models(
            asr_model_dir=asr_model_dir,
            vad_model_dir=vad_model_dir,
            punc_model_dir=punc_model_dir,
            quantize=quantize,
            num_threads=num_threads,
            device_id=device_id,
            asr_workers=self.asr_workers,
            load_workers=self.load_workers,
        )
        self.vad_model = loaded_models.vad_model
        self.punc_model = loaded_models.punc_model
        self.asr_models = loaded_models.asr_models
        self.asr_hotword_mode = loaded_models.asr_hotword_mode
        logger.info(
            "OFFLINE ONNX ASR runtime loaded: hotword_mode=%s asr_workers=%s",
            self.asr_hotword_mode,
            len(self.asr_models),
        )
        self._asr_model_pool: Queue = Queue()
        for model in self.asr_models:
            self._asr_model_pool.put(model)
        self._vad_lock = threading.Lock()
        self._punc_lock = threading.Lock()

    def generate(self, audio_path: str, hotwords: Optional[List[str]] = None, **_: Any) -> Dict[str, Any]:
        audio_data = self._load_audio(audio_path)
        vad_segments = self._detect_speech_segments(audio_data)
        raw_text, timestamps, asr_segments = self._run_asr_segments(audio_data, vad_segments, hotwords)
        if raw_text:
            with self._punc_lock:
                punc_result = self.punc_model(raw_text)
        else:
            punc_result = ""
        final_text = extract_model_text(punc_result) or raw_text

        return {
            "text": final_text,
            "raw_text": raw_text,
            "vad_segments": vad_segments,
            "timestamps": timestamps,
            "asr_segments": asr_segments,
        }

    def merge_speaker_result(self, payload: Dict[str, Any], speaker: Optional[SpeakerResult]) -> Dict[str, Any]:
        return merge_onnx_speaker_result(payload, speaker)

    def _detect_speech_segments(self, audio_data: Any) -> List[List[int]]:
        with self._vad_lock:
            vad_result = self.vad_model(audio_data, kwargs=True)
        vad_segments = extract_vad_segments(vad_result)
        if vad_segments:
            return vad_segments
        duration_ms = int(len(audio_data) * 1000 / self.sample_rate)
        return [[0, duration_ms]]

    def _load_audio(self, audio_path: str):
        import librosa

        audio_data, _ = librosa.load(audio_path, sr=self.sample_rate)
        return audio_data

    def _run_asr_segments(self, audio_data, vad_segments: List[List[int]], hotwords: Optional[List[str]]):
        def decode_segment(index: int, start_ms: int, end_ms: int) -> ASRSegmentDecode:
            segment_audio, segment_offset_ms = slice_audio_by_ms(
                audio_data,
                start_ms=start_ms,
                end_ms=end_ms,
                padding_ms=self.vad_padding_ms,
                sample_rate=self.sample_rate,
            )
            if getattr(segment_audio, "size", 0) == 0:
                return empty_asr_segment_decode(index, start_ms, end_ms)
            model = self._asr_model_pool.get()
            try:
                asr_result = self._invoke_asr_model(model, segment_audio, hotwords)
                segment_text = extract_model_text(asr_result)
                segment_timestamps = extract_timestamps(asr_result) or []
                shifted_timestamps = offset_timestamps(segment_timestamps, segment_offset_ms)
                filled_timestamps = fill_timestamp_tokens(segment_text, shifted_timestamps)
                return asr_segment_decode(
                    index=index,
                    text=segment_text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    timestamps=filled_timestamps,
                )
            finally:
                self._asr_model_pool.put(model)

        decodes: List[ASRSegmentDecode] = []
        with ThreadPoolExecutor(max_workers=len(self.asr_models)) as executor:
            futures = [
                executor.submit(decode_segment, index, start_ms, end_ms)
                for index, (start_ms, end_ms) in enumerate(vad_segments)
            ]
            for future in as_completed(futures):
                decodes.append(future.result())

        return combine_asr_segment_decodes(decodes, len(vad_segments))

    def _invoke_asr_model(self, model: Any, segment_audio: Any, hotwords: Optional[List[str]]) -> Any:
        asr_hotword_mode = getattr(self, "asr_hotword_mode", ASR_HOTWORD_MODE_PLAIN)
        if asr_hotword_mode == ASR_HOTWORD_MODE_SEACO:
            return model(segment_audio, hotwords=self._format_seaco_hotwords(hotwords))
        try:
            return model(segment_audio, hotwords=hotwords) if hotwords else model(segment_audio)
        except TypeError:
            logger.warning("OFFLINE ONNX 当前 Paraformer 不支持 hotwords 参数，已降级为无热词识别")
            return model(segment_audio)

    @staticmethod
    def _format_seaco_hotwords(hotwords: Any) -> str:
        if not hotwords:
            return ""
        if isinstance(hotwords, str):
            return hotwords.strip()
        words = []
        for item in hotwords:
            if isinstance(item, str):
                word = item.strip()
            elif isinstance(item, dict):
                word = str(item.get("hotword") or "").strip()
            elif isinstance(item, (list, tuple)) and item:
                word = str(item[-1] or "").strip()
            else:
                word = ""
            if word:
                words.append(word)
        return " ".join(words)

    @staticmethod
    def _fill_timestamp_tokens(text: str, timestamps: List[List]) -> List[List]:
        return fill_timestamp_tokens(text, timestamps)


class OfflineONNXRecognizer(BaseOfflineRecognizer):
    """OFFLINE ONNX recognizer that composes funasr_onnx and the SPK engine."""

    backend_name = "onnx"

    def __init__(self, model_manager: Any):
        super().__init__(model_manager)

    def load_model(self) -> OfflineONNXModelBundle:
        return self.model_manager.get_offline_onnx_model_bundle()

    async def run_inference(self, model: OfflineONNXModelBundle, request: OfflineRecognitionRequest) -> Any:
        payload = await asyncio.to_thread(
            model.generate,
            request.audio_path,
            request.hotwords,
            **request.generate_kwargs,
        )
        self._log_onnx_asr_result(request.audio_path, payload)
        speaker = await self._recognize_speaker(request.audio_path, request.generate_kwargs)
        self._log_standalone_spk_result(request.audio_path, speaker)
        merged_payload = model.merge_speaker_result(payload, speaker)
        self._log_merge_result(payload, speaker, merged_payload)
        return merged_payload

    async def _recognize_speaker(self, audio_path: str, generate_kwargs: Dict[str, Any]) -> SpeakerResult:
        return await self._recognize_required_speaker(
            audio_path,
            generate_kwargs,
            source="offline_onnx",
        )

    def parse_result(self, payload: Dict[str, Any]) -> RecognitionResult:
        result = build_recognition_result("offline", is_final=True)
        sentence_info = (payload or {}).get("sentence_info", [])
        result.segments = extract_segments_from_sentence_info(
            sentence_info,
            speaker_normalizer=normalize_speaker_id,
        )
        result.full_text = build_full_text_with_speaker(result.segments) if result.segments else (payload or {}).get("text", "")
        if payload and payload.get("speaker_error"):
            result.metadata["speaker_error"] = payload["speaker_error"]
        if payload and payload.get("speaker_result"):
            result.metadata["speaker_result"] = payload["speaker_result"]
        return normalize_recognition_result(result, mode="offline", is_final=True)

    def _log_onnx_asr_result(self, audio_path: str, payload: Dict[str, Any]) -> None:
        summary = {
            "text_length": len(str((payload or {}).get("text") or "")),
            "raw_text_length": len(str((payload or {}).get("raw_text") or "")),
            "vad_segment_count": len((payload or {}).get("vad_segments") or []),
            "timestamp_count": len((payload or {}).get("timestamps") or []),
            "asr_segment_count": len((payload or {}).get("asr_segments") or []),
        }
        logger.info("OFFLINE ONNX ASR 摘要: %s", self._stringify_for_log(summary))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OFFLINE ONNX ASR 详细结果: %s", self._stringify_for_log({
                **summary,
                "audio_path": audio_path,
                "text_preview": text_preview((payload or {}).get("text")),
                "payload": payload,
            }))

    def _log_standalone_spk_result(self, audio_path: str, speaker_result: SpeakerResult) -> None:
        summary = {
            "speaker_count": speaker_result.speaker_count if isinstance(speaker_result, SpeakerResult) else 0,
            "segment_count": len(speaker_result.segments) if isinstance(speaker_result, SpeakerResult) else 0,
            "has_error": bool(
                speaker_result.error
                if isinstance(speaker_result, SpeakerResult)
                else None
            ),
        }
        logger.info("OFFLINE ONNX standalone SPK 摘要: %s", self._stringify_for_log(summary))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OFFLINE ONNX standalone SPK 详细结果: %s", self._stringify_for_log({
                **summary,
                "audio_path": audio_path,
                "speaker_ids": speaker_result.speaker_ids if isinstance(speaker_result, SpeakerResult) else [],
                "error": speaker_result.error if isinstance(speaker_result, SpeakerResult) else None,
                "metadata": speaker_result.metadata if isinstance(speaker_result, SpeakerResult) else None,
                "segments": (
                    [segment.to_dict() for segment in speaker_result.segments]
                    if isinstance(speaker_result, SpeakerResult)
                    else []
                ),
            }))

    def _log_merge_result(
        self,
        original_payload: Dict[str, Any],
        speaker_result: Optional[SpeakerResult],
        merged_payload: Dict[str, Any],
    ) -> None:
        summary = {
            "original_text_length": len(str((original_payload or {}).get("text") or "")),
            "original_vad_segment_count": len((original_payload or {}).get("vad_segments") or []),
            "original_timestamp_count": len((original_payload or {}).get("timestamps") or []),
            "original_asr_segment_count": len((original_payload or {}).get("asr_segments") or []),
            "speaker_segment_count": (
                len(speaker_result.segments)
                if isinstance(speaker_result, SpeakerResult) and speaker_result.segments
                else 0
            ),
            "has_speaker_error": bool(
                speaker_result.error
                if isinstance(speaker_result, SpeakerResult)
                else None
            ),
            "merged_sentence_count": len((merged_payload or {}).get("sentence_info") or []),
        }
        logger.info("OFFLINE ONNX ASR/SPK 合并摘要: %s", self._stringify_for_log(summary))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OFFLINE ONNX ASR/SPK 合并详细结果: %s", self._stringify_for_log({
                **summary,
                "original_text_preview": text_preview((original_payload or {}).get("text")),
                "original_payload": original_payload,
                "speaker_error": (
                    speaker_result.error
                    if isinstance(speaker_result, SpeakerResult)
                    else None
                ),
                "speaker_segments": (
                    [segment.to_dict() for segment in speaker_result.segments]
                    if isinstance(speaker_result, SpeakerResult) and speaker_result.segments
                    else []
                ),
                "merged_sentence_info": (merged_payload or {}).get("sentence_info"),
                "merged_speaker_result": (merged_payload or {}).get("speaker_result"),
            }))

    @staticmethod
    def _stringify_for_log(payload: Any) -> str:
        return json_for_log(payload)
