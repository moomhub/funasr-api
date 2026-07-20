"""Offline PT recognition flow extracted from the engine layer."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.core.config.errors import ResultParseError
from src.core.debug_logging import json_for_log, text_preview, log_exception
from src.core.results import SpeakerResult
from src.core.results.builders import build_recognition_result
from src.core.results.normalizers import normalize_recognition_result
from src.core.results.types import RecognitionResult
from src.core.text import build_full_text_with_speaker, extract_segments_from_sentence_info
from src.engine_runtime.engines.offline.base import BaseOfflineRecognizer, OfflineRecognitionRequest
from src.engine_runtime.engines.offline.pt.speaker_merge import (
    merge_pt_sentence_info_with_speaker,
)
from src.engine_runtime.engines.offline.timestamp_utils import normalize_timestamps

logger = logging.getLogger(__name__)


class PTOfflineRecognizer(BaseOfflineRecognizer):
    """Encapsulates offline PT model loading and recognition behavior."""

    backend_name = "pt"

    def load_model(self) -> Any:
        return self.model_manager.get_offline_model()

    async def run_inference(self, model: Any, request: OfflineRecognitionRequest) -> Any:
        logger.info("🔄 调用 OFFLINE PT 模型处理音频")
        logger.debug("OFFLINE PT input: audio_path=%s hotword_count=%s generate_kwargs=%s",
                     request.audio_path, len(request.hotwords or []), self._stringify_for_log(request.generate_kwargs))
        asr_result = await asyncio.to_thread(
            model.generate,
            request.audio_path,
            batch_size_s=300,
            return_spk_res=False,
            hotword=request.hotwords,
            **request.generate_kwargs,
        )
        self._log_pt_asr_result(request.audio_path, asr_result)

        asr_payload = self._unwrap_asr_result(asr_result)
        sentence_info = self._extract_asr_sentence_info(asr_payload)
        if not self._has_usable_timestamps(sentence_info):
            logger.warning("OFFLINE PT ASR 未返回可用时间戳，跳过 SPK 并返回纯 ASR 结果")
            return {
                "asr_result": asr_result,
                "speaker_result": None,
            }

        # 中文注释：PT 主识别完成后，再单独跑一次整段音频的 SPK，
        # 后续说话人分配与重组都以 standalone SPK 的结果为准。
        speaker_result = await self._recognize_speaker(request.audio_path, request.generate_kwargs)
        return {
            "asr_result": asr_result,
            "speaker_result": speaker_result,
        }

    async def _recognize_speaker(self, audio_path: str, generate_kwargs: Dict[str, Any]) -> SpeakerResult:
        speaker_result = await self._recognize_required_speaker(
            audio_path,
            generate_kwargs,
            source="offline_pt",
        )
        self._log_standalone_spk_result(
            audio_path,
            speaker_result.metadata.get("raw_payload"),
            speaker_result,
        )
        return speaker_result

    def parse_result(self, payload: Any) -> RecognitionResult:
        result = build_recognition_result("offline", is_final=True)

        if not payload:
            logger.warning("FunASR 返回空结果")
            return normalize_recognition_result(result, mode="offline", is_final=True)

        try:
            payload = payload if isinstance(payload, dict) else {"asr_result": payload}
            asr_result = self._unwrap_asr_result(payload.get("asr_result"))
            sentence_info = self._extract_asr_sentence_info(asr_result)
            if not sentence_info:
                logger.warning("句子信息为空")
                return normalize_recognition_result(result, mode="offline", is_final=True)

            if not self._has_usable_timestamps(sentence_info):
                result.segments = extract_segments_from_sentence_info(sentence_info)
                result.full_text = str(asr_result.get("text") or "".join(
                    segment.text for segment in result.segments
                ))
                return normalize_recognition_result(result, mode="offline", is_final=True)

            speaker_result = payload.get("speaker_result")
            merged_sentence_info = merge_pt_sentence_info_with_speaker(sentence_info, speaker_result)
            self._log_merge_result(sentence_info, speaker_result, merged_sentence_info)
            result.segments = extract_segments_from_sentence_info(merged_sentence_info)
            result.full_text = build_full_text_with_speaker(result.segments)

            if isinstance(speaker_result, SpeakerResult):
                if speaker_result.error:
                    result.metadata["speaker_error"] = speaker_result.error
                elif speaker_result.segments:
                    result.metadata["speaker_result"] = speaker_result.to_dict()

            return normalize_recognition_result(result, mode="offline", is_final=True)
        except Exception as exc:
            log_exception(logger, logging.ERROR, "OFFLINE PT result parsing", exc)
            raise ResultParseError(f"OFFLINE PT 结果解析失败: {exc}") from exc

    @staticmethod
    def _extract_asr_sentence_info(asr_result: Any) -> List[Dict[str, Any]]:
        if not isinstance(asr_result, dict):
            return []

        sentence_info = asr_result.get("sentence_info") or []
        normalized = []
        for sentence in sentence_info:
            if not isinstance(sentence, dict):
                continue
            item = dict(sentence)
            if not item.get("text") and item.get("sentence"):
                item["text"] = item["sentence"]
            normalized.append(item)
        if normalized:
            return normalized

        text = str(asr_result.get("text") or "")
        if not text:
            return []
        timestamps = normalize_timestamps(asr_result.get("timestamp")) or []
        return [
            {
                "text": text,
                "start": timestamps[0][1] if timestamps else 0,
                "end": timestamps[-1][2] if timestamps else 0,
                "timestamp": timestamps,
            }
        ]

    @staticmethod
    def _has_usable_timestamps(sentence_info: List[Dict[str, Any]]) -> bool:
        return any(normalize_timestamps(sentence.get("timestamp")) for sentence in sentence_info)

    def _log_pt_asr_result(self, audio_path: str, asr_result: Any) -> None:
        asr_payload = self._unwrap_asr_result(asr_result)
        sentence_info = asr_payload.get("sentence_info", []) if isinstance(asr_payload, dict) else []
        summary = {
            "keys": sorted(asr_payload.keys()) if isinstance(asr_payload, dict) else [],
            "text_length": len(str(asr_payload.get("text") or "")) if isinstance(asr_payload, dict) else 0,
            "sentence_count": len(sentence_info),
            "timestamp_count": len(asr_payload.get("timestamp") or []) if isinstance(asr_payload, dict) else 0,
            "vad": self._extract_vad_related(asr_payload),
        }
        logger.info("OFFLINE PT ASR 摘要: %s", self._stringify_for_log(summary))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OFFLINE PT ASR 详细结果: %s", self._stringify_for_log({
                **summary,
                "audio_path": audio_path,
                "text_preview": text_preview(asr_payload.get("text") if isinstance(asr_payload, dict) else None),
                "sentence_info": sentence_info,
                "timestamp": asr_payload.get("timestamp") if isinstance(asr_payload, dict) else None,
                "raw_result": asr_result,
            }))

    def _log_standalone_spk_result(
        self,
        audio_path: str,
        raw_payload: Any,
        speaker_result: SpeakerResult,
    ) -> None:
        summary = {
            "speaker_count": speaker_result.speaker_count,
            "segment_count": len(speaker_result.segments),
            "has_error": bool(speaker_result.error),
        }
        logger.info("OFFLINE PT standalone SPK 摘要: %s", self._stringify_for_log(summary))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OFFLINE PT standalone SPK 详细结果: %s", self._stringify_for_log({
                **summary,
                "audio_path": audio_path,
                "speaker_ids": speaker_result.speaker_ids,
                "error": speaker_result.error,
                "segments": [segment.to_dict() for segment in speaker_result.segments],
                "raw_payload": raw_payload,
            }))

    def _log_merge_result(
        self,
        original_sentence_info: List[Dict[str, Any]],
        speaker_result: Optional[SpeakerResult],
        merged_sentence_info: List[Dict[str, Any]],
    ) -> None:
        summary = {
            "original_sentence_count": len(original_sentence_info or []),
            "speaker_segment_count": (
                len(speaker_result.segments)
                if isinstance(speaker_result, SpeakerResult)
                else 0
            ),
            "merged_sentence_count": len(merged_sentence_info or []),
        }
        logger.info("OFFLINE PT ASR/SPK 合并摘要: %s", self._stringify_for_log(summary))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OFFLINE PT ASR/SPK 合并详细结果: %s", self._stringify_for_log({
                **summary,
                "original_sentence_info": original_sentence_info,
                "speaker_segments": (
                    [segment.to_dict() for segment in speaker_result.segments]
                    if isinstance(speaker_result, SpeakerResult) and speaker_result.segments
                    else []
                ),
                "merged_sentence_info": merged_sentence_info,
            }))

    @staticmethod
    def _unwrap_asr_result(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload or {}

    @staticmethod
    def _extract_vad_related(asr_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(asr_payload, dict):
            return {}
        return {
            key: value
            for key, value in asr_payload.items()
            if "vad" in str(key).lower()
        }

    @staticmethod
    def _stringify_for_log(payload: Any) -> str:
        return json_for_log(payload)
