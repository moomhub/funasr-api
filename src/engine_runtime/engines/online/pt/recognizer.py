"""PT-backed online recognizer implementation."""

import asyncio
from typing import Any

from src.engine_runtime.engines.online.base import (
    BaseOnlineRecognizer,
    OnlinePTModelBundle,
    OnlineRecognitionRequest,
    OnlineSessionRequest,
)
from src.engine_runtime.engines.online.realtime_session import OnlineRealtimeSession


class PTOnlineRecognizer(BaseOnlineRecognizer):
    backend_name = "pt"

    def load_models(self) -> OnlinePTModelBundle:
        return self.model_manager.get_online_pt_model_bundle()

    def create_session(self, request: OnlineSessionRequest) -> OnlineRealtimeSession:
        processing_config = self.model_manager.processing_config
        models = self.load_models()
        return OnlineRealtimeSession(
            streaming_asr_model=models.streaming_asr,
            vad_model=models.vad,
            final_model=models.final_asr,
            punc_model=models.punc,
            hotwords=request.hotwords,
            sample_rate=request.sample_rate,
            decode_interval=request.decode_interval,
            first_decode_ms=processing_config.online_first_decode_ms,
            chunk_ms=processing_config.online_chunk_ms,
            chunk_size=processing_config.online_chunk_size,
            vad_pre_padding_ms=processing_config.online_vad_pre_padding_ms,
            vad_post_padding_ms=processing_config.online_vad_post_padding_ms,
            vad_merge_gap_ms=request.vad_merge_gap_ms,
            vad_min_final_ms=request.vad_min_final_ms,
            vad_max_final_ms=request.vad_max_final_ms,
        )

    async def run_inference(self, models: OnlinePTModelBundle, request: OnlineRecognitionRequest) -> Any:
        return await asyncio.to_thread(
            models.streaming_asr.generate,
            input=request.audio_path,
            hotwords=request.hotwords,
            **request.generate_kwargs,
        )
