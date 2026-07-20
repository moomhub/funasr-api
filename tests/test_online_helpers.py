from __future__ import annotations

import numpy as np

from src.engine_runtime.engines.online.audio_quality import (
    has_acceptable_final_text,
    has_effective_speech_audio,
)
from src.engine_runtime.engines.online.final_results import (
    build_final_result,
    build_locked_sentence,
    empty_final_result,
    final_text_from_result,
)
from src.engine_runtime.engines.online.metrics import new_metrics, record_decode_metrics
from src.engine_runtime.engines.online.session_policy import (
    audio_trim_keep_from_sample,
    effective_segment_merge_gap_ms,
    required_tail_gap_ms,
    should_flush_pending_segment,
)
from src.engine_runtime.engines.online.text_boundary import (
    repair_previous_boundary_overlap,
)
from src.engine_runtime.engines.online.timestamp_utils import (
    extract_global_timestamps,
    extract_timestamp_payload,
)


def test_extract_timestamp_payload_walks_nested_model_results():
    payload = {
        "result": [
            {
                "value": {
                    "timestamp": [["你", 10, 90], ["好", 90, 180]],
                }
            }
        ]
    }

    assert extract_timestamp_payload(payload) == [["你", 10, 90], ["好", 90, 180]]


def test_extract_global_timestamps_offsets_local_model_timestamps():
    payload = {"timestamp": [["你", 100, 180], ["好", 180, 280]]}

    assert extract_global_timestamps(
        payload,
        raw_text="你好",
        start_ms=1000,
        end_ms=1200,
        padded_start_ms=900,
        padded_end_ms=1300,
    ) == [["你", 1000, 1080], ["好", 1080, 1180]]


def test_has_effective_speech_audio_rejects_silence_and_accepts_signal():
    assert has_effective_speech_audio(np.zeros(1600, dtype=np.float32)) is False
    assert has_effective_speech_audio(np.full(1600, 0.02, dtype=np.float32)) is True


def test_has_acceptable_final_text_rejects_repeated_model_hallucination():
    assert has_acceptable_final_text("啊啊啊啊啊啊啊啊啊啊啊啊") is False
    assert has_acceptable_final_text("这是一个正常的识别结果") is True


def test_repair_previous_boundary_overlap_only_changes_nearby_previous_text():
    repaired_previous, current = repair_previous_boundary_overlap(
        previous_text="你好你好。",
        previous_end_ms=1000,
        current_text="你好",
        current_start_ms=1200,
        merge_gap_ms=500,
    )

    assert repaired_previous == "你好。"
    assert current == "你好"


def test_repair_previous_boundary_overlap_does_not_change_distant_text():
    repaired_previous, current = repair_previous_boundary_overlap(
        previous_text="你好你好。",
        previous_end_ms=1000,
        current_text="你好",
        current_start_ms=2000,
        merge_gap_ms=500,
    )

    assert repaired_previous == "你好你好。"
    assert current == "你好"


def test_final_result_helpers_build_timestamps_and_locked_sentence_payload():
    assert final_text_from_result({"final_text": "你好"}) == "你好"
    assert final_text_from_result(None) == ""
    assert empty_final_result(10, 20, 0, 30)["timestamp"] == []

    result = build_final_result(
        final_text="你好",
        raw_text="你好",
        raw_payload={"timestamp": [["你", 100, 180], ["好", 180, 280]]},
        start_ms=1000,
        end_ms=1300,
        padded_start_ms=900,
        padded_end_ms=1300,
    )
    assert result["timestamp"] == [["你", 1000, 1080], ["好", 1080, 1180]]
    assert result["tokens"][0]["text"] == "你"

    sentence = build_locked_sentence(result, 1000, 1300, "forced_final_tail")
    assert sentence["text"] == "你好"
    assert sentence["source"] == "forced_final_tail"
    assert sentence["raw_text"] == "你好"


def test_record_decode_metrics_updates_count_totals_and_peaks():
    metrics = new_metrics()

    record_decode_metrics(metrics, "partial", 12)
    record_decode_metrics(metrics, "partial", 20)
    record_decode_metrics(metrics, "final", 35)

    assert metrics["partial_decodes"] == 2
    assert metrics["partial_decode_time_ms"] == 32
    assert metrics["last_partial_decode_time_ms"] == 20
    assert metrics["max_partial_decode_time_ms"] == 20
    assert metrics["final_decodes"] == 1
    assert metrics["final_decode_time_ms"] == 35
    assert metrics["last_final_decode_time_ms"] == 35
    assert metrics["max_final_decode_time_ms"] == 35


def test_pending_segment_policy_delays_short_segments_until_tail_gap_is_large_enough():
    merge_gap_ms = effective_segment_merge_gap_ms(1200, 800)

    assert required_tail_gap_ms(600, vad_min_final_ms=2500, segment_merge_gap_ms=merge_gap_ms) == 2000
    assert should_flush_pending_segment(
        {"start": 0, "end": 600},
        current_duration_ms=1800,
        vad_max_final_ms=12000,
        vad_min_final_ms=2500,
        segment_merge_gap_ms=merge_gap_ms,
    ) is False
    assert should_flush_pending_segment(
        {"start": 0, "end": 600},
        current_duration_ms=2800,
        vad_max_final_ms=12000,
        vad_min_final_ms=2500,
        segment_merge_gap_ms=merge_gap_ms,
    ) is True


def test_pending_segment_policy_force_and_active_speech_rules():
    assert should_flush_pending_segment(
        {"start": 1000, "end": 1200},
        force=True,
        current_duration_ms=1200,
        vad_max_final_ms=12000,
        vad_min_final_ms=2500,
        segment_merge_gap_ms=2000,
    ) is True
    assert should_flush_pending_segment(
        {"start": 0, "end": 13000},
        current_duration_ms=13000,
        vad_max_final_ms=12000,
        vad_min_final_ms=2500,
        segment_merge_gap_ms=2000,
        active_speech_start=13050,
    ) is True
    assert should_flush_pending_segment(
        {"start": 0, "end": 3000},
        current_duration_ms=6000,
        vad_max_final_ms=12000,
        vad_min_final_ms=2500,
        segment_merge_gap_ms=2000,
        active_speech_start=4500,
    ) is False


def test_audio_trim_policy_keeps_samples_needed_by_pending_and_active_speech():
    assert audio_trim_keep_from_sample(
        last_stream_samples=64000,
        vad_fed_samples=80000,
        pending_final_segments=[{"start": 3000, "end": 5000}],
        vad_pre_padding_ms=500,
        sample_rate=16000,
    ) == 40000

    assert audio_trim_keep_from_sample(
        last_stream_samples=64000,
        vad_fed_samples=80000,
        pending_final_segments=[],
        vad_pre_padding_ms=500,
        sample_rate=16000,
        active_speech_start=2000,
    ) == 24000
