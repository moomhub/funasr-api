"""Pure audio and text quality checks used by online finalization."""

from __future__ import annotations

from typing import Any

import numpy as np

from .text_boundary import BOUNDARY_PUNCTUATION

MIN_EFFECTIVE_AUDIO_RMS = 0.0015
MIN_EFFECTIVE_AUDIO_PEAK = 0.01
ACTIVE_SAMPLE_THRESHOLD = 0.003
MIN_ACTIVE_SAMPLE_RATIO = 0.01
FINAL_TEXT_IGNORED_CHARS = set(BOUNDARY_PUNCTUATION + " \t\r\n\"'“”‘’()（）[]【】")
CHINESE_NUMERAL_CHARS = set("零〇一二三四五六七八九十百千万亿两第")


def has_effective_speech_audio(audio: np.ndarray) -> bool:
    if audio.size == 0:
        return False
    absolute = np.abs(np.asarray(audio, dtype=np.float32).reshape(-1))
    peak = float(np.max(absolute))
    if peak < MIN_EFFECTIVE_AUDIO_PEAK:
        return False
    rms = float(np.sqrt(np.mean(np.square(absolute))))
    active_ratio = float(np.mean(absolute >= ACTIVE_SAMPLE_THRESHOLD))
    return rms >= MIN_EFFECTIVE_AUDIO_RMS or active_ratio >= MIN_ACTIVE_SAMPLE_RATIO


def has_acceptable_final_text(text: str) -> bool:
    chars = [ch for ch in text.strip() if ch not in FINAL_TEXT_IGNORED_CHARS]
    if not chars:
        return False

    if len(chars) < 12:
        return True

    max_run = 1
    current_run = 1
    for previous, current in zip(chars, chars[1:]):
        if current == previous:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    if max_run >= 4:
        return False

    char_count = len(chars)
    unique_ratio = len(set(chars)) / char_count
    if char_count >= 20 and unique_ratio < 0.2:
        return False

    counts: dict[str, int] = {}
    for char in chars:
        counts[char] = counts.get(char, 0) + 1
    if char_count >= 16 and max(counts.values()) / char_count >= 0.45:
        return False

    numeral_ratio = sum(char in CHINESE_NUMERAL_CHARS for char in chars) / char_count
    if numeral_ratio >= 0.4 and max_run >= 3:
        return False

    return True


__all__ = [
    "ACTIVE_SAMPLE_THRESHOLD",
    "CHINESE_NUMERAL_CHARS",
    "FINAL_TEXT_IGNORED_CHARS",
    "MIN_ACTIVE_SAMPLE_RATIO",
    "MIN_EFFECTIVE_AUDIO_PEAK",
    "MIN_EFFECTIVE_AUDIO_RMS",
    "has_acceptable_final_text",
    "has_effective_speech_audio",
]
