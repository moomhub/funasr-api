"""Utilities for building sentence_info from timestamps or VAD segments."""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Characters stripped when counting sentence characters for timestamp alignment
_STRIP_CHARS = frozenset("。！？，、：；\"'（）【】《》 \t\r\n")


def normalize_speaker_id(value: Any) -> Any:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def build_sentence_info_from_timestamps(text: str, timestamps: List[List]) -> List[Dict[str, Any]]:
    sentence_info: List[Dict[str, Any]] = []
    sentences = re.split(r"([。！？])", text)
    complete_sentences: List[str] = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            complete_sentences.append(sentences[i] + sentences[i + 1])
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        complete_sentences.append(sentences[-1])
    word_index = 0
    for sentence in complete_sentences:
        if not sentence.strip() or word_index >= len(timestamps):
            continue
        sentence_chars = [c for c in sentence if c not in _STRIP_CHARS]
        end_index = min(word_index + len(sentence_chars), len(timestamps) - 1)
        start_ms = timestamps[word_index][1] if len(timestamps[word_index]) >= 2 else 0
        end_ms = timestamps[end_index][2] if len(timestamps[end_index]) >= 3 else timestamps[end_index][1]
        sentence_info.append({"text": sentence.strip(), "start": int(start_ms), "end": int(end_ms)})
        word_index = end_index + 1
    return sentence_info


def build_sentence_info_from_vad(text: str, vad_segments: List[List[int]]) -> List[Dict[str, Any]]:
    sentence_info: List[Dict[str, Any]] = []
    sentences = [s.strip() + "。" for s in text.split("。") if s.strip()]
    if not sentences and text:
        sentences = [text]
    for i, seg in enumerate(vad_segments):
        if i < len(sentences):
            sentence_info.append({"text": sentences[i], "start": seg[0], "end": seg[1]})
    return sentence_info
