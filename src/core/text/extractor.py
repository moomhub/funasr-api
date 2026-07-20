"""Text extraction utilities."""

import numbers
from typing import Any

from .cleaner import clean_online_asr_text


def extract_online_text(funasr_result: Any) -> str:
    """Extract text from PT/ONNX online result payloads."""
    if funasr_result is None:
        return ""

    if isinstance(funasr_result, str):
        return clean_online_asr_text(funasr_result)

    if isinstance(funasr_result, bytes):
        return clean_online_asr_text(funasr_result.decode("utf-8", errors="ignore"))

    if isinstance(funasr_result, numbers.Number):
        return ""

    if isinstance(funasr_result, tuple):
        if len(funasr_result) == 2 and isinstance(funasr_result[0], str) and isinstance(funasr_result[1], list):
            return clean_online_asr_text(funasr_result[0])
        parts = [extract_online_text(item) for item in funasr_result]
        return clean_online_asr_text("".join(part for part in parts if part))

    if isinstance(funasr_result, dict):
        for key in ("text", "preds", "sentence", "value"):
            text = extract_online_text(funasr_result.get(key))
            if text:
                return text
        return ""

    if isinstance(funasr_result, list):
        parts = [extract_online_text(item) for item in funasr_result]
        return clean_online_asr_text("".join(part for part in parts if part))

    return clean_online_asr_text(str(funasr_result))


def extract_model_text(result: Any) -> str:
    """Extract plain text from common FunASR/funasr-onnx payload shapes."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="ignore")
    if isinstance(result, numbers.Number):
        return ""
    if isinstance(result, tuple):
        if len(result) == 2 and isinstance(result[0], str) and isinstance(result[1], list):
            return result[0]
        return "".join(part for part in (extract_model_text(item) for item in result) if part)
    if isinstance(result, dict):
        for key in ("text", "preds", "sentence", "value"):
            text = extract_model_text(result.get(key))
            if text:
                return text
        return ""
    if isinstance(result, list):
        return "".join(part for part in (extract_model_text(item) for item in result) if part)
    return str(result)
