"""Core result types and utilities."""

from .builders import build_error_recognition_result, build_recognition_result
from .normalizers import normalize_recognition_result
from .types import RecognitionResult, Segment, SpeakerResult, SpeakerSegment

__all__ = [
    "RecognitionResult",
    "Segment",
    "SpeakerResult",
    "SpeakerSegment",
    "build_error_recognition_result",
    "build_recognition_result",
    "normalize_recognition_result",
]
