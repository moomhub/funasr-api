"""Core result types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _normalize_json_compatible(value: Any) -> Any:
    """Convert numpy-like scalars/containers to plain Python JSON-safe values."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _normalize_json_compatible(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_compatible(item) for item in value]
    return value


@dataclass
class Segment:
    """Recognition segment."""
    text: str
    start: float
    end: float
    speaker: Any = 0
    is_final: bool = False
    confidence: float = 1.0
    timestamp: Optional[List] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': _normalize_json_compatible(self.text),
            'start': _normalize_json_compatible(self.start),
            'end': _normalize_json_compatible(self.end),
            'speaker': _normalize_json_compatible(self.speaker),
            'is_final': _normalize_json_compatible(self.is_final),
            'confidence': _normalize_json_compatible(self.confidence),
            'timestamp': _normalize_json_compatible(self.timestamp),
            'duration': _normalize_json_compatible(self.end - self.start)
        }


@dataclass
class SpeakerSegment:
    """Standalone speaker diarization segment."""

    speaker: Any
    start: float
    end: float
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": _normalize_json_compatible(self.speaker),
            "start": _normalize_json_compatible(self.start),
            "end": _normalize_json_compatible(self.end),
            "confidence": _normalize_json_compatible(self.confidence),
            "duration": _normalize_json_compatible(self.end - self.start),
        }


@dataclass
class SpeakerResult:
    """Standalone speaker diarization result."""

    segments: List[SpeakerSegment] = None
    speaker_ids: List[Any] = None
    speaker_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.segments is None:
            self.segments = []
        if self.speaker_ids is None:
            self.speaker_ids = []
        if self.metadata is None:
            self.metadata = {}
        if self.speaker_count == 0 and self.speaker_ids:
            self.speaker_count = len(self.speaker_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "speaker_ids": _normalize_json_compatible(self.speaker_ids),
            "speaker_count": _normalize_json_compatible(self.speaker_count),
            "error": _normalize_json_compatible(self.error),
            "metadata": _normalize_json_compatible(self.metadata),
        }


@dataclass
class RecognitionResult:
    """Recognition result."""
    mode: str                           # offline/online
    segments: List[Segment] = None
    speakers: List[Dict] = None         # Speaker info [{speaker_id, start, end}]
    speaker_count: int = 0
    speaker_ids: List[Any] = None
    full_text: str = ""
    processing_time: float = 0.0
    is_final: bool = False
    stage: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.segments is None:
            self.segments = []
        if self.speakers is None:
            self.speakers = []
        if self.speaker_ids is None:
            self.speaker_ids = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': _normalize_json_compatible(self.mode),
            'stage': _normalize_json_compatible(self.stage),
            'segments': [s.to_dict() for s in self.segments],
            'speakers': _normalize_json_compatible(self.speakers),
            'speaker_count': _normalize_json_compatible(self.speaker_count),
            'speaker_ids': _normalize_json_compatible(self.speaker_ids),
            'full_text': _normalize_json_compatible(self.full_text),
            'processing_time': _normalize_json_compatible(self.processing_time),
            'is_final': _normalize_json_compatible(self.is_final),
            'error': _normalize_json_compatible(self.error),
            'metadata': _normalize_json_compatible(self.metadata),
        }
