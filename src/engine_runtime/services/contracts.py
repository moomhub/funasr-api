"""Stable service contracts used by runtime workflows.

Workflow code should depend on these protocols instead of concrete PT/ONNX
recognizers. Concrete services own model loading and framework-specific calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.core.results import RecognitionResult, SpeakerResult


@dataclass
class ServiceHealth:
    """Current service readiness details."""

    name: str
    mode: str
    backend: str
    loaded: bool = False
    available: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "backend": self.backend,
            "loaded": self.loaded,
            "available": self.available,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ModelPreloadStatus:
    """Result of a preload attempt."""

    service_name: str
    mode: str
    backend: str
    loaded: bool
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "mode": self.mode,
            "backend": self.backend,
            "loaded": self.loaded,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class OfflineAsrRequest:
    audio_path: str
    hotwords: Optional[List[str]] = None
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeakerRequest:
    audio_path: str
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PreloadableService(Protocol):
    name: str
    mode: str
    backend: str

    @property
    def is_loaded(self) -> bool:
        ...

    def preload(self) -> ModelPreloadStatus:
        ...

    def health(self) -> ServiceHealth:
        ...


@runtime_checkable
class OfflineAsrService(PreloadableService, Protocol):
    async def recognize(self, request: OfflineAsrRequest) -> RecognitionResult:
        ...


@runtime_checkable
class OnlineAsrService(PreloadableService, Protocol):
    def create_realtime_session(self, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class SpeakerService(PreloadableService, Protocol):
    async def diarize(self, request: SpeakerRequest) -> SpeakerResult:
        ...


@runtime_checkable
class ResultMergeService(Protocol):
    name: str

    async def merge(
        self,
        recognition: RecognitionResult,
        speaker: Optional[SpeakerResult] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RecognitionResult:
        ...
