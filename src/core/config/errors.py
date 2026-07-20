"""Shared runtime exception hierarchy."""


class EngineRuntimeError(RuntimeError):
    """Base runtime error for engine orchestration."""


class EngineConfigurationError(EngineRuntimeError):
    """Raised when runtime configuration is invalid or incomplete."""


class ModelResolutionError(EngineRuntimeError):
    """Raised when a model name/path cannot be resolved from config or cache."""


class ModelLoadError(EngineRuntimeError):
    """Raised when a backend model stack cannot be loaded."""


class InferenceExecutionError(EngineRuntimeError):
    """Raised when model inference execution fails."""


class ResultParseError(EngineRuntimeError):
    """Raised when backend output cannot be normalized into the shared result shape."""


def format_runtime_error(
    *,
    mode: str,
    backend_name: str,
    exc: Exception,
    operation: str = "推理",
) -> str:
    """Convert arbitrary exceptions into the shared runtime error wording."""
    if isinstance(exc, EngineRuntimeError):
        return str(exc)

    backend = (backend_name or "engine").upper()
    return str(InferenceExecutionError(f"{mode.upper()} {backend} {operation}失败: {exc}"))
