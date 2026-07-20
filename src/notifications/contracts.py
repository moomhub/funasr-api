"""Notification payload contracts used by task queue result hooks."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TaskMessage:
    task_id: str
    filename: str = ""
    status: str = "completed"
    processing_time: float = 0.0
    full_text: str = ""
    error_msg: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchMessage:
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_processing_time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
