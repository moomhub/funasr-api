"""Configuration data-transfer objects."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306
    username: str = "root"
    password: str = "password"
    database: str = "funasr_tasks"
    pool_size: int = 20
    pool_recycle: int = 3600
    echo: bool = False

    @property
    def url(self) -> str:
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class DatabaseConfig:
    mysql: MySQLConfig = field(default_factory=MySQLConfig)


@dataclass
class ModelVariantConfig:
    asr: Optional[str] = None
    streaming_asr: Optional[str] = None
    final_asr: Optional[str] = None
    vad: Optional[str] = None
    punc: Optional[str] = None
    spk: Optional[str] = None


@dataclass
class ModeModelConfig:
    enabled: str = "pt"
    pt: ModelVariantConfig = field(default_factory=ModelVariantConfig)
    onnx: ModelVariantConfig = field(default_factory=ModelVariantConfig)


@dataclass
class SpeakerModeConfig:
    spk: Optional[str] = None


@dataclass
class EnginesModelsConfig:
    offline: ModeModelConfig = field(default_factory=ModeModelConfig)
    online: ModeModelConfig = field(default_factory=ModeModelConfig)
    spk: SpeakerModeConfig = field(default_factory=SpeakerModeConfig)


@dataclass
class EnginesConfig:
    enabled: List[str] = field(default_factory=lambda: ["offline"])
    device: str = "cpu"
    model_dir: str = "./damo"
    disable_model_update: bool = True
    auto_model_download: bool = True
    models: EnginesModelsConfig = field(default_factory=EnginesModelsConfig)


@dataclass
class ProcessingConfig:
    default_mode: str = "offline"
    max_concurrent_tasks: int = 4
    worker_threads: int = 4
    timeout_seconds: int = 3600
    temp_dir: str = "./temp"
    cleanup_on_complete: bool = True
    max_temp_age_hours: int = 24
    offline_async_enabled: bool = True
    offline_async_allow_immediate: bool = True
    online_queue_max_chunks: int = 32
    online_decode_interval: float = 0.48
    online_first_decode_ms: int = 600
    online_chunk_ms: int = 600
    online_chunk_size: List[int] = field(default_factory=lambda: [0, 10, 5])
    online_vad_pre_padding_ms: int = 350
    online_vad_post_padding_ms: int = 800
    online_vad_merge_gap_ms: int = 1200
    online_vad_min_final_ms: int = 2500
    online_vad_max_final_ms: int = 12000


@dataclass
class HotwordConfig:
    enabled: bool = True
    source: str = "database"
    default_ids: List[int] = field(default_factory=list)


__all__ = [name for name in globals() if name.endswith("Config")]
