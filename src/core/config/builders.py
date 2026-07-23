"""Builders for typed configuration data-transfer objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Dict

from .coercion import as_bool, as_float, as_int, as_list
from .schema import (
    DatabaseConfig,
    EnginesConfig,
    EnginesModelsConfig,
    HotwordConfig,
    LocalStorageConfig,
    ModeModelConfig,
    ModelVariantConfig,
    MySQLConfig,
    ProcessingConfig,
    S3StorageConfig,
    SpeakerModeConfig,
    SQLiteConfig,
    StorageConfig,
)


class TypedConfigBuilder:
    """Build schema objects from raw config without owning source loading."""

    def __init__(
        self,
        config: Mapping[str, Any],
        get_env: Callable[..., Any],
        get_runtime_paths: Callable[[], Dict[str, str]],
    ):
        self.config = config
        self.get_env = get_env
        self.get_runtime_paths = get_runtime_paths

    def database(self) -> DatabaseConfig:
        database_cfg = self._section("database")
        sqlite_cfg = self._section("database", "sqlite")
        mysql_cfg = self._section("database", "mysql")
        database_type = str(
            self._env(
                "database.type",
                "DATABASE_TYPE",
                database_cfg.get("type", "sqlite"),
            )
            or "sqlite"
        ).strip().lower()
        runtime_paths = self.get_runtime_paths()
        sqlite = SQLiteConfig(
            path=runtime_paths["sqlite_path"],
            pool_size=as_int(
                self._env(
                    "database.sqlite.pool_size",
                    "SQLITE_POOL_SIZE",
                    sqlite_cfg.get("pool_size", 5),
                )
            ),
            pool_recycle=as_int(
                self._env(
                    "database.sqlite.pool_recycle",
                    "SQLITE_POOL_RECYCLE",
                    sqlite_cfg.get("pool_recycle", 3600),
                )
            ),
            echo=as_bool(
                self._env(
                    "database.sqlite.echo",
                    "SQLITE_ECHO",
                    sqlite_cfg.get("echo", False),
                ),
                False,
            ),
        )
        mysql = MySQLConfig(
            host=self._env("database.mysql.host", "DB_HOST", mysql_cfg.get("host", "localhost")),
            port=as_int(self._env("database.mysql.port", "DB_PORT", mysql_cfg.get("port", 3306))),
            username=self._env(
                "database.mysql.username",
                "DB_USER",
                mysql_cfg.get("username", "root"),
            ),
            password=self._env(
                "database.mysql.password",
                "DB_PASSWORD",
                mysql_cfg.get("password", "password"),
            ),
            database=self._env(
                "database.mysql.database",
                "DB_NAME",
                mysql_cfg.get("database", "funasr_tasks"),
            ),
            pool_size=as_int(
                self._env("database.mysql.pool_size", "DB_POOL_SIZE", mysql_cfg.get("pool_size", 20))
            ),
            pool_recycle=as_int(
                self._env(
                    "database.mysql.pool_recycle",
                    "DB_POOL_RECYCLE",
                    mysql_cfg.get("pool_recycle", 3600),
                )
            ),
            echo=as_bool(
                self._env("database.mysql.echo", "DB_ECHO", mysql_cfg.get("echo", False)),
                False,
            ),
        )
        return DatabaseConfig(type=database_type, sqlite=sqlite, mysql=mysql)

    def storage(self) -> StorageConfig:
        storage_cfg = self._section("storage")
        local_cfg = self._section("storage", "local")
        s3_cfg = self._section("storage", "s3")
        storage_type = str(
            self._env(
                "storage.type",
                "STORAGE_TYPE",
                storage_cfg.get("type", "local"),
            )
            or "local"
        ).strip().lower()
        runtime_paths = self.get_runtime_paths()
        local = LocalStorageConfig(
            root=runtime_paths["local_files_dir"],
            prefix=str(
                self._env(
                    "storage.local.prefix",
                    "STORAGE_LOCAL_PREFIX",
                    local_cfg.get("prefix", "audio"),
                )
                or ""
            ).strip("/\\"),
        )
        s3 = S3StorageConfig(
            endpoint=str(self._env("storage.s3.endpoint", "S3_ENDPOINT", s3_cfg.get("endpoint", "")) or ""),
            access_key=str(self._env("storage.s3.access_key", "S3_ACCESS_KEY", s3_cfg.get("access_key", "")) or ""),
            secret_key=str(self._env("storage.s3.secret_key", "S3_SECRET_KEY", s3_cfg.get("secret_key", "")) or ""),
            bucket=str(self._env("storage.s3.bucket", "S3_BUCKET", s3_cfg.get("bucket", "funasr-audio")) or ""),
            region=str(self._env("storage.s3.region", "S3_REGION", s3_cfg.get("region", "us-east-1")) or ""),
            prefix=str(self._env("storage.s3.prefix", "S3_PREFIX", s3_cfg.get("prefix", "audio")) or "").strip("/"),
        )
        return StorageConfig(type=storage_type, local=local, s3=s3)

    def engines(self) -> EnginesConfig:
        engines_cfg = self._section("engines")
        models_cfg = self._section("engines", "models")
        enabled = as_list(
            self._env(
                "engines.enabled",
                "ENGINES_ENABLED",
                engines_cfg.get("enabled", ["offline"]),
            ),
            ["offline"],
        )

        models = EnginesModelsConfig(
            offline=self._mode_models(self._section("engines", "models", "offline")),
            online=self._mode_models(self._section("engines", "models", "online")),
            spk=SpeakerModeConfig(spk=self._mapping(models_cfg.get("spk")).get("spk")),
        )
        runtime_paths = self.get_runtime_paths()
        return EnginesConfig(
            enabled=enabled,
            device=self._env(
                "engines.device",
                "ENGINES_DEVICE",
                engines_cfg.get("device", "cpu"),
            ),
            model_dir=runtime_paths["models_dir"],
            disable_model_update=as_bool(
                self._env(
                    "engines.disable_model_update",
                    "ENGINES_DISABLE_MODEL_UPDATE",
                    engines_cfg.get("disable_model_update", True),
                ),
                True,
            ),
            auto_model_download=as_bool(
                self._env(
                    "engines.auto_model_download",
                    "ENGINES_AUTO_MODEL_DOWNLOAD",
                    engines_cfg.get("auto_model_download", True),
                ),
                True,
            ),
            models=models,
        )

    def processing(self) -> ProcessingConfig:
        processing_cfg = self._section("processing")
        offline_cfg = self._section("processing", "offline_async")
        offline_processing_cfg = self._section("processing", "offline")
        online_cfg = self._section("processing", "online")
        runtime_paths = self.get_runtime_paths()

        return ProcessingConfig(
            default_mode=self._env(
                "processing.default_mode",
                "PROCESSING_DEFAULT_MODE",
                processing_cfg.get("default_mode", "offline"),
            ),
            max_concurrent_tasks=as_int(
                self._env(
                    "processing.max_concurrent_tasks",
                    "PROCESSING_MAX_CONCURRENT_TASKS",
                    processing_cfg.get("max_concurrent_tasks", 4),
                )
            ),
            worker_threads=as_int(
                self._env(
                    "processing.worker_threads",
                    "PROCESSING_WORKER_THREADS",
                    processing_cfg.get("worker_threads", 4),
                )
            ),
            timeout_seconds=as_int(
                self._env(
                    "processing.timeout_seconds",
                    "PROCESSING_TIMEOUT_SECONDS",
                    processing_cfg.get("timeout_seconds", 3600),
                )
            ),
            temp_dir=runtime_paths["temp_dir"],
            cleanup_on_complete=as_bool(
                self._env(
                    "processing.cleanup_on_complete",
                    "PROCESSING_CLEANUP_ON_COMPLETE",
                    processing_cfg.get("cleanup_on_complete", True),
                ),
                True,
            ),
            max_temp_age_hours=as_int(
                self._env(
                    "processing.max_temp_age_hours",
                    "PROCESSING_MAX_TEMP_AGE_HOURS",
                    processing_cfg.get("max_temp_age_hours", 24),
                )
            ),
            offline_async_enabled=as_bool(
                self._env(
                    "processing.offline_async.enabled",
                    "OFFLINE_ASYNC_ENABLED",
                    offline_cfg.get("enabled", True),
                ),
                True,
            ),
            offline_async_allow_immediate=as_bool(
                self._env(
                    "processing.offline_async.allow_immediate",
                    "OFFLINE_ASYNC_ALLOW_IMMEDIATE",
                    offline_cfg.get("allow_immediate", True),
                ),
                True,
            ),
            offline_spk_verification_enabled=as_bool(
                self._env(
                    "processing.offline.spk_verification_enabled",
                    "OFFLINE_SPK_VERIFICATION_ENABLED",
                    offline_processing_cfg.get("spk_verification_enabled", True),
                ),
                True,
            ),
            online_queue_max_chunks=as_int(
                self._env(
                    "processing.online.queue_max_chunks",
                    "ONLINE_QUEUE_MAX_CHUNKS",
                    online_cfg.get("queue_max_chunks", 32),
                )
            ),
            online_decode_interval=as_float(
                self._env(
                    "processing.online.decode_interval",
                    "ONLINE_DECODE_INTERVAL",
                    online_cfg.get("decode_interval", 0.48),
                )
            ),
            online_first_decode_ms=as_int(
                self._env(
                    "processing.online.first_decode_ms",
                    "ONLINE_FIRST_DECODE_MS",
                    online_cfg.get("first_decode_ms", 600),
                )
            ),
            online_chunk_ms=as_int(
                self._env(
                    "processing.online.chunk_ms",
                    "ONLINE_CHUNK_MS",
                    online_cfg.get("chunk_ms", 600),
                )
            ),
            online_chunk_size=[
                as_int(item)
                for item in as_list(
                    self._env(
                        "processing.online.chunk_size",
                        "ONLINE_CHUNK_SIZE",
                        online_cfg.get("chunk_size", [0, 10, 5]),
                    ),
                    [0, 10, 5],
                )
            ],
            online_vad_pre_padding_ms=as_int(
                self._env(
                    "processing.online.vad_pre_padding_ms",
                    "ONLINE_VAD_PRE_PADDING_MS",
                    online_cfg.get("vad_pre_padding_ms", 350),
                )
            ),
            online_vad_post_padding_ms=as_int(
                self._env(
                    "processing.online.vad_post_padding_ms",
                    "ONLINE_VAD_POST_PADDING_MS",
                    online_cfg.get("vad_post_padding_ms", 800),
                )
            ),
            online_vad_merge_gap_ms=as_int(
                self._env(
                    "processing.online.vad_merge_gap_ms",
                    "ONLINE_VAD_MERGE_GAP_MS",
                    online_cfg.get("vad_merge_gap_ms", 1200),
                )
            ),
            online_vad_min_final_ms=as_int(
                self._env(
                    "processing.online.vad_min_final_ms",
                    "ONLINE_VAD_MIN_FINAL_MS",
                    online_cfg.get("vad_min_final_ms", 2500),
                )
            ),
            online_vad_max_final_ms=as_int(
                self._env(
                    "processing.online.vad_max_final_ms",
                    "ONLINE_VAD_MAX_FINAL_MS",
                    online_cfg.get("vad_max_final_ms", 12000),
                )
            ),
        )

    def hotwords(self) -> HotwordConfig:
        hotword_cfg = self._section("hotwords")
        return HotwordConfig(
            enabled=as_bool(
                self._env(
                    "hotwords.enabled",
                    "HOTWORDS_ENABLED",
                    hotword_cfg.get("enabled", True),
                ),
                True,
            ),
            source=self._env(
                "hotwords.source",
                "HOTWORDS_SOURCE",
                hotword_cfg.get("source", "database"),
            ),
            default_ids=[
                as_int(item)
                for item in as_list(
                    self._env(
                        "hotwords.default_ids",
                        "HOTWORDS_DEFAULT_IDS",
                        hotword_cfg.get("default_ids", []),
                    ),
                    [],
                )
            ],
        )

    def _mode_models(self, config: Mapping[str, Any]) -> ModeModelConfig:
        return ModeModelConfig(
            enabled=config.get("enabled", "pt"),
            pt=ModelVariantConfig(**self._mapping(config.get("pt"))),
            onnx=ModelVariantConfig(**self._mapping(config.get("onnx"))),
        )

    def _env(self, key: str, env_var: str, default: Any) -> Any:
        return self.get_env(key, env_var, default)

    def _section(self, *parts: str) -> Mapping[str, Any]:
        value: Any = self.config
        for part in parts:
            if not isinstance(value, Mapping):
                return {}
            value = value.get(part)
        return self._mapping(value)

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["TypedConfigBuilder"]
