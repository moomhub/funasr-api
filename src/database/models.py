"""
SQLAlchemy ORM 模型
定义所有数据库表结构
"""

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean,
    Float, JSON, Index, SmallInteger, UniqueConstraint, text as sql_text
)
from sqlalchemy.dialects.mysql import TINYINT
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone
from src.core.ids import new_task_id

Base = declarative_base()


class Hotword(Base):
    """热词表 - 存储各种格式的热词"""
    __tablename__ = 'hotwords'
    
    id = Column(Integer, primary_key=True)
    
    # 热词名称（用于展示和查询）
    name = Column(String(100), nullable=False)
    
    # 严格 JSON 热词数组，权重存储在每个元素的 weight 字段中
    text = Column(Text, nullable=False)
    
    # 状态
    enabled = Column(Boolean, nullable=False, default=True, server_default=sql_text("1"))
    is_deleted = Column(Boolean, nullable=False, default=False, server_default=sql_text("0"))
    
    # 时间戳
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # 索引
    __table_args__ = (
        Index('idx_enabled_deleted', 'enabled', 'is_deleted'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'text': self.text,
            'enabled': self.enabled,
            'is_deleted': self.is_deleted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class OfflineTask(Base):
    """离线任务表"""
    __tablename__ = 'offline_tasks'
    
    # 主键
    id = Column(String(36), primary_key=True, default=new_task_id)
    
    # 文件信息
    filename = Column(String(255), nullable=False)
    source_task_id = Column(String(36))
    file_size = Column(Integer)
    s3_key = Column(String(512))
    file_hash = Column(String(128))
    vip = Column(Boolean, default=False)
    
    # 任务状态
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    handle_status = Column(
        SmallInteger().with_variant(TINYINT(), "mysql"),
        nullable=False,
        default=2,
        server_default=sql_text("2"),
    )
    is_deleted = Column(Boolean, nullable=False, default=False, server_default=sql_text("0"))
    
    # 结果字段
    full_text = Column(Text)
    segments = Column(JSON)
    word_timestamps = Column(JSON)
    processing_time = Column(Float)
    
    # 错误处理
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # 时间戳
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # 新增字段：用户信息和热词配置
    email = Column(String(255))  # 用户邮箱（非必填）
    hotwords = Column(Text)  # 严格 JSON 热词数组（非必填）
    hotword_id = Column(Integer)  # 热词ID，用于从数据库查询热词（非必填）
    
    # 索引
    __table_args__ = (
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
        Index('idx_offline_tasks_source_task_id', 'source_task_id'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'filename': self.filename,
            'source_task_id': self.source_task_id,
            'file_size': self.file_size,
            's3_key': self.s3_key,
            'file_hash': self.file_hash,
            'vip': self.vip,
            'status': self.status,
            'handle_status': self.handle_status,
            'is_deleted': self.is_deleted,
            'full_text': self.full_text,
            'segments': self.segments,
            'word_timestamps': self.word_timestamps,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'email': self.email,
            'hotwords': self.hotwords,
            'hotword_id': self.hotword_id,
        }


class SpkTask(Base):
    """独立 SPK 任务表。"""
    __tablename__ = 'spk_tasks'

    id = Column(String(36), primary_key=True, default=new_task_id)
    filename = Column(String(255), nullable=False)
    source_task_id = Column(String(36))
    file_size = Column(Integer)
    email = Column(String(255))
    vip = Column(Boolean, default=False)

    s3_key = Column(String(512))
    file_hash = Column(String(128))
    status = Column(String(20), default='pending')
    handle_status = Column(
        SmallInteger().with_variant(TINYINT(), "mysql"),
        nullable=False,
        default=2,
        server_default=sql_text("2"),
    )
    is_deleted = Column(Boolean, nullable=False, default=False, server_default=sql_text("0"))

    result = Column(JSON)
    segments = Column(JSON)
    speaker_ids = Column(JSON)
    speaker_count = Column(Integer, default=0)
    processing_time = Column(Float)

    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    __table_args__ = (
        Index('idx_spk_status', 'status'),
        Index('idx_spk_created_at', 'created_at'),
        Index('idx_spk_tasks_source_task_id', 'source_task_id'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'source_task_id': self.source_task_id,
            'file_size': self.file_size,
            'email': self.email,
            'vip': self.vip,
            's3_key': self.s3_key,
            'file_hash': self.file_hash,
            'status': self.status,
            'handle_status': self.handle_status,
            'is_deleted': self.is_deleted,
            'result': self.result,
            'segments': self.segments,
            'speaker_ids': self.speaker_ids,
            'speaker_count': self.speaker_count,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class S3File(Base):
    """S3/本地归档文件索引表，用于按 hash 去重。"""
    __tablename__ = 's3_files'

    id = Column(Integer, primary_key=True)
    task_key = Column(String(64), nullable=False)
    task_type = Column(String(32), nullable=False)
    storage_backend = Column(String(16), default='s3')  # s3/local
    bucket_name = Column(String(255))
    s3_key = Column(String(512), nullable=False)
    original_filename = Column(String(255))
    stored_filename = Column(String(255), nullable=False)
    file_sha256 = Column(String(128), nullable=False)
    hash_algorithm = Column(String(32), default='sha256')
    file_size = Column(Integer)
    content_type = Column(String(100))
    local_path = Column(String(1024))
    upload_status = Column(String(20), default='uploaded')
    is_reused = Column(Boolean, default=False)
    is_deleted = Column(Boolean, nullable=False, default=False, server_default=sql_text("0"))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint('file_sha256', name='uq_s3_files_sha256'),
        Index('idx_s3_files_task_key', 'task_key'),
        Index('idx_s3_files_s3_key', 's3_key'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'task_key': self.task_key,
            'task_type': self.task_type,
            'storage_backend': self.storage_backend,
            'bucket_name': self.bucket_name,
            's3_key': self.s3_key,
            'original_filename': self.original_filename,
            'stored_filename': self.stored_filename,
            'file_sha256': self.file_sha256,
            'hash_algorithm': self.hash_algorithm,
            'file_size': self.file_size,
            'content_type': self.content_type,
            'local_path': self.local_path,
            'upload_status': self.upload_status,
            'is_reused': self.is_reused,
            'is_deleted': self.is_deleted,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


