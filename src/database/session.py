"""
数据库会话管理
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging
from contextlib import contextmanager
from typing import Generator

from src.core.debug_logging import mask_url, log_exception

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseSchemaError(RuntimeError):
    """Raised when an existing database is missing required schema columns."""


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_url: str, pool_size: int = 20, pool_recycle: int = 3600, echo: bool = False):
        """
        初始化数据库管理器
        
        参数：
            db_url: 数据库连接字符串
            pool_size: 连接池大小
            pool_recycle: 连接回收时间（秒）
            echo: 是否打印 SQL 语句
        """
        self.db_url = db_url
        
        engine_kwargs = {
            "poolclass": QueuePool,
            "pool_size": pool_size,
            "max_overflow": 10,
            "pool_recycle": pool_recycle,
            "echo": echo,
            "pool_pre_ping": True,
        }
        if db_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self.engine = create_engine(db_url, **engine_kwargs)
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database manager initialized: backend=%s", self.engine.url.get_backend_name())
        logger.debug("Database connection URL: %s", mask_url(db_url))
    
    def init_db(self):
        """创建缺失的表，并严格校验已有表的当前 schema。"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self._validate_schema()
            logger.info("✅ 数据库表创建成功")
        except Exception as exc:
            log_exception(logger, logging.ERROR, "Database schema initialization", exc)
            raise

    def _validate_schema(self) -> None:
        """Require every declared column while allowing deployment-specific extras."""
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names())
        missing = []

        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                missing.append(f"{table.name}（整表缺失）")
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing_columns:
                    missing.append(f"{table.name}.{column.name}")

        if missing:
            details = ", ".join(missing)
            raise DatabaseSchemaError(
                "数据库 schema 不兼容，缺少必需表或列："
                f"{details}。应用不会自动迁移，请由部署方备份后升级或重建数据库。"
            )
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()

    def check_connection(self) -> None:
        """Verify that the selected database accepts a simple query."""
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        使用上下文管理器获取会话
        
        使用方法：
            with db.session_scope() as session:
                # 操作数据库
                pass
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            log_exception(logger, logging.ERROR, "Database transaction", exc)
            raise
        finally:
            session.close()
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()
        logger.info("✅ 数据库连接已关闭")


__all__ = ["DatabaseManager", "DatabaseSchemaError"]

