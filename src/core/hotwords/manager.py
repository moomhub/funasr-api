"""Hotword manager implementation."""

import logging
from typing import Any, List

from src.core.debug_logging import log_exception

logger = logging.getLogger(__name__)


class HotwordManager:
    """热词管理器 - 直接从数据库读取热词"""
    
    def __init__(self, config: Any, provider: Any):
        """初始化热词管理器"""
        hw_config = config.get_hotword_config()
        
        self.enabled = hw_config.enabled
        self.source = hw_config.source
        self.provider = provider
        
        logger.info(f"✅ 热词管理器初始化完成 (来源: {self.source})")
    
    def get_hotwords(self) -> List:
        """获取热词列表（FunASR 格式）
        
        直接从数据库读取所有启用的热词，不使用缓存。
        
        返回：
            热词列表，格式：[[frequency, "word"], ...]
        """
        if not self.enabled:
            logger.debug("热词功能已禁用")
            return []
        
        # 直接从数据库获取所有启用的热词
        if self.source == "database":
            try:
                hotwords = self.provider.get_hotwords()
                
                logger.debug(f"✅ 从数据库加载热词: {len(hotwords)} 个")
                return hotwords
            except Exception as exc:
                log_exception(logger, logging.ERROR, "Hotword provider loading", exc)
                return []
        
        return []

