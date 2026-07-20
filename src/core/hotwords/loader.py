"""热词加载和解析逻辑"""

import logging
import json
from dataclasses import dataclass
from typing import Any, List, Optional

from src.core.debug_logging import log_exception

logger = logging.getLogger(__name__)

HOTWORD_FORMAT_ERROR = (
    'hotwords 必须是 JSON 数组，元素格式必须为 '
    '{"weight": 1-100, "hotword": "非空字符串"}'
)


class InvalidHotwordFormatError(ValueError):
    """Raised when custom hotwords do not match the canonical JSON schema."""


@dataclass(frozen=True)
class ResolvedHotwords:
    """Model-ready hotwords plus their logical entry count."""

    value: Any
    count: int


def parse_hotwords(hotwords_text: Any) -> List[dict]:
    """Parse the canonical ``[{weight, hotword}]`` JSON representation."""
    if hotwords_text is None:
        return []
    if isinstance(hotwords_text, str):
        try:
            items = json.loads(hotwords_text)
        except json.JSONDecodeError as exc:
            raise InvalidHotwordFormatError(HOTWORD_FORMAT_ERROR) from exc
    else:
        raise InvalidHotwordFormatError(HOTWORD_FORMAT_ERROR)
    return _normalize_hotword_items(items)


def _normalize_hotword_items(items: Any) -> List[dict]:
    if not isinstance(items, list):
        raise InvalidHotwordFormatError(HOTWORD_FORMAT_ERROR)
    result = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"weight", "hotword"}:
            raise InvalidHotwordFormatError(HOTWORD_FORMAT_ERROR)
        weight = item["weight"]
        word = item["hotword"]
        if isinstance(weight, bool) or not isinstance(weight, int) or not 1 <= weight <= 100:
            raise InvalidHotwordFormatError(HOTWORD_FORMAT_ERROR)
        if not isinstance(word, str) or not word.strip():
            raise InvalidHotwordFormatError(HOTWORD_FORMAT_ERROR)
        result.append({"weight": weight, "hotword": word.strip()})
    return result


def format_hotwords_for_model(hotwords: List[dict], weighted: bool = False) -> Any:
    """Format normalized hotwords for the currently configured model."""
    if weighted:
        return [[item["weight"], item["hotword"]] for item in hotwords if item.get("hotword")]
    return " ".join(item["hotword"] for item in hotwords if item.get("hotword"))


def get_hotwords_by_id(hotword_id: int, lookup: Any = None) -> List:
    """从数据库获取指定 ID 的热词
    
    参数：
        hotword_id: 热词ID
        
    返回：
        FunASR 格式的热词列表
    """
    if lookup is None:
        raise ValueError("hotword lookup is required")
    try:
        return lookup(hotword_id)
    except Exception as exc:
        log_exception(
            logger,
            logging.ERROR,
            "Hotword lookup",
            exc,
            context={"hotword_id": hotword_id},
        )
        return []


def resolve_hotwords_with_priority(
    custom_hotwords: Optional[str] = None,
    hotword_id: Optional[int] = None,
    default_hotword_ids: Optional[List[int]] = None,
    config: Any = None,
    hotword_lookup: Any = None,
) -> ResolvedHotwords:
    """按优先级加载热词
    
    优先级：
    1. custom_hotwords（传递的热词字符串）
    2. hotword_id（数据库查询）
    3. default_hotword_ids（默认热词ID列表）
    
    参数：
        custom_hotwords: 严格热词 JSON 数组
        hotword_id: 热词ID
        default_hotword_ids: 默认热词ID列表
        
    返回：
        FunASR 格式的热词列表：[[frequency, "word"], ...]
    """
    # 优先级 1: 使用传递的热词
    if config is None:
        raise ValueError("config is required")
    weighted = config.get("hotwords.model_format", "plain") == "weighted"

    if custom_hotwords is not None:
        logger.info(f"✅ 使用自定义热词")
        result = parse_hotwords(custom_hotwords)
        if result:
            return ResolvedHotwords(
                value=format_hotwords_for_model(result, weighted=weighted),
                count=len(result),
            )
        logger.warning(f"⚠️ 自定义热词解析失败，尝试下一优先级")
    
    # 优先级 2: 使用热词ID
    if hotword_id is not None:
        logger.info(f"✅ 使用热词ID: {hotword_id}")
        result = get_hotwords_by_id(hotword_id, lookup=hotword_lookup)
        if result:
            return ResolvedHotwords(
                value=format_hotwords_for_model(result, weighted=weighted),
                count=len(result),
            )
        logger.warning(f"⚠️ 热词ID未找到，尝试下一优先级")
    
    # 优先级 3: 使用默认热词ID
    if default_hotword_ids:
        logger.info(f"✅ 使用默认热词ID: {default_hotword_ids}")
        result = []
        for hw_id in default_hotword_ids:
            hotwords = get_hotwords_by_id(hw_id, lookup=hotword_lookup)
            result.extend(hotwords)
        if result:
            return ResolvedHotwords(
                value=format_hotwords_for_model(result, weighted=weighted),
                count=len(result),
            )
        logger.warning(f"⚠️ 默认热词ID未找到")
    
    # 都没有，返回空列表
    logger.info("ℹ️ 未加载任何热词")
    return ResolvedHotwords(value=[], count=0)


def load_hotwords_with_priority(
    custom_hotwords: Optional[str] = None,
    hotword_id: Optional[int] = None,
    default_hotword_ids: Optional[List[int]] = None,
    config: Any = None,
    hotword_lookup: Any = None,
) -> Any:
    """Load model-ready hotwords while preserving the existing public return type."""
    return resolve_hotwords_with_priority(
        custom_hotwords=custom_hotwords,
        hotword_id=hotword_id,
        default_hotword_ids=default_hotword_ids,
        config=config,
        hotword_lookup=hotword_lookup,
    ).value
