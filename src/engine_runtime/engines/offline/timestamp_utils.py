"""时间戳处理工具函数 - 消除 PT/ONNX recognizer 中的重复代码"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.core.text_utils import is_ascii_word_char, is_text_separator


def normalize_timestamps(data: Any) -> Optional[List[List[Any]]]:
    """标准化时间戳数据为 [[text, start_ms, end_ms], ...] 格式

    支持输入：
    - JSON 字符串
    - [[text, start, end], ...] (3元素)
    - [[start, end], ...] (2元素，text 填空字符串)

    返回：
        标准化后的时间戳列表，或 None（如果输入无效）
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return None
    if not isinstance(data, list):
        return None

    normalized: List[List[Any]] = []
    for item in data:
        if not isinstance(item, (list, tuple)):
            continue
        if len(item) == 3:
            normalized.append([str(item[0]), int(item[1]), int(item[2])])
        elif len(item) == 2:
            normalized.append(["", int(item[0]), int(item[1])])
    return normalized or None


def offset_timestamps(timestamps: List[List[Any]], offset_ms: int) -> List[List[Any]]:
    """将所有时间戳偏移指定毫秒数

    参数：
        timestamps: [[text, start, end], ...] 格式的时间戳列表
        offset_ms: 偏移量（毫秒）

    返回：
        偏移后的时间戳列表
    """
    if not timestamps or offset_ms <= 0:
        return timestamps
    return [
        [item[0], int(item[1] + offset_ms), int(item[2] + offset_ms)]
        for item in timestamps
        if len(item) >= 3
    ]


def maybe_offset_timestamps(
    timestamps: List[List[Any]],
    start_ms: int,
    end_ms: int,
) -> List[List[Any]]:
    """根据句子时间范围判断是否需要偏移时间戳

    当时间戳是相对偏移（从0开始）且句子有正的 start_ms 时，
    将时间戳偏移到句子的绝对时间范围内。

    参数：
        timestamps: 时间戳列表
        start_ms: 句子开始时间（毫秒）
        end_ms: 句子结束时间（毫秒）

    返回：
        可能偏移后的时间戳列表
    """
    if not timestamps or start_ms <= 0:
        return timestamps
    last_end = max(int(item[2]) for item in timestamps if len(item) >= 3)
    sentence_duration = max(0, end_ms - start_ms)
    if sentence_duration and last_end <= sentence_duration + 50:
        return [
            [item[0], int(item[1] + start_ms), int(item[2] + start_ms)]
            for item in timestamps
        ]
    return timestamps


def align_text_to_timestamps(
    text: str,
    timestamps: List[List[Any]],
    *,
    include_timestamp_detail: bool = False,
) -> List[Dict[str, Any]]:
    """将文本与时间戳对齐，生成带时间信息的 token 列表

    算法：
    1. 遍历文本字符，按 ASCII 单词或单个中文字符分词
    2. 跳过分隔符（附加到前一个 token）
    3. 每个 token 对应一个时间戳

    参数：
        text: 待对齐的文本
        timestamps: [[text, start, end], ...] 格式的时间戳列表
        include_timestamp_detail: 是否在结果中包含 timestamp 详情（PT 模式需要）

    返回：
        [{"text": str, "start": int, "end": int, "timestamp"?: list}, ...]
    """
    tokens = []
    timestamp_index = 0
    pending_prefix = ""
    index = 0

    while index < len(text):
        char = text[index]
        if is_text_separator(char):
            if tokens:
                tokens[-1]["text"] += char
            else:
                pending_prefix += char
            index += 1
            continue

        if is_ascii_word_char(char):
            next_index = index + 1
            while next_index < len(text) and is_ascii_word_char(text[next_index]):
                next_index += 1
            unit = text[index:next_index]
            index = next_index
        else:
            unit = char
            index += 1

        if timestamp_index >= len(timestamps):
            if tokens:
                tokens[-1]["text"] += unit
            continue

        timestamp_item = timestamps[timestamp_index]
        timestamp_index += 1
        if len(timestamp_item) < 3:
            continue

        token: Dict[str, Any] = {
            "text": pending_prefix + unit,
            "start": int(timestamp_item[1]),
            "end": int(timestamp_item[2]),
        }
        if include_timestamp_detail:
            token["timestamp"] = [list(timestamp_item)]

        tokens.append(token)
        pending_prefix = ""

    if pending_prefix and tokens:
        tokens[-1]["text"] += pending_prefix
    return tokens


def fill_timestamp_tokens(text: str, timestamps: List[List]) -> List[List]:
    """用实际 token 文本填充时间戳

    当时间戳数量与 token 数量匹配时，
    将时间戳中的占位符替换为实际的 token 文本。

    参数：
        text: 原始文本
        timestamps: [[text, start, end], ...] 格式的时间戳列表

    返回：
        填充后的时间戳列表
    """
    if not text or not timestamps:
        return timestamps

    raw_tokens = [token for token in str(text).split() if token]
    if len(raw_tokens) != len(timestamps):
        raw_tokens = [
            char
            for char in str(text)
            if char.strip() and not is_text_separator(char)
        ]
    if len(raw_tokens) != len(timestamps):
        return timestamps

    filled = []
    for token, item in zip(raw_tokens, timestamps):
        if len(item) >= 3:
            filled.append([token, int(item[1]), int(item[2])])
        elif len(item) == 2:
            filled.append([token, int(item[0]), int(item[1])])
    return filled
