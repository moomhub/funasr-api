"""文本处理工具函数 - 消除重复代码"""

from typing import Any


def is_ascii_word_char(char: str) -> bool:
    """判断字符是否是 ASCII 单词字符（字母、数字、撇号、连字符）"""
    return char.isascii() and (char.isalpha() or char.isdigit() or char in {"'", "-"})


def is_text_separator(char: str) -> bool:
    """判断字符是否是文本分隔符（空格、标点符号）"""
    return char.isspace() or char in set("。！？!?，,、：:；;（）()【】[]《》<>""''…")


def compact_token_text(text: Any) -> str:
    """压缩 token 文本，移除空格"""
    return str(text or "").replace(" ", "").strip()


def text_tokens(text: Any) -> list:
    """将文本拆分为 token 列表"""
    return [char for char in str(text or "").replace(" ", "") if char.strip()]
