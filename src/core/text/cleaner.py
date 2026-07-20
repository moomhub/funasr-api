"""Text cleaning utilities."""

import re


def clean_online_asr_text(text: str) -> str:
    """Clean common FunASR markup and transport artifacts."""
    if not text:
        return ""

    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[Ｏ\[\]&＆|｜]", "", text)
    text = re.sub(r"/sil|endofbreak|FFFF", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
