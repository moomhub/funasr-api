"""Speaker-related text utilities."""

from typing import Any, Callable, Dict, List, Optional

from src.core.results.types import Segment


def extract_segments_from_sentence_info(
    sentence_info: List[Dict],
    *,
    speaker_normalizer: Optional[Callable[[Any], Any]] = None,
) -> List[Segment]:
    """
    从句子信息提取段
    
    参数：
        sentence_info: FunASR 返回的句子信息
    
    返回：
        段列表
    """
    segments = []
    
    for sent in sentence_info:
        speaker = sent.get("spk", 0)
        if speaker_normalizer is not None:
            speaker = speaker_normalizer(speaker)
        segment = Segment(
            text=sent.get("text", ""),
            start=sent.get("start", 0),
            end=sent.get("end", 0),
            speaker=speaker,
            is_final=True,
            timestamp=sent.get("timestamp", None)
        )
        segments.append(segment)
    
    return segments


def build_full_text_with_speaker(segments: List[Segment]) -> str:
    """构建带说话人标记的完整文本"""
    if not segments:
        return ""
    
    # 分组：同一个说话人的连续文本不加分隔符
    result = []
    current_speaker = None
    current_text = []
    
    for seg in segments:
        if seg.speaker != current_speaker:
            if current_text:
                speaker_text = "".join(current_text)
                result.append(f"[说话人 {current_speaker}] {speaker_text}")
            
            current_speaker = seg.speaker
            current_text = [seg.text]
        else:
            current_text.append(seg.text)
    
    # 处理最后一个说话人
    if current_text:
        speaker_text = "".join(current_text)
        result.append(f"[说话人 {current_speaker}] {speaker_text}")
    
    return "\n".join(result)
