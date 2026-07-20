"""Pure helpers for cross-segment text boundary de-duplication."""

from __future__ import annotations


BOUNDARY_PUNCTUATION = "，。！？；：、,.!?;:"
REPEAT_SEPARATORS = "，,、"


def split_trailing_punctuation(text: str) -> tuple[str, str]:
    core = text.rstrip()
    punctuation = ""
    while core and core[-1] in BOUNDARY_PUNCTUATION:
        punctuation = core[-1] + punctuation
        core = core[:-1].rstrip()
    return core, punctuation


def phrase_for_repeat_match(text: str) -> str:
    core, _ = split_trailing_punctuation(text.strip())
    return core.strip()


def remove_repeated_previous_suffix(
    previous_text: str,
    current_text: str,
    gap_ms: int,
    merge_gap_ms: int,
) -> str:
    previous_core, previous_punctuation = split_trailing_punctuation(previous_text)
    current_core = current_text.lstrip()
    max_overlap = min(12, len(previous_core), len(current_core))
    for size in range(max_overlap, 0, -1):
        if size == 1 and gap_ms > min(600, merge_gap_ms):
            continue
        if not previous_core.endswith(current_core[:size]):
            continue
        trimmed = previous_core[:-size].rstrip()
        if not trimmed:
            return previous_text
        return f"{trimmed}{previous_punctuation}"
    return previous_text


def collapse_adjacent_repeated_phrases(text: str) -> str:
    current = text
    for separator in REPEAT_SEPARATORS:
        parts = current.split(separator)
        if len(parts) < 2:
            continue
        collapsed = []
        for part in parts:
            normalized = phrase_for_repeat_match(part)
            previous = phrase_for_repeat_match(collapsed[-1]) if collapsed else ""
            if normalized and normalized == previous and 1 <= len(normalized) <= 12:
                continue
            collapsed.append(part)
        current = separator.join(collapsed)
    return current.strip()


def repair_previous_boundary_overlap(
    previous_text: str,
    previous_end_ms: int,
    current_text: str,
    current_start_ms: int,
    merge_gap_ms: int,
) -> tuple[str, str]:
    """Return possibly repaired previous text plus stripped current text."""
    current_text = current_text.strip()
    if not current_text:
        return previous_text, current_text

    try:
        gap_ms = int(current_start_ms) - int(previous_end_ms)
    except (TypeError, ValueError):
        return previous_text, current_text

    if gap_ms < 0 or gap_ms > merge_gap_ms:
        return previous_text, current_text

    repaired_previous = remove_repeated_previous_suffix(
        previous_text,
        current_text,
        gap_ms,
        merge_gap_ms,
    )
    return repaired_previous, current_text


__all__ = [
    "BOUNDARY_PUNCTUATION",
    "collapse_adjacent_repeated_phrases",
    "phrase_for_repeat_match",
    "repair_previous_boundary_overlap",
    "remove_repeated_previous_suffix",
    "split_trailing_punctuation",
]
