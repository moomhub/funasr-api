"""Text merging utilities."""

from .cleaner import clean_online_asr_text


def merge_online_partial_text(previous: str, current: str) -> str:
    """Prefer the latest streaming hypothesis while tolerating delta-style outputs."""
    previous = clean_online_asr_text(previous)
    current = clean_online_asr_text(current)

    if not previous:
        return current
    if not current:
        return previous
    if current.startswith(previous) or previous.startswith(current):
        return current

    common_prefix = 0
    for prev_char, curr_char in zip(previous, current):
        if prev_char != curr_char:
            break
        common_prefix += 1
    if common_prefix > 0:
        return current

    max_overlap = min(len(previous), len(current))
    for size in range(max_overlap, 0, -1):
        if previous.endswith(current[:size]):
            return f"{previous}{current[size:]}"

    return current
