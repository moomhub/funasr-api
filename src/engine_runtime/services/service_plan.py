"""Pure planning helpers for runtime service preload dependencies."""

from __future__ import annotations

from typing import Iterable, List


def enabled_service_keys(enabled_modes: Iterable[str]) -> List[str]:
    keys: List[str] = []
    seen: set[str] = set()
    service_keys_by_mode = {
        "offline": ("offline_asr", "speaker"),
        "online": ("online_asr",),
        "spk": ("speaker",),
    }

    for mode in enabled_modes:
        for key in service_keys_by_mode.get(mode, ()):
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def required_service_modes(mode: str) -> List[str]:
    if mode == "offline":
        return ["offline", "spk"]
    if mode == "online":
        return ["online"]
    if mode == "spk":
        return ["spk"]
    return [mode]


__all__ = ["enabled_service_keys", "required_service_modes"]
