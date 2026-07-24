"""Pure planning helpers for runtime service preload dependencies."""

from __future__ import annotations

from typing import Iterable, List


def enabled_service_keys(
    enabled_modes: Iterable[str],
    *,
    offline_backend: str = "pt",
    offline_spk_verification_enabled: bool = True,
) -> List[str]:
    keys: List[str] = []
    seen: set[str] = set()
    offline_requires_speaker = (
        str(offline_backend).lower() == "onnx"
        or offline_spk_verification_enabled
    )
    service_keys_by_mode = {
        "offline": (
            ("offline_asr", "speaker")
            if offline_requires_speaker
            else ("offline_asr",)
        ),
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


def required_service_modes(
    mode: str,
    *,
    offline_backend: str = "pt",
    offline_spk_verification_enabled: bool = True,
) -> List[str]:
    if mode == "offline":
        offline_requires_speaker = (
            str(offline_backend).lower() == "onnx"
            or offline_spk_verification_enabled
        )
        return ["offline", "spk"] if offline_requires_speaker else ["offline"]
    if mode == "online":
        return ["online"]
    if mode == "spk":
        return ["spk"]
    return [mode]


__all__ = ["enabled_service_keys", "required_service_modes"]
