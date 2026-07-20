"""Chunked audio buffering for realtime sessions."""

from __future__ import annotations

from typing import List

import numpy as np


class ChunkedAudioBuffer:
    """Keep absolute sample offsets while discarding already-consumed chunks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start_sample = 0
        self.total_samples = 0
        self.chunks: List[np.ndarray] = []

    def replace(self, audio: np.ndarray) -> None:
        normalized = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.start_sample = 0
        self.total_samples = len(normalized)
        self.chunks = [normalized] if normalized.size else []

    def append(self, audio: np.ndarray) -> None:
        normalized = np.asarray(audio, dtype=np.float32).reshape(-1)
        if normalized.size:
            self.chunks.append(normalized)
            self.total_samples += normalized.size

    def slice(self, start_sample: int, end_sample: int) -> np.ndarray:
        start = max(start_sample, self.start_sample)
        end = min(end_sample, self.total_samples)
        if end <= start or not self.chunks:
            return np.array([], dtype=np.float32)

        collected: List[np.ndarray] = []
        current = self.start_sample
        for chunk in self.chunks:
            chunk_end = current + len(chunk)
            if chunk_end <= start:
                current = chunk_end
                continue
            if current >= end:
                break
            local_start = max(start - current, 0)
            local_end = min(end - current, len(chunk))
            if local_end > local_start:
                collected.append(chunk[local_start:local_end])
            current = chunk_end

        if not collected:
            return np.array([], dtype=np.float32)
        if len(collected) == 1:
            return collected[0].copy()
        return np.concatenate(collected)

    def drop_before(self, sample_index: int) -> None:
        target = max(self.start_sample, min(sample_index, self.total_samples))
        if target <= self.start_sample or not self.chunks:
            return

        new_chunks: List[np.ndarray] = []
        current = self.start_sample
        for chunk in self.chunks:
            chunk_end = current + len(chunk)
            if chunk_end <= target:
                current = chunk_end
                continue
            if target > current:
                chunk = chunk[target - current:]
                current = target
            new_chunks.append(chunk)
            current = chunk_end

        self.chunks = new_chunks
        self.start_sample = target


__all__ = ["ChunkedAudioBuffer"]
