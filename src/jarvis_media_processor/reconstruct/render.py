"""Render placements onto a silence canvas matching the base duration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from jarvis_media_processor.reconstruct.models import Placement


def _slice_samples(
    samples: NDArray[np.floating],
    sample_rate: int,
    start: float,
    end: float,
) -> NDArray[np.float64]:
    start_i = max(0, int(round(start * sample_rate)))
    end_i = min(samples.shape[-1], int(round(end * sample_rate)))
    if end_i <= start_i:
        return np.zeros(0, dtype=np.float64)
    if samples.ndim == 1:
        return np.asarray(samples[start_i:end_i], dtype=np.float64)
    return np.asarray(samples[:, start_i:end_i], dtype=np.float64)


def _ensure_2d(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


def _time_stretch(
    chunk: NDArray[np.float64],
    *,
    source_duration: float,
    target_duration: float,
    sample_rate: int,
) -> NDArray[np.float64]:
    """Stretch/compress audio to ``target_duration`` via resampling.

    For mild speed changes this approximates ``atempo`` well enough for tests
    and first reconstruction passes. Large ratios may sound unnatural.
    """
    target_n = max(1, int(round(target_duration * sample_rate)))
    if chunk.shape[-1] == 0:
        return np.zeros((chunk.shape[0], target_n), dtype=np.float64)
    if abs(source_duration - target_duration) < 1e-6 and chunk.shape[-1] == target_n:
        return chunk
    stretched = resample(chunk, target_n, axis=-1)
    return np.asarray(stretched, dtype=np.float64)


def _apply_edges(
    chunk: NDArray[np.float64],
    *,
    fade_samples: int,
) -> NDArray[np.float64]:
    if fade_samples <= 0 or chunk.shape[-1] == 0:
        return chunk
    fade = min(fade_samples, chunk.shape[-1] // 2)
    if fade <= 0:
        return chunk
    out = chunk.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
    out[:, :fade] *= ramp
    out[:, -fade:] *= ramp[::-1]
    return out


def render_placements(
    donor_samples: NDArray[np.floating],
    sample_rate: int,
    placements: list[Placement],
    *,
    base_duration: float,
) -> NDArray[np.float64]:
    """Mix placements onto a zero canvas of ``base_duration``.

    Missing intervals remain silence and do not shift later content.
    """
    donor = _ensure_2d(donor_samples)
    channels = donor.shape[0]
    total = max(1, int(round(base_duration * sample_rate)))
    canvas = np.zeros((channels, total), dtype=np.float64)

    for placement in placements:
        donor_chunk = _slice_samples(
            donor,
            sample_rate,
            placement.donor_start,
            placement.donor_end,
        )
        donor_chunk = _ensure_2d(donor_chunk)
        source_duration = max(1e-6, placement.donor_end - placement.donor_start)
        target_duration = max(1e-6, placement.base_end - placement.base_start)
        stretched = _time_stretch(
            donor_chunk,
            source_duration=source_duration,
            target_duration=target_duration,
            sample_rate=sample_rate,
        )
        fade = int(round(placement.crossfade_seconds * sample_rate))
        stretched = _apply_edges(stretched, fade_samples=fade)

        start_i = max(0, int(round(placement.base_start * sample_rate)))
        end_i = min(total, start_i + stretched.shape[-1])
        length = end_i - start_i
        if length <= 0:
            continue
        canvas[:, start_i:end_i] += stretched[:, :length]

    # Soft clip to [-1, 1]
    peak = np.max(np.abs(canvas)) if canvas.size else 0.0
    if peak > 1.0:
        canvas = canvas / peak
    return canvas
