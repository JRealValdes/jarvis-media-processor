"""Visual activity envelope — consecutive-frame variation as a 1-D signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from jarvis_media_processor.video.hashing import phash_distance
from jarvis_media_processor.video.models import SampledFrame

# pHash with hash_size=8 yields 64 bits.
_DEFAULT_MAX_DISTANCE = 64


@dataclass(frozen=True)
class ActivityEnvelope:
    """Timeline activity: one float per sampled frame (0 = static)."""

    times: NDArray[np.float64]
    values: NDArray[np.float64]
    sample_period: float

    def __post_init__(self) -> None:
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have the same length")


def build_activity_envelope(
    frames: list[SampledFrame],
    *,
    max_hash_distance: int = _DEFAULT_MAX_DISTANCE,
) -> ActivityEnvelope:
    """Build ``v(t)`` from consecutive pHash distances.

    ``v[0] = 0``; ``v[i] = distance(frame[i-1], frame[i]) / max_hash_distance``.
    Larger values mean abrupt visual change (shot cuts, big motion).
    """
    if not frames:
        return ActivityEnvelope(
            times=np.zeros(0, dtype=np.float64),
            values=np.zeros(0, dtype=np.float64),
            sample_period=1.0,
        )
    ordered = sorted(frames, key=lambda f: f.time_seconds)
    times = np.asarray([f.time_seconds for f in ordered], dtype=np.float64)
    values = np.zeros(len(ordered), dtype=np.float64)
    denom = float(max(max_hash_distance, 1))
    for i in range(1, len(ordered)):
        values[i] = phash_distance(ordered[i - 1].phash, ordered[i].phash) / denom
    if len(times) >= 2:
        period = float(np.median(np.diff(times)))
    else:
        period = 1.0
    if period <= 0:
        period = 1.0
    return ActivityEnvelope(times=times, values=values, sample_period=period)


def normalize_activity(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Zero-mean, unit-variance (or zeros if flat)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    std = float(np.std(arr))
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def activity_peak_times(
    envelope: ActivityEnvelope,
    *,
    min_prominence: float = 0.15,
    min_distance_samples: int = 2,
) -> list[float]:
    """Local maxima of the activity envelope (shot-cut / motion landmarks)."""
    if envelope.values.size < 3:
        return []
    peaks, _ = find_peaks(
        envelope.values,
        prominence=min_prominence,
        distance=max(1, min_distance_samples),
    )
    return [float(envelope.times[i]) for i in peaks]
