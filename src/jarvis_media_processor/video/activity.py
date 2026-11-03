"""Visual activity envelope — consecutive-frame variation as a 1-D signal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import find_peaks

from jarvis_media_processor.video.hashing import phash_distance
from jarvis_media_processor.video.models import SampledFrame
from jarvis_media_processor.video.preprocess import structure_change, structure_map

# pHash with hash_size=8 yields 64 bits.
_DEFAULT_MAX_DISTANCE = 64
_DEFAULT_STRUCTURE_SIZE = 64


@dataclass(frozen=True)
class ActivityEnvelope:
    """Timeline activity: one float per sampled frame (0 = static)."""

    times: NDArray[np.float64]
    values: NDArray[np.float64]
    sample_period: float

    def __post_init__(self) -> None:
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have the same length")


def _sample_period(times: NDArray[np.float64]) -> float:
    if times.size >= 2:
        period = float(np.median(np.diff(times)))
    else:
        period = 1.0
    return period if period > 0 else 1.0


def build_activity_envelope(
    frames: list[SampledFrame],
    *,
    max_hash_distance: int = _DEFAULT_MAX_DISTANCE,
) -> ActivityEnvelope:
    """Build ``v(t)`` from consecutive pHash distances.

    ``v[0] = 0``; ``v[i] = distance(frame[i-1], frame[i]) / max_hash_distance``.
    Larger values mean abrupt visual change (shot cuts, big motion).

    Prefer :func:`build_fused_activity_envelope` for production curve walking;
    pure pHash-diff remains as baseline / benchmark component.
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
    return ActivityEnvelope(times=times, values=values, sample_period=_sample_period(times))


def build_structural_activity_envelope(
    frames: Sequence[SampledFrame] | Sequence[Path] | Sequence[Image.Image],
    *,
    times: Sequence[float] | None = None,
    target_size: int = _DEFAULT_STRUCTURE_SIZE,
    crop_bars: bool = True,
) -> ActivityEnvelope:
    """Build ``v(t)`` from consecutive structural-map change (encode-robust).

    Accepts ``SampledFrame`` (reads ``path``), image paths, or open PIL images.
    When passing bare images/paths, provide ``times`` (defaults to ``0..n-1``).
    """
    maps, sample_times = _load_structure_series(
        frames, times=times, target_size=target_size, crop_bars=crop_bars
    )
    if not sample_times:
        return ActivityEnvelope(
            times=np.zeros(0, dtype=np.float64),
            values=np.zeros(0, dtype=np.float64),
            sample_period=1.0,
        )
    times_arr = np.asarray(sample_times, dtype=np.float64)
    values = np.zeros(len(maps), dtype=np.float64)
    for i in range(1, len(maps)):
        values[i] = structure_change(maps[i - 1], maps[i])
    return ActivityEnvelope(
        times=times_arr, values=values, sample_period=_sample_period(times_arr)
    )


def build_fused_activity_envelope(
    frames: list[SampledFrame],
    *,
    structural_weight: float = 0.5,
    max_hash_distance: int = _DEFAULT_MAX_DISTANCE,
    target_size: int = _DEFAULT_STRUCTURE_SIZE,
    crop_bars: bool = True,
) -> ActivityEnvelope:
    """Fuse pHash-diff and structural activity for production curve walking.

    Each series is scaled to its own peak, then combined as
    ``max(phash_norm, structural_weight * structural_norm)``. Pure structural
    curves improve cross-encode shape match but can alias through flat black
    ads; keeping a pHash floor preserves sharp encode-local cuts while the
    structural term still lifts shared edge-change landmarks.
    """
    phash = build_activity_envelope(frames, max_hash_distance=max_hash_distance)
    structural = build_structural_activity_envelope(
        frames, target_size=target_size, crop_bars=crop_bars
    )
    if phash.values.size == 0:
        return phash
    ph_peak = float(np.max(phash.values)) + 1e-12
    st_peak = float(np.max(structural.values)) + 1e-12
    weight = max(0.0, float(structural_weight))
    fused = np.maximum(phash.values / ph_peak, weight * (structural.values / st_peak))
    return ActivityEnvelope(
        times=phash.times,
        values=fused.astype(np.float64, copy=False),
        sample_period=phash.sample_period,
    )


def _load_structure_series(
    frames: Sequence[SampledFrame] | Sequence[Path] | Sequence[Image.Image],
    *,
    times: Sequence[float] | None,
    target_size: int,
    crop_bars: bool,
) -> tuple[list[NDArray[np.float64]], list[float]]:
    if not frames:
        return [], []
    first = frames[0]
    maps: list[NDArray[np.float64]] = []
    sample_times: list[float] = []

    if isinstance(first, SampledFrame):
        sampled = [f for f in frames if isinstance(f, SampledFrame)]
        ordered = sorted(sampled, key=lambda f: f.time_seconds)
        for item in ordered:
            with Image.open(item.path) as img:
                maps.append(
                    structure_map(img, target_size=target_size, crop_bars=crop_bars)
                )
            sample_times.append(float(item.time_seconds))
        return maps, sample_times

    if times is not None and len(times) != len(frames):
        raise ValueError("times must match the number of frames")

    for index, item in enumerate(frames):
        if isinstance(item, Image.Image):
            maps.append(structure_map(item, target_size=target_size, crop_bars=crop_bars))
        else:
            path = Path(item)
            with Image.open(path) as img:
                maps.append(
                    structure_map(img, target_size=target_size, crop_bars=crop_bars)
                )
        sample_times.append(float(times[index]) if times is not None else float(index))
    return maps, sample_times


def aligned_activity_correlation(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    speed: float = 1.0,
    offset_seconds: float = 0.0,
    base_start: float | None = None,
    base_end: float | None = None,
    period: float | None = None,
) -> float:
    """Pearson-like correlation of activity shapes under ``donor ≈ speed*base+offset``.

    Used to compare pHash-diff vs structural envelopes on known-aligned spans
    without involving the curve-walk ranker.
    """
    if base.times.size < 4 or donor.times.size < 4:
        return 0.0
    step = float(period or np.median([base.sample_period, donor.sample_period]) or 1.0)
    lo = float(base.times[0] if base_start is None else base_start)
    hi = float(base.times[-1] if base_end is None else base_end)
    if hi - lo < 4 * step:
        return 0.0
    grid = np.arange(lo, hi + step * 0.5, step, dtype=np.float64)
    base_vals = np.interp(grid, base.times, base.values, left=0.0, right=0.0)
    donor_grid = speed * grid + offset_seconds
    donor_vals = np.interp(donor_grid, donor.times, donor.values, left=0.0, right=0.0)
    a = normalize_activity(base_vals)
    b = normalize_activity(donor_vals)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


def normalize_activity(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Zero-mean, unit-variance (or zeros if flat)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    std = float(np.std(arr))
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def _window_values(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    center: float,
    half: float,
    period: float,
) -> NDArray[np.float64]:
    if times.size == 0 or half <= 0:
        return np.zeros(0, dtype=np.float64)
    grid = np.arange(center - half, center + half + period * 0.5, period, dtype=np.float64)
    return np.interp(grid, times, values, left=0.0, right=0.0)


def local_activity_score(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_time: float,
    speed: float = 1.0,
    half_window: float = 12.0,
) -> float:
    """Normalized correlation of local activity shapes (derivative match)."""
    period = float(np.median([base.sample_period, donor.sample_period])) or 1.0
    bw = _window_values(base.times, base.values, base_time, half_window, period)
    donor_half = half_window * max(speed, 1e-6)
    dw = _window_values(donor.times, donor.values, donor_time, donor_half, period)
    n = min(bw.size, dw.size)
    if n < 4:
        return 0.0
    a = normalize_activity(bw[:n])
    b = normalize_activity(dw[:n])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


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
