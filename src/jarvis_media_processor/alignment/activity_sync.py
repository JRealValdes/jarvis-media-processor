"""Sync timelines by cross-correlating visual activity envelopes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from jarvis_media_processor.alignment.evidence import TimePair
from jarvis_media_processor.video.activity import (
    ActivityEnvelope,
    activity_peak_times,
    normalize_activity,
)


@dataclass(frozen=True)
class ActivitySyncResult:
    """Global transform implied by activity-envelope correlation."""

    speed: float
    offset_seconds: float
    score: float
    method: str


def _resample_to_period(
    envelope: ActivityEnvelope,
    period: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample envelope values onto a uniform grid with step ``period``."""
    if envelope.times.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    t0 = float(envelope.times[0])
    t1 = float(envelope.times[-1])
    if t1 <= t0:
        return envelope.times.copy(), envelope.values.copy()
    grid = np.arange(t0, t1 + period * 0.5, period, dtype=np.float64)
    values = np.interp(grid, envelope.times, envelope.values)
    return grid, values


def estimate_offset_crosscorr(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
) -> ActivitySyncResult:
    """Estimate constant offset via normalized cross-correlation.

    Model: ``donor_time ≈ base_time + offset``.
    A positive offset means donor events occur later than the same base events.
    """
    period = float(np.median([base.sample_period, donor.sample_period]))
    if period <= 0:
        period = 1.0
    _bt, base_v = _resample_to_period(base, period)
    _dt, donor_v = _resample_to_period(donor, period)
    if base_v.size < 4 or donor_v.size < 4:
        return ActivitySyncResult(speed=1.0, offset_seconds=0.0, score=0.0, method="crosscorr")

    a = normalize_activity(base_v)
    b = normalize_activity(donor_v)
    corr = np.correlate(a, b, mode="full")
    # np.correlate lag convention is opposite to donor≈base+offset: negate.
    lags = np.arange(-(len(b) - 1), len(a), dtype=np.int64)
    best_i = int(np.argmax(corr))
    lag = int(lags[best_i])
    peak = float(corr[best_i])
    score = max(
        0.0,
        min(1.0, abs(peak) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
    )
    return ActivitySyncResult(
        speed=1.0,
        offset_seconds=-float(lag) * period,
        score=score,
        method="crosscorr",
    )


def estimate_speed_offset_activity(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    speed_min: float = 0.85,
    speed_max: float = 1.2,
    speed_steps: int = 36,
) -> ActivitySyncResult:
    """Grid search over mild speed changes, then cross-correlate.

    Resamples the donor envelope in time so that
    ``donor_time ≈ speed * base_time + offset``.
    """
    period = float(np.median([base.sample_period, donor.sample_period]))
    if period <= 0:
        period = 1.0
    _bt, base_v = _resample_to_period(base, period)
    if base_v.size < 4 or donor.values.size < 4:
        return estimate_offset_crosscorr(base, donor)

    best = estimate_offset_crosscorr(base, donor)
    a = normalize_activity(base_v)

    for speed in np.linspace(speed_min, speed_max, speed_steps):
        # Compress/expand donor timeline: length scales with 1/speed relative to base clock.
        # If donor runs "faster" on wall clock (speed<1 in donor≈speed*base+off),
        # donor envelope is shorter in base-time units when warped onto base grid.
        n = max(4, int(round(len(donor.values) / max(speed, 1e-6))))
        warped = np.asarray(resample(donor.values, n), dtype=np.float64)
        if warped.size < 4:
            continue
        b = normalize_activity(warped)
        corr = np.correlate(a, b, mode="full")
        lags = np.arange(-(len(b) - 1), len(a), dtype=np.int64)
        best_i = int(np.argmax(corr))
        lag = int(lags[best_i])
        score = max(
            0.0,
            min(1.0, abs(float(corr[best_i])) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
        )
        # Prefer scores near speed=1 on ties.
        tie = (score, -abs(float(speed) - 1.0))
        best_tie = (best.score, -abs(best.speed - 1.0))
        if tie > best_tie:
            best = ActivitySyncResult(
                speed=float(speed),
                offset_seconds=-float(lag) * period,
                score=score,
                method="activity_speed_grid",
            )
    return best


def activity_time_pairs(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    sync: ActivitySyncResult | None = None,
    peak_prominence: float = 0.12,
    pair_tolerance: float = 2.0,
    weight: float = 1.2,
) -> list[TimePair]:
    """Build visual evidence pairs from activity peaks under a global model.

    Peaks on the base envelope are mapped with ``donor ≈ speed * base + offset``
    and matched to the nearest donor peak within ``pair_tolerance`` seconds.
    """
    result = sync or estimate_speed_offset_activity(base, donor)
    base_peaks = activity_peak_times(base, min_prominence=peak_prominence)
    donor_peaks = activity_peak_times(donor, min_prominence=peak_prominence)
    if not base_peaks or not donor_peaks:
        # Fall back to dense envelope samples as soft pairs along the model.
        pairs: list[TimePair] = []
        step = max(1, len(base.times) // 40)
        for t in base.times[::step]:
            d = float(result.speed * t + result.offset_seconds)
            if donor.times.size and donor.times[0] - 1 <= d <= donor.times[-1] + 1:
                pairs.append(
                    TimePair(
                        base_time=float(t),
                        donor_time=d,
                        weight=weight * 0.5 * max(result.score, 0.1),
                        source="video_activity_model",
                    )
                )
        return pairs

    donor_arr = np.asarray(donor_peaks, dtype=np.float64)
    pairs = []
    for bt in base_peaks:
        expected = float(result.speed * bt + result.offset_seconds)
        j = int(np.argmin(np.abs(donor_arr - expected)))
        dt = float(donor_arr[j])
        if abs(dt - expected) <= pair_tolerance:
            pairs.append(
                TimePair(
                    base_time=float(bt),
                    donor_time=dt,
                    weight=weight * max(result.score, 0.2),
                    source="video_activity_peak",
                )
            )
    return pairs
