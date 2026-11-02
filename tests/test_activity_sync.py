"""Tests for visual activity envelopes and cross-correlation sync."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis_media_processor.alignment.activity_sync import (
    activity_time_pairs,
    estimate_offset_crosscorr,
    estimate_speed_offset_activity,
)
from jarvis_media_processor.video.activity import (
    ActivityEnvelope,
    activity_peak_times,
    build_activity_envelope,
)
from jarvis_media_processor.video.models import SampledFrame


def _frame(t: float, phash: str) -> SampledFrame:
    return SampledFrame(time_seconds=t, path=Path(f"{t:.3f}.png"), phash=phash)


def test_build_activity_envelope_spikes_on_hash_jumps() -> None:
    frames = [
        _frame(0.0, "0000000000000000"),
        _frame(1.0, "0000000000000000"),
        _frame(2.0, "ffffffffffffffff"),  # jump
        _frame(3.0, "ffffffffffffffff"),
        _frame(4.0, "0000000000000000"),  # jump
        _frame(5.0, "0000000000000000"),
    ]
    env = build_activity_envelope(frames, max_hash_distance=64)
    assert env.values[0] == 0.0
    assert env.values[1] == 0.0
    assert env.values[2] == 1.0
    assert env.values[4] == 1.0
    peaks = activity_peak_times(env, min_prominence=0.2)
    assert 2.0 in peaks
    assert 4.0 in peaks


def _pulse_envelope(
    length: int,
    pulse_at: list[int],
    *,
    period: float = 1.0,
) -> ActivityEnvelope:
    values = np.zeros(length, dtype=np.float64)
    for i in pulse_at:
        values[i] = 1.0
        if i + 1 < length:
            values[i + 1] = 0.4
    times = np.arange(length, dtype=np.float64) * period
    return ActivityEnvelope(times=times, values=values, sample_period=period)


def test_crosscorr_recovers_positive_offset() -> None:
    # Same pulse pattern; donor delayed by 5 s ⇒ offset = +5.
    base = _pulse_envelope(80, [10, 30, 55])
    donor = _pulse_envelope(80, [15, 35, 60])
    result = estimate_offset_crosscorr(base, donor)
    assert result.offset_seconds == pytest.approx(5.0, abs=1.0)
    assert result.score > 0.3


def test_crosscorr_recovers_negative_offset() -> None:
    base = _pulse_envelope(80, [20, 40, 60])
    donor = _pulse_envelope(80, [12, 32, 52])  # donor earlier ⇒ offset ≈ -8
    result = estimate_offset_crosscorr(base, donor)
    assert result.offset_seconds == pytest.approx(-8.0, abs=1.0)


def test_speed_grid_recovers_mild_compression() -> None:
    speed = 0.96
    offset = -5.0
    base_idx = [20, 40, 60, 80, 100]
    donor_idx = [int(round(speed * i + offset)) for i in base_idx]
    base = _pulse_envelope(120, base_idx)
    donor = _pulse_envelope(120, donor_idx)
    result = estimate_speed_offset_activity(
        base,
        donor,
        speed_min=0.92,
        speed_max=1.05,
        speed_steps=14,
    )
    assert abs(result.speed - speed) < 0.05
    assert abs(result.offset_seconds - offset) < 3.0


def test_activity_time_pairs_follow_sync_model() -> None:
    base = _pulse_envelope(100, [10, 30, 50, 70])
    donor = _pulse_envelope(100, [18, 38, 58, 78])  # +8 s
    sync = estimate_offset_crosscorr(base, donor)
    pairs = activity_time_pairs(base, donor, sync=sync, pair_tolerance=2.0)
    assert len(pairs) >= 3
    assert all(p.source.startswith("video_activity") for p in pairs)
    mean_off = float(np.mean([p.donor_time - p.base_time for p in pairs]))
    assert mean_off == pytest.approx(8.0, abs=1.5)
