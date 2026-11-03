"""Tests for curve-walk surplus-donor / missing-base detection and carving."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis_media_processor.alignment.curve_walk import (
    find_initial_lock,
    walk_activity_curves,
)
from jarvis_media_processor.alignment.evidence import (
    TimePair,
    filter_pairs_excluding_donor,
)
from jarvis_media_processor.alignment.models import (
    AlignmentDocument,
    AlignmentSegment,
    SyncStrategyLevel,
    TimelineInterval,
)
from jarvis_media_processor.alignment.unrelated import (
    apply_unrelated_to_document,
    split_segment_excluding_donor,
)
from jarvis_media_processor.reconstruct.plan import plan_placements
from jarvis_media_processor.video.activity import ActivityEnvelope


def _pulse_envelope(
    length: int,
    pulse_at: list[int],
    *,
    period: float = 1.0,
) -> ActivityEnvelope:
    values = np.zeros(length, dtype=np.float64)
    for i in pulse_at:
        if 0 <= i < length:
            values[i] = 1.0
            if i + 1 < length:
                values[i + 1] = 0.4
    times = np.arange(length, dtype=np.float64) * period
    return ActivityEnvelope(times=times, values=values, sample_period=period)


def _donor_with_ad_block() -> tuple[ActivityEnvelope, ActivityEnvelope, tuple[float, float]]:
    """Matching pulse trains with a flat mid insert on the donor (+offset after)."""
    base_pulses = list(range(2, 80, 4))
    base = _pulse_envelope(80, base_pulses)
    pre = list(range(2, 40, 4))
    post = [60 + (i - 40) for i in range(42, 80, 4)]
    donor = _pulse_envelope(100, pre + post)  # 40..60 left flat
    return base, donor, (40.0, 60.0)


def test_curve_walk_finds_inserted_ad_block() -> None:
    base, donor, (ad0, ad1) = _donor_with_ad_block()
    result = walk_activity_curves(
        base,
        donor,
        half_window=6.0,
        hop_seconds=2.0,
        min_score=0.25,
        break_score=0.18,
        confirm_seconds=4.0,
        min_surplus_seconds=6.0,
        max_surplus_seconds=80.0,
        missing_base_fill_seconds=3.0,
    )
    assert result.unrelated_donor_intervals, "expected surplus donor span"
    # Flat insert is ~40..60; allow some walk/confirm slack on the edges.
    assert any(
        s.start <= ad0 + 10 and s.end >= ad1 - 10
        for s in result.unrelated_donor_intervals
    )


def test_curve_walk_prefers_long_surplus_over_short_false_peak() -> None:
    """A brief mid-insert activity coincidence must not truncate a long ad."""
    # Same shape as the flat-ad fixture, but the insert is longer and contains a
    # single decoy peak that would look like a +12s resume if score-only.
    base_pulses = list(range(2, 80, 4))
    base = _pulse_envelope(80, base_pulses)
    pre = list(range(2, 40, 4))
    decoy = [52]
    post = [80 + (i - 40) for i in range(42, 80, 4)]
    donor = _pulse_envelope(120, pre + decoy + post)
    result = walk_activity_curves(
        base,
        donor,
        half_window=6.0,
        hop_seconds=2.0,
        min_score=0.25,
        break_score=0.18,
        confirm_seconds=4.0,
        min_surplus_seconds=6.0,
        max_surplus_seconds=100.0,
        missing_base_fill_seconds=3.0,
    )
    assert result.unrelated_donor_intervals, "expected surplus donor span"
    # True insert is ~40s; a short false resume ending near the decoy (~52) must
    # not win over the full jump (surplus end near the real post train).
    assert any(
        (s.end - s.start) >= 30 and s.end >= 75
        for s in result.unrelated_donor_intervals
    )
    assert not any(
        s.end <= 60 and (s.end - s.start) < 25
        for s in result.unrelated_donor_intervals
    )


def test_initial_lock_takes_earliest_confirm_not_max_score() -> None:
    """A later louder peak against the donor head must not hide an early lock."""
    # Shared early pulse train (true lock ~ offset 0). Later, a single huge
    # base spike that correlates with donor head by chance at a wrong offset.
    base_vals = np.zeros(100, dtype=np.float64)
    donor_vals = np.zeros(100, dtype=np.float64)
    for i in list(range(2, 30, 3)):
        base_vals[i] = 1.0
        base_vals[i + 1] = 0.45
        donor_vals[i] = 1.0
        donor_vals[i + 1] = 0.45
    base_vals[70] = 1.0
    base_vals[71] = 1.0
    base_vals[72] = 1.0
    donor_vals[6] = 1.0  # donor-head-ish energy for the false late correlation
    base = ActivityEnvelope(
        times=np.arange(100, dtype=np.float64),
        values=base_vals,
        sample_period=1.0,
    )
    donor = ActivityEnvelope(
        times=np.arange(100, dtype=np.float64),
        values=donor_vals,
        sample_period=1.0,
    )
    lock = find_initial_lock(
        base,
        donor,
        half_window=5.0,
        min_score=0.25,
        max_offset_seconds=80.0,
        speed_candidates=(1.0, 0.96),
    )
    assert lock is not None
    assert lock.base_time < 40.0
    assert abs(lock.offset_seconds) < 15.0


def test_curve_lock_ignores_failed_frame_confirm() -> None:
    """Initial lock is curve-based; failed pHash must not block cross-encode."""
    base, donor, _ = _donor_with_ad_block()
    result = walk_activity_curves(
        base,
        donor,
        half_window=6.0,
        hop_seconds=2.0,
        min_score=0.25,
        break_score=0.18,
        confirm_seconds=4.0,
        min_surplus_seconds=6.0,
        max_surplus_seconds=80.0,
        frame_confirm=lambda _b, _d: False,
    )
    assert result.locks, "expected curve lock"
    assert result.unrelated_donor_intervals, "expected surplus ad span"


def test_clean_envelopes_have_no_surplus() -> None:
    base = _pulse_envelope(120, list(range(2, 120, 3)))
    donor = _pulse_envelope(120, list(range(7, 125, 3)))  # +5 offset
    result = walk_activity_curves(
        base,
        donor,
        half_window=6.0,
        hop_seconds=2.0,
        min_score=0.25,
        confirm_seconds=6.0,
        min_surplus_seconds=12.0,
        max_surplus_seconds=40.0,
    )
    assert result.unrelated_donor_intervals == []


def test_missing_base_extra_scene_detected() -> None:
    """Base has an extra mid scene absent from donor."""
    # Head shared; base-only coded burst 40..52; tail shared (donor has no burst).
    base_vals = np.zeros(90, dtype=np.float64)
    donor_vals = np.zeros(78, dtype=np.float64)
    for i in list(range(2, 40, 4)):
        base_vals[i] = 1.0
        base_vals[i + 1] = 0.4
        donor_vals[i] = 1.0
        donor_vals[i + 1] = 0.4
    # Unique base-only Morse-like burst (not present on donor).
    for i, amp in zip(range(40, 52), [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0], strict=True):
        base_vals[i] = float(amp)
    for i in range(52, 90, 4):
        base_vals[i] = 1.0
        base_vals[i + 1] = 0.4
        di = i - 12
        if 0 <= di < 78:
            donor_vals[di] = 1.0
            if di + 1 < 78:
                donor_vals[di + 1] = 0.4
    base = ActivityEnvelope(
        times=np.arange(90, dtype=np.float64),
        values=base_vals,
        sample_period=1.0,
    )
    donor = ActivityEnvelope(
        times=np.arange(78, dtype=np.float64),
        values=donor_vals,
        sample_period=1.0,
    )
    result = walk_activity_curves(
        base,
        donor,
        half_window=5.0,
        hop_seconds=1.0,
        min_score=0.25,
        break_score=0.18,
        confirm_seconds=3.0,
        min_surplus_seconds=8.0,
        max_surplus_seconds=40.0,
        missing_base_fill_seconds=3.0,
    )
    assert result.missing_base_intervals, "expected a missing-base gap"
    assert any(s.end - s.start >= 3.0 for s in result.missing_base_intervals)
    assert any(s.start <= 50 and s.end >= 42 for s in result.missing_base_intervals)


def test_filter_pairs_excluding_donor() -> None:
    pairs = [
        TimePair(1.0, 1.0, 1.0, "video_match"),
        TimePair(2.0, 5.0, 1.0, "video_match"),
        TimePair(3.0, 9.0, 1.0, "video_match"),
    ]
    kept = filter_pairs_excluding_donor(pairs, [(4.0, 6.0)])
    assert [p.donor_time for p in kept] == [1.0, 9.0]


def test_split_segment_excludes_ad_and_plan_leaves_silence() -> None:
    segment = AlignmentSegment(
        base_start=0.0,
        base_end=10.0,
        donor_start=0.0,
        donor_end=10.0,
        speed=1.0,
        offset_seconds=0.0,
        confidence=0.9,
    )
    exclude = [
        TimelineInterval(
            start=3.0, end=5.0, timeline="donor", reason="curve_break_surplus_donor"
        )
    ]
    parts = split_segment_excluding_donor(segment, exclude)
    assert len(parts) == 2
    assert parts[0].donor_end == pytest.approx(3.0)
    assert parts[1].donor_start == pytest.approx(5.0)

    doc = AlignmentDocument(
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.9,
        global_offset_seconds=0.0,
        global_speed=1.0,
        segments=[segment],
        unrelated_donor_intervals=exclude,
    )
    placements, _skipped = plan_placements(
        doc, min_segment_seconds=0.05, crossfade_seconds=0.0
    )
    assert len(placements) == 2
    covered = sorted((p.base_start, p.base_end) for p in placements)
    assert covered[0][1] <= 3.0 + 1e-6
    assert covered[1][0] >= 5.0 - 1e-6


def test_apply_unrelated_records_missing_base() -> None:
    doc = AlignmentDocument(
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.8,
        segments=[
            AlignmentSegment(
                base_start=0.0,
                base_end=6.0,
                donor_start=0.0,
                donor_end=6.0,
                confidence=0.8,
            )
        ],
    )
    apply_unrelated_to_document(
        doc,
        [
            TimelineInterval(
                start=2.0,
                end=4.0,
                timeline="donor",
                reason="curve_break_surplus_donor",
            )
        ],
        missing_base=[
            TimelineInterval(
                start=10.0,
                end=14.0,
                timeline="base",
                reason="curve_break_missing_base",
            )
        ],
    )
    assert len(doc.unrelated_donor_intervals) == 1
    assert len(doc.missing_base_intervals) == 1
    assert len(doc.segments) == 2
