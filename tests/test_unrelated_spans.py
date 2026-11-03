"""Tests for curve-walk surplus-donor / missing-base detection and carving."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis_media_processor.alignment.curve_walk import (
    _find_forward_surplus,
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


def _donor_with_mid_speed_change() -> tuple[ActivityEnvelope, ActivityEnvelope, float]:
    """Pre matches at speed 1.0; after ~40s the same content continues at 0.92.

    Donor samples are interpolated from the base envelope (not snapped pulses),
    so a wrong linear speed diverges clearly over a multi-hop horizon.
    """
    true_speed = 0.92
    break_t = 40.0
    base = _pulse_envelope(140, list(range(2, 140, 3)))
    rem = float(base.times[-1] - break_t)
    donor_end = break_t + true_speed * rem + 2.0
    times = np.arange(0.0, donor_end + 0.5, base.sample_period)
    values = np.zeros_like(times)
    for i, t in enumerate(times):
        if t <= break_t:
            bq = float(t)
        else:
            bq = break_t + (float(t) - break_t) / true_speed
        values[i] = float(np.interp(bq, base.times, base.values, left=0.0, right=0.0))
    donor = ActivityEnvelope(
        times=times, values=values, sample_period=base.sample_period
    )
    return base, donor, true_speed


def test_curve_walk_refines_speed_instead_of_inventing_ad() -> None:
    """A mild mid-timeline break from wrong speed must not become surplus."""
    base, donor, true_speed = _donor_with_mid_speed_change()
    result = walk_activity_curves(
        base,
        donor,
        half_window=6.0,
        hop_seconds=2.0,
        min_score=0.25,
        break_score=0.18,
        confirm_seconds=6.0,
        min_surplus_seconds=8.0,
        max_surplus_seconds=80.0,
        missing_base_fill_seconds=3.0,
        # Force a wrong initial speed so the walk must refine mid-way.
        speed_candidates=(1.0,),
    )
    assert not result.unrelated_donor_intervals, (
        f"speed drift must not invent ads; got {result.unrelated_donor_intervals}"
    )
    assert any("Speed refine" in n for n in result.notes), result.notes
    refined = [lock for lock in result.locks if abs(lock.speed - true_speed) < 0.04]
    assert refined, f"expected a lock near {true_speed}; locks={result.locks}"


def test_curve_walk_still_finds_ad_when_speed_refine_cannot_help() -> None:
    """Real flat insert must still win over a bogus speed re-fit."""
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
    assert any(
        s.start <= ad0 + 10 and s.end >= ad1 - 10
        for s in result.unrelated_donor_intervals
    )


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


def test_forward_surplus_prefers_earliest_frame_supported_resume() -> None:
    """A weak true post-ad lock with frames must beat a louder later alias."""
    # Pre 0..40 match; flat insert 40..60; post resumes at donor 60 (= +20).
    # A second pulse train at donor 88 creates a high-score +48 alias.
    base = _pulse_envelope(100, list(range(2, 100, 4)))
    pre = list(range(2, 40, 4))
    post = [60 + (i - 40) for i in range(42, 100, 4)]
    alias = [88 + (i - 40) for i in range(42, 60, 4)]
    donor = _pulse_envelope(140, pre + post + alias)

    def frame_confirm(base_t: float, donor_t: float) -> bool:
        # True cut is base≈40 ↔ donor≈60; allow a couple hops of early break.
        return abs((donor_t - base_t) - 20.0) <= 1.0 and 38.0 <= base_t <= 46.0

    hit = _find_forward_surplus(
        base,
        donor,
        base_time=40.0,
        offset=0.0,
        speed=1.0,
        half_window=6.0,
        period=1.0,
        min_score=0.25,
        min_surplus_seconds=6.0,
        max_surplus_seconds=80.0,
        break_score=0.18,
        donor_start=0.0,
        donor_end=139.0,
        frame_confirm=frame_confirm,
        confirm_seconds=6.0,
        hop_seconds=2.0,
    )
    assert hit is not None
    gap_start, gap_end, new_off, _score = hit
    assert gap_start == pytest.approx(40.0, abs=1.0)
    assert gap_end == pytest.approx(60.0, abs=2.0)
    assert new_off == pytest.approx(20.0, abs=1.0)
    assert (gap_end - gap_start) < 35.0  # must not latch the +48 alias


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


def test_backwards_refine_preserves_lock_quality() -> None:
    """Backwards refine must not walk into a weak head and collapse the score."""
    from jarvis_media_processor.alignment.curve_walk import (
        CurveLock,
        _backwards_refine_lock,
    )

    # Matching pulse train only after t=40; earlier samples are near-zero noise
    # that can still clear a very low absolute floor.
    base = _pulse_envelope(100, list(range(42, 90, 3)))
    donor = _pulse_envelope(100, list(range(42, 90, 3)))
    lock = CurveLock(
        speed=1.0,
        offset_seconds=0.0,
        base_time=48.0,
        donor_time=48.0,
        score=1.0,
    )
    refined = _backwards_refine_lock(
        base,
        donor,
        lock,
        half_window=5.0,
        period=1.0,
        hop=2.0,
        min_score=0.2,
    )
    # May step one hop before the first pulse (half-window overlap) but must
    # not collapse into the empty head.
    assert refined.base_time >= 36.0
    assert refined.score >= 0.7


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
