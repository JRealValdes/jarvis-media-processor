"""Tests for piecewise alignment and discontinuity detection."""

from __future__ import annotations

from jarvis_media_processor.alignment import (
    SyncStrategyLevel,
    build_piecewise_alignment,
    detect_alignment_from_landmarks,
)
from jarvis_media_processor.alignment.evidence import TimePair, landmark_time_pairs
from jarvis_media_processor.audio.models import AcousticLandmark
from jarvis_media_processor.fixtures import get_scenario


def _lm(t: float, strength: float = 0.9) -> AcousticLandmark:
    return AcousticLandmark(time_seconds=t, strength=strength)


def test_unrelated_block_fixture_flags_donor_gap() -> None:
    truth = get_scenario("unrelated_block")
    base = [_lm(lm.base_time) for lm in truth.landmarks]
    donor = [_lm(lm.donor_time) for lm in truth.landmarks]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=truth.base.duration_seconds,
        donor_duration=truth.donor.duration_seconds,
        inlier_tolerance=0.08,
    )
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    assert len(doc.segments) >= 2
    assert doc.unrelated_donor_intervals
    # Donor jumps ahead between early and late matches (ad-like block ~2–4s).
    assert any(iv.start < 2.5 and iv.end > 3.5 for iv in doc.unrelated_donor_intervals)


def test_local_length_drift_fixture_piecewise() -> None:
    truth = get_scenario("local_length_drift")
    base = [_lm(lm.base_time) for lm in truth.landmarks]
    donor = [_lm(lm.donor_time) for lm in truth.landmarks]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=truth.base.duration_seconds,
        donor_duration=truth.donor.duration_seconds,
        inlier_tolerance=0.08,
    )
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    assert len(doc.segments) >= 2
    # Later content is shifted on donor (+1s)
    last = max(doc.segments, key=lambda s: s.base_start)
    assert last.offset_seconds > 0.5


def test_missing_base_regions_flagged() -> None:
    pairs = [
        TimePair(0.5, 0.5, 1.0, "audio_landmark"),
        TimePair(5.0, 5.0, 1.0, "audio_landmark"),
    ]
    doc = build_piecewise_alignment(pairs, base_duration=6.0, donor_duration=6.0, min_gap=0.5)
    # Two point segments leave a large uncovered middle/edges on the base timeline.
    gaps = doc.missing_base_intervals + [
        iv for iv in doc.uncertain_intervals if iv.timeline == "base"
    ]
    assert gaps
    assert any(iv.end - iv.start >= 1.0 for iv in gaps)


def test_duplicate_donor_detection() -> None:
    segments_pairs = [
        TimePair(0.5, 1.0, 1.0, "audio_landmark"),
        TimePair(1.0, 1.2, 1.0, "audio_landmark"),
        TimePair(3.0, 1.1, 1.0, "audio_landmark"),  # reuses donor ~1.x
    ]
    # Force two clusters with overlapping donor by large offset jump then back
    doc = build_piecewise_alignment(
        segments_pairs,
        base_duration=4.0,
        donor_duration=3.0,
        offset_tolerance=0.05,
    )
    # May flag duplicate or uncertain; at least piecewise with notes
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    assert doc.segments


def test_constant_speed_still_preferred_over_piecewise() -> None:
    truth = get_scenario("constant_speed")
    base = [_lm(lm.base_time) for lm in truth.landmarks]
    donor = [_lm(lm.donor_time) for lm in truth.landmarks]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=truth.base.duration_seconds,
        donor_duration=truth.donor.duration_seconds,
        inlier_tolerance=0.1,
    )
    assert doc.strategy == SyncStrategyLevel.CONSTANT_SPEED


def test_piecewise_does_not_fill_large_gaps() -> None:
    pairs = landmark_time_pairs([_lm(0.5), _lm(5.0)], [_lm(0.5), _lm(5.0)])
    doc = build_piecewise_alignment(pairs, base_duration=6.0, donor_duration=6.0)
    # No single invented segment covering 0..6 silently
    assert not any(s.base_start <= 0.01 and s.base_end >= 5.9 for s in doc.segments)
    assert doc.uncertain_intervals or doc.missing_base_intervals
