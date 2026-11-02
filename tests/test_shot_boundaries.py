"""Tests for shot-cut detection and shot-anchor boosting."""

from __future__ import annotations

from pathlib import Path

from jarvis_media_processor.video.match import match_frames
from jarvis_media_processor.video.models import SampledFrame
from jarvis_media_processor.video.shots import detect_shot_boundaries, shot_boundary_times


def _frame(t: float, phash: str) -> SampledFrame:
    return SampledFrame(time_seconds=t, path=Path(f"{t:.0f}.png"), phash=phash, role="base")


def test_detect_shot_boundaries_on_abrupt_phash_jump() -> None:
    # Consecutive identical hashes → no cut; then a large jump.
    frames = [
        _frame(0.0, "0000000000000000"),
        _frame(1.0, "0000000000000000"),
        _frame(2.0, "ffffffffffffffff"),  # abrupt change
        _frame(3.0, "ffffffffffffffff"),
    ]
    cuts = detect_shot_boundaries(frames, min_distance=20)
    assert len(cuts) == 1
    assert cuts[0].time_seconds == 2.0
    assert shot_boundary_times(frames, min_distance=20) == {2.0}


def test_match_frames_marks_and_boosts_shot_anchors() -> None:
    base = [
        _frame(0.0, "0000000000000000"),
        _frame(1.0, "ffffffffffffffff"),  # shot start on base
    ]
    donor = [
        SampledFrame(
            time_seconds=0.5,
            path=Path("d0.png"),
            phash="0000000000000000",
            role="donor",
        ),
        SampledFrame(
            time_seconds=1.5,
            path=Path("d1.png"),
            phash="ffffffffffffffff",
            role="donor",
        ),
    ]
    matches = match_frames(
        base,
        donor,
        max_distance=4,
        max_matches_per_base=1,
        base_shot_times={1.0},
        donor_shot_times={1.5},
        shot_confidence_boost=0.15,
    )
    shot_matches = [m for m in matches if m.shot_anchor]
    assert shot_matches
    assert all(m.confidence >= 0.15 for m in shot_matches)
