"""Tests for escalating sync alignment."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_media_processor.alignment import (
    AlignmentDocument,
    SyncStrategyLevel,
    detect_alignment_from_landmarks,
    estimate_alignment,
    estimate_alignment_visual_first,
)
from jarvis_media_processor.alignment.detect import detect_alignment_from_artifacts
from jarvis_media_processor.alignment.evidence import TimePair, landmark_time_pairs
from jarvis_media_processor.audio.models import AcousticLandmark, AudioAnalysisArtifact
from jarvis_media_processor.fixtures import get_scenario
from jarvis_media_processor.video.models import FrameCorrespondence


def _lm(t: float, strength: float = 0.9) -> AcousticLandmark:
    return AcousticLandmark(time_seconds=t, strength=strength, kind="energy_peak")


def test_alignment_document_json_roundtrip() -> None:
    doc = detect_alignment_from_landmarks(
        [_lm(0.5), _lm(1.5)],
        [_lm(0.5), _lm(1.5)],
        base_duration=2.0,
        donor_duration=2.0,
    )
    restored = AlignmentDocument.model_validate_json(doc.model_dump_json())
    assert restored.strategy == doc.strategy
    assert restored.segments


def test_already_synced_from_landmarks() -> None:
    base = [_lm(0.5), _lm(2.5), _lm(4.5)]
    donor = [_lm(0.52), _lm(2.48), _lm(4.51)]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=6.0,
        donor_duration=6.0,
        inlier_tolerance=0.15,
    )
    assert doc.strategy == SyncStrategyLevel.ALREADY_SYNCED
    assert doc.global_offset_seconds is not None
    assert abs(doc.global_offset_seconds) < 0.1


def test_constant_offset_two_seconds() -> None:
    offset = 2.0
    base = [_lm(0.5), _lm(1.5), _lm(3.0)]
    donor = [_lm(0.5 + offset), _lm(1.5 + offset), _lm(3.0 + offset)]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=4.0,
        donor_duration=6.0,
        inlier_tolerance=0.12,
    )
    assert doc.strategy == SyncStrategyLevel.CONSTANT_OFFSET
    assert doc.global_offset_seconds is not None
    assert abs(doc.global_offset_seconds - offset) < 0.08
    # Positive offset: base starts at 0, donor starts at offset (no stretch).
    assert len(doc.segments) == 1
    seg = doc.segments[0]
    assert abs(seg.base_start) < 1e-6
    assert abs(seg.donor_start - offset) < 0.08
    assert abs((seg.base_end - seg.base_start) - (seg.donor_end - seg.donor_start)) < 0.05


def test_constant_offset_negative_leaves_leading_base_silence() -> None:
    offset = -2.0
    base = [_lm(2.5), _lm(3.5), _lm(4.5)]
    donor = [_lm(0.5), _lm(1.5), _lm(2.5)]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=5.0,
        donor_duration=4.0,
        inlier_tolerance=0.12,
    )
    assert doc.strategy == SyncStrategyLevel.CONSTANT_OFFSET
    assert doc.global_offset_seconds is not None
    assert abs(doc.global_offset_seconds - offset) < 0.08
    seg = doc.segments[0]
    assert seg.base_start == pytest.approx(-offset, abs=0.08)
    assert seg.donor_start == pytest.approx(0.0, abs=0.05)
    assert abs((seg.base_end - seg.base_start) - (seg.donor_end - seg.donor_start)) < 0.05


def test_constant_speed_hypothesis() -> None:
    # Donor is 1.1× faster ⇒ event times compress: donor_t = base_t / 1.1
    # Alignment model stores donor ≈ global_speed * base + offset.
    playback = 1.1
    linear = 1.0 / playback
    base_times = [0.5, 2.5, 4.5]
    base = [_lm(t) for t in base_times]
    donor = [_lm(t * linear) for t in base_times]
    pairs = landmark_time_pairs(base, donor)
    speed_doc = estimate_alignment(
        pairs,
        base_duration=6.0,
        donor_duration=6.0 * linear,
        already_synced_tolerance=0.01,
        inlier_tolerance=0.08,
        min_inlier_ratio=0.5,
    )
    assert speed_doc.strategy == SyncStrategyLevel.CONSTANT_SPEED
    assert speed_doc.global_speed is not None
    assert abs(speed_doc.global_speed - linear) < 0.05


def test_piecewise_gaps_only_outside_covered_spans() -> None:
    pairs = [
        TimePair(1.0, 1.0, 1.0, "audio_landmark"),
        TimePair(2.0, 2.0, 1.0, "audio_landmark"),
        TimePair(5.0, 6.0, 1.0, "audio_landmark"),
        TimePair(6.0, 7.0, 1.0, "audio_landmark"),
    ]
    doc = estimate_alignment(
        pairs,
        base_duration=8.0,
        donor_duration=9.0,
        inlier_tolerance=0.1,
        min_inlier_ratio=0.85,
    )
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    # Leading uncovered only — interiors of expanded segments must not be "missing".
    missing = doc.missing_base_intervals
    assert missing
    assert missing[0].start == pytest.approx(0.0, abs=0.05)
    assert all(interval.end <= doc.segments[0].base_start + 1e-6 for interval in missing[:1])
    for interval in missing:
        for seg in doc.segments:
            overlap = min(interval.end, seg.base_end) - max(interval.start, seg.base_start)
            assert overlap <= 1e-6


def test_piecewise_when_offset_changes() -> None:
    # First half offset 0, second half offset +1s — should not accept global offset.
    pairs = [
        TimePair(0.5, 0.5, 1.0, "audio_landmark"),
        TimePair(1.0, 1.0, 1.0, "audio_landmark"),
        TimePair(3.0, 4.0, 1.0, "audio_landmark"),
        TimePair(4.0, 5.0, 1.0, "audio_landmark"),
    ]
    doc = estimate_alignment(
        pairs,
        base_duration=5.0,
        donor_duration=6.0,
        inlier_tolerance=0.1,
        min_inlier_ratio=0.85,
    )
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    assert doc.uncertain_intervals or len(doc.segments) >= 2
    # Expanded spans must be long enough for reconstruction (not 1 ms stubs).
    assert all((s.base_end - s.base_start) >= 0.05 for s in doc.segments)
    assert doc.segments[0].base_start == pytest.approx(0.5, abs=0.05)
    assert doc.segments[-1].base_end == pytest.approx(4.0, abs=0.05)


def test_fixture_catalog_offset_scenario_expectation() -> None:
    truth = get_scenario("constant_offset")
    assert truth.global_offset_seconds == 1.0
    base = [_lm(lm.base_time) for lm in truth.landmarks]
    donor = [_lm(lm.donor_time) for lm in truth.landmarks]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=truth.base.duration_seconds,
        donor_duration=truth.donor.duration_seconds,
    )
    assert doc.strategy == SyncStrategyLevel.CONSTANT_OFFSET
    assert doc.global_offset_seconds is not None
    assert abs(doc.global_offset_seconds - 1.0) < 0.08


def test_fixture_already_synced_expectation() -> None:
    truth = get_scenario("already_synced")
    base = [_lm(lm.base_time) for lm in truth.landmarks]
    donor = [_lm(lm.donor_time) for lm in truth.landmarks]
    doc = detect_alignment_from_landmarks(
        base,
        donor,
        base_duration=truth.base.duration_seconds,
        donor_duration=truth.donor.duration_seconds,
    )
    assert doc.strategy == SyncStrategyLevel.ALREADY_SYNCED


def test_visual_first_accepts_speed_despite_conflicting_audio() -> None:
    # Visual: donor ≈ 0.96 * base - 28 (Spider-Man-like).
    visual = [
        TimePair(float(t), 0.96 * t - 28.0, 0.9, "video_match")
        for t in range(40, 500, 10)
    ]
    # Conflicting audio Cartesian noise around a wrong ~-50s offset.
    audio = [
        TimePair(float(t), float(t) - 50.0, 0.95, "audio_landmark")
        for t in (50, 100, 150, 200, 250)
    ] + [
        TimePair(float(t), float(t) - 48.0, 0.9, "audio_landmark")
        for t in (60, 110, 160, 210, 260)
    ]
    doc = estimate_alignment_visual_first(
        visual,
        audio,
        base_duration=600.0,
        donor_duration=580.0,
        visual_inlier_tolerance=1.5,
        min_visual_pairs=8,
    )
    assert doc.strategy == SyncStrategyLevel.CONSTANT_SPEED
    assert doc.global_speed is not None
    assert abs(doc.global_speed - 0.96) < 0.03
    assert doc.evidence.get("fusion") == "visual_only_global"
    assert any("Visual-first" in n for n in doc.notes)


def test_weak_audio_does_not_replace_strong_visual_offset() -> None:
    visual = [
        TimePair(float(t), float(t) - 8.0, 1.0, "video_shot_anchor")
        for t in range(10, 200, 5)
    ]
    # Dense wrong audio pairs that would win a naive merge.
    audio = [
        TimePair(b, d, 1.0, "audio_landmark")
        for b in (10.0, 20.0, 30.0, 40.0, 50.0)
        for d in (100.0, 110.0, 120.0, 130.0, 140.0)
    ]
    merged_naive = estimate_alignment(
        visual + audio,
        base_duration=220.0,
        donor_duration=220.0,
        inlier_tolerance=0.12,
    )
    visual_first = estimate_alignment_visual_first(
        visual,
        audio,
        base_duration=220.0,
        donor_duration=220.0,
        visual_inlier_tolerance=1.0,
    )
    assert visual_first.strategy == SyncStrategyLevel.CONSTANT_OFFSET
    assert visual_first.global_offset_seconds is not None
    assert abs(visual_first.global_offset_seconds - (-8.0)) < 0.2
    # Naive merge is allowed to fail or pick something else; visual-first must keep -8.
    assert visual_first.strategy != merged_naive.strategy or (
        merged_naive.global_offset_seconds is not None
        and abs(merged_naive.global_offset_seconds - (-8.0)) < 0.2
    )


def test_detect_from_artifacts_uses_visual_first() -> None:
    base_audio = AudioAnalysisArtifact(
        source_path=Path("base"),
        sample_rate=16_000,
        channels=1,
        duration_seconds=30.0,
        landmarks=[_lm(5.0), _lm(15.0)],
        energy_hop_seconds=0.01,
    )
    donor_audio = AudioAnalysisArtifact(
        source_path=Path("donor"),
        sample_rate=16_000,
        channels=1,
        duration_seconds=30.0,
        # Intentionally wrong acoustic times vs visual.
        landmarks=[_lm(20.0), _lm(25.0)],
        energy_hop_seconds=0.01,
    )
    visual = [
        FrameCorrespondence(
            base_time=float(t),
            donor_time=float(t) + 2.0,
            distance=0,
            confidence=1.0,
            shot_anchor=(t % 10 == 0),
        )
        for t in range(0, 28, 2)
    ]
    doc = detect_alignment_from_artifacts(
        base_audio,
        donor_audio,
        visual_matches=visual,
        visual_inlier_tolerance=1.0,
        min_visual_pairs=4,
    )
    assert doc.strategy in {
        SyncStrategyLevel.CONSTANT_OFFSET,
        SyncStrategyLevel.ALREADY_SYNCED,
    }
    assert doc.global_offset_seconds is not None
    assert abs(doc.global_offset_seconds - 2.0) < 0.25
