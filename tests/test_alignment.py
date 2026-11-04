"""Tests for escalating sync alignment."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_media_processor.alignment import (
    AlignmentDocument,
    ChunkTimingHint,
    SyncStrategyLevel,
    detect_alignment_from_landmarks,
    estimate_alignment,
    estimate_alignment_per_donor_chunk,
    estimate_alignment_visual_first,
)
from jarvis_media_processor.alignment.estimate import (
    bridge_surplus_donor_gaps,
    fit_speed_offset,
    subtract_base_intervals,
)
from jarvis_media_processor.alignment.detect import detect_alignment_from_artifacts
from jarvis_media_processor.alignment.evidence import TimePair, landmark_time_pairs
from jarvis_media_processor.alignment.models import AlignmentSegment, TimelineInterval
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
    # Generic hard case: mild timeline compression + large constant offset.
    speed = 0.96
    offset = -28.0
    visual = [
        TimePair(float(t), speed * t + offset, 0.9, "video_match")
        for t in range(40, 500, 10)
    ]
    # Conflicting acoustic noise around a wrong constant offset.
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
    assert abs(doc.global_speed - speed) < 0.03
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


def test_per_donor_chunk_recovers_dual_offsets_around_ad() -> None:
    """Mid-roll ad ⇒ two legitimate offsets; chunked fit should keep both."""
    pre_offset = -28.0
    post_offset = -4.0
    ad = (180.0, 204.0)
    visual = [
        TimePair(float(t), float(t) + pre_offset, 0.95, "video_match")
        for t in range(40, 180, 8)
    ] + [
        TimePair(float(t), float(t) + post_offset, 0.95, "video_match")
        for t in range(210, 520, 8)
    ]
    keep = [(0.0, ad[0]), (ad[1], 600.0)]
    doc = estimate_alignment_per_donor_chunk(
        visual,
        keep,
        base_duration=600.0,
        donor_duration=624.0,
        inlier_tolerance=1.5,
        min_inlier_ratio=0.55,
        min_chunk_pairs=6,
    )
    assert doc is not None
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    assert int(doc.evidence.get("fitted_chunks", 0)) == 2
    offsets = sorted(seg.offset_seconds for seg in doc.segments)
    assert abs(offsets[0] - pre_offset) < 0.5
    assert abs(offsets[1] - post_offset) < 0.5


def test_per_donor_chunk_falls_back_to_phash_when_activity_pollutes() -> None:
    """Wrong activity offsets in a chunk must not block a clean pHash fit."""
    pre_offset = -28.0
    post_offset = -4.0
    visual = [
        TimePair(float(t), float(t) + pre_offset, 0.95, "video_match")
        for t in range(40, 180, 8)
    ] + [
        TimePair(float(t), float(t) + post_offset, 0.95, "video_match")
        for t in range(210, 520, 8)
    ] + [
        TimePair(float(t), 0.96 * float(t) - 11.0, 1.2, "video_activity_peak")
        for t in range(220, 500, 8)
    ]
    doc = estimate_alignment_per_donor_chunk(
        visual,
        [(0.0, 180.0), (204.0, 600.0)],
        base_duration=600.0,
        donor_duration=624.0,
        inlier_tolerance=1.5,
        min_chunk_pairs=6,
    )
    assert doc is not None
    assert int(doc.evidence.get("fitted_chunks", 0)) == 2
    offsets = sorted(seg.offset_seconds for seg in doc.segments)
    assert abs(offsets[0] - pre_offset) < 0.5
    assert abs(offsets[1] - post_offset) < 0.5
    assert any("pHash-only" in n for n in doc.notes)


def test_visual_first_uses_chunked_fit_when_donor_exclude_set() -> None:
    pre_offset = -28.0
    post_offset = -4.0
    visual = [
        TimePair(float(t), float(t) + pre_offset, 0.95, "video_match")
        for t in range(40, 180, 8)
    ] + [
        # Contaminating pairs whose donor times sit inside the excluded ad.
        TimePair(100.0 + i, 182.0 + i, 0.9, "video_match")
        for i in range(0, 20, 5)
    ] + [
        TimePair(float(t), float(t) + post_offset, 0.95, "video_match")
        for t in range(210, 520, 8)
    ]
    audio = [
        TimePair(float(t), float(t) - 50.0, 0.95, "audio_landmark")
        for t in (50, 100, 150, 250, 300)
    ]
    doc = estimate_alignment_visual_first(
        visual,
        audio,
        base_duration=600.0,
        donor_duration=624.0,
        visual_inlier_tolerance=1.5,
        min_visual_pairs=8,
        donor_exclude=[(180.0, 204.0)],
    )
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    assert doc.evidence.get("fusion") == "visual_chunked_exclusions"
    assert int(doc.evidence.get("fitted_chunks", 0)) == 2
    offsets = sorted(seg.offset_seconds for seg in doc.segments)
    assert abs(offsets[0] - pre_offset) < 0.5
    assert abs(offsets[1] - post_offset) < 0.5


def test_failed_chunk_recovers_via_sibling_model() -> None:
    """A keep-chunk with no global fit borrows the nearest fitted sibling."""
    post_offset = -3.2
    post_speed = 0.96
    # Chunk 0: noisy / contradictory pairs that cannot accept a global model.
    noisy = [
        TimePair(float(t), float((t % 17) * 3), 0.9, "video_match")
        for t in range(10, 160, 8)
    ]
    # Chunk 1: clean speed/offset map after an unrelated surplus gap.
    clean = [
        TimePair(
            float(t),
            post_speed * float(t) + post_offset,
            0.95,
            "video_match",
        )
        for t in range(220, 520, 8)
    ]
    # Bad head lock that must NOT win when a sibling exists.
    bad_lock = ChunkTimingHint(
        speed=1.0,
        offset_seconds=-2.0,
        score=0.9,
        donor_time=12.0,
    )
    doc = estimate_alignment_per_donor_chunk(
        noisy + clean,
        [(0.0, 180.0), (204.0, 600.0)],
        base_duration=600.0,
        donor_duration=624.0,
        inlier_tolerance=1.5,
        min_inlier_ratio=0.55,
        min_chunk_pairs=6,
        timing_hints=[bad_lock],
    )
    assert doc is not None
    assert doc.strategy == SyncStrategyLevel.PIECEWISE
    assert int(doc.evidence.get("recovered_chunks", 0)) == 1
    assert len(doc.segments) == 2
    early = min(doc.segments, key=lambda s: s.donor_start)
    assert early.donor_start == pytest.approx(0.0)
    assert early.donor_end == pytest.approx(180.0)
    assert early.speed == pytest.approx(post_speed, abs=0.02)
    assert early.offset_seconds == pytest.approx(post_offset, abs=0.5)
    assert any("sibling chunk model" in n for n in doc.notes)
    # Sibling preferred over the (wrong) head lock.
    assert not any("curve_walk_lock" in n for n in doc.notes)


def test_failed_chunk_recovers_via_best_lock_hint() -> None:
    """Without a sibling fit, use the highest-score lock inside the keep."""
    # Only one keep-chunk, and its pairs are pure noise → no global fit.
    noisy = [
        TimePair(float(t), float((t % 13) * 5), 0.9, "video_match")
        for t in range(10, 200, 8)
    ]
    weak_head = ChunkTimingHint(
        speed=1.0,
        offset_seconds=-2.0,
        score=0.4,
        donor_time=12.0,
    )
    strong_mid = ChunkTimingHint(
        speed=0.96,
        offset_seconds=-3.2,
        score=0.85,
        donor_time=100.0,
    )
    doc = estimate_alignment_per_donor_chunk(
        noisy,
        [(0.0, 220.0)],
        base_duration=600.0,
        donor_duration=624.0,
        inlier_tolerance=1.5,
        min_chunk_pairs=6,
        timing_hints=[weak_head, strong_mid],
    )
    assert doc is not None
    assert int(doc.evidence.get("recovered_chunks", 0)) == 1
    assert len(doc.segments) == 1
    seg = doc.segments[0]
    assert seg.speed == pytest.approx(0.96, abs=0.01)
    assert seg.offset_seconds == pytest.approx(-3.2, abs=0.05)
    assert any("curve_walk_lock" in n for n in doc.notes)


def test_failed_chunk_skipped_without_sibling_or_hint() -> None:
    """No sibling and no lock hint ⇒ keep previous skip behavior."""
    noisy_a = [
        TimePair(float(t), float((t % 11) * 7), 0.9, "video_match")
        for t in range(10, 160, 8)
    ]
    noisy_b = [
        TimePair(float(t), float((t % 9) * 6), 0.9, "video_match")
        for t in range(220, 400, 8)
    ]
    # Single failing keep, no hints → nothing to emit.
    assert (
        estimate_alignment_per_donor_chunk(
            noisy_a,
            [(0.0, 180.0)],
            base_duration=600.0,
            donor_duration=200.0,
            inlier_tolerance=1.5,
            min_chunk_pairs=6,
            timing_hints=None,
        )
        is None
    )
    # Two failing keeps, no hints → still nothing.
    assert (
        estimate_alignment_per_donor_chunk(
            noisy_a + noisy_b,
            [(0.0, 180.0), (204.0, 420.0)],
            base_duration=600.0,
            donor_duration=450.0,
            inlier_tolerance=1.5,
            min_chunk_pairs=6,
        )
        is None
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


def test_estimate_prefers_speed_model_over_speed1_approximation() -> None:
    """A short window must not accept donor≈base-19@1.0 when 0.96 fits better."""
    # True map: donor = 0.96 * base + 2.5. Over base 480..600 that also looks
    # roughly like offset -19 at speed 1.0 — offset-first escalation used to win.
    pairs = [
        TimePair(float(t), 0.96 * float(t) + 2.5, 0.95, "video_match")
        for t in range(480, 600, 4)
    ]
    doc = estimate_alignment(
        pairs,
        base_duration=600.0,
        donor_duration=630.0,
        inlier_tolerance=1.5,
        min_inlier_ratio=0.55,
        speed_tolerance=0.03,
    )
    assert doc.strategy == SyncStrategyLevel.CONSTANT_SPEED
    assert doc.global_speed is not None
    assert abs(doc.global_speed - 0.96) < 0.02
    assert doc.global_offset_seconds is not None
    assert abs(doc.global_offset_seconds - 2.5) < 1.0


def test_fit_speed_offset_recovers_non_grid_retime() -> None:
    """Mild retimes (e.g. 23.976/25) must not quantize to the nearest 0.01 grid."""
    true_speed = 23.976 / 25.0  # ≈ 0.95904 — between coarse 0.95 and 0.96
    true_offset = -1.05
    pairs = [
        TimePair(float(t), true_speed * float(t) + true_offset, 0.95, "video_match")
        for t in range(500, 3500, 10)
    ]
    fit = fit_speed_offset(pairs, inlier_tolerance=1.5)
    assert fit is not None
    assert abs(fit.speed - true_speed) < 0.001
    assert abs(fit.offset - true_offset) < 0.05
    # Must not snap to the coarse 0.01 grid (e.g. exact 0.96).
    assert abs(fit.speed - 0.96) > 0.0005


def test_long_chunk_splits_when_single_affine_cannot_explain() -> None:
    """Two retimes in one keep-chunk: split instead of one compromised map."""
    # First half: offset map; second half: 0.96 speed — whole-chunk global fails,
    # halves succeed. Title-agnostic recovery via split (not a PAL-only knob).
    pairs = [
        TimePair(float(t), float(t) - 2.0, 0.95, "video_match")
        for t in range(40, 400, 8)
    ] + [
        TimePair(float(t), 0.96 * float(t) + 20.0, 0.95, "video_match")
        for t in range(420, 900, 8)
    ]
    doc = estimate_alignment_per_donor_chunk(
        pairs,
        [(0.0, 950.0)],
        base_duration=1000.0,
        donor_duration=950.0,
        inlier_tolerance=1.5,
        min_inlier_ratio=0.55,
        min_chunk_pairs=6,
    )
    assert doc is not None
    assert len(doc.segments) >= 2
    assert any("splitting" in n for n in doc.notes)


def test_lock_for_donor_chunk_prefers_resume_near_chunk_start() -> None:
    """Later speed-refine locks must not outrank the post-gap resume lock."""
    from jarvis_media_processor.alignment.curve_walk import CurveLock
    from jarvis_media_processor.alignment.detect import _lock_for_donor_chunk

    resume = CurveLock(
        speed=0.92,
        offset_seconds=20.84,
        base_time=477.0,
        donor_time=459.7,
        score=0.56,
    )
    drifted = CurveLock(
        speed=1.02,
        offset_seconds=-30.5,
        base_time=551.0,
        donor_time=531.5,
        score=0.78,
    )
    picked = _lock_for_donor_chunk([resume, drifted], 459.7, 629.0)
    assert picked is resume

    head = CurveLock(
        speed=0.96,
        offset_seconds=-28.0,
        base_time=42.0,
        donor_time=12.0,
        score=0.4,
    )
    late_pre = CurveLock(
        speed=0.92,
        offset_seconds=-10.0,
        base_time=447.0,
        donor_time=401.0,
        score=0.71,
    )
    picked_pre = _lock_for_donor_chunk([head, late_pre], 0.0, 426.7)
    assert picked_pre is head


def test_bridge_surplus_when_neighbor_affines_agree() -> None:
    """False surplus that maps onto the base hole under both neighbors is stitched."""
    # Abstract geometry: different neighbor offsets, but mapped endpoints for the
    # excluded donor span agree within the bridge tolerance (~2s).
    d0, d1 = 1000.0, 1050.0
    left = AlignmentSegment(
        base_start=800.0,
        base_end=1046.9,
        donor_start=760.0,
        donor_end=d0,
        speed=0.96,
        offset_seconds=-5.0,
        confidence=0.65,
        reliable=True,
    )
    right = AlignmentSegment(
        base_start=1099.0,
        base_end=1400.0,
        donor_start=d1,
        donor_end=1340.0,
        speed=0.9552,
        offset_seconds=0.2,
        confidence=0.91,
        reliable=True,
    )
    surplus = [(d0, d1)]
    bridges, recovered, notes = bridge_surplus_donor_gaps(
        [left, right],
        surplus,
        base_duration=1500.0,
    )
    assert len(bridges) == 1
    assert recovered == [(d0, d1)]
    bridge = bridges[0]
    assert bridge.donor_start == pytest.approx(d0)
    assert bridge.donor_end == pytest.approx(d1)
    assert bridge.base_start == pytest.approx(1047.0, abs=2.0)
    assert bridge.base_end == pytest.approx(1099.0, abs=2.0)
    assert any("Bridged surplus donor" in n for n in notes)
    assert any("Neighbor offsets differ" in n for n in notes)

    missing = [
        TimelineInterval(
            start=1060.0,
            end=1110.0,
            timeline="base",
            reason="curve_break_missing_base",
        )
    ]
    trimmed = subtract_base_intervals(
        missing,
        [(bridge.base_start, bridge.base_end)],
    )
    # Overlap with the bridge is removed; a short tail after the bridge may remain.
    assert all(item.end > bridge.base_end - 1e-6 or item.start >= bridge.base_end for item in trimmed)
    assert not any(item.start < bridge.base_end - 1.0 and item.end > bridge.base_start + 1.0 for item in trimmed if item.end - item.start > 5)


def test_bridge_surplus_skipped_when_neighbor_affines_disagree() -> None:
    """True mid-roll ads: neighbors disagree on mapped base → do not bridge."""
    left = AlignmentSegment(
        base_start=0.0,
        base_end=100.0,
        donor_start=0.0,
        donor_end=100.0,
        speed=1.0,
        offset_seconds=0.0,
        confidence=0.9,
        reliable=True,
    )
    # Post-ad chunk resumes ~30s later on base for the same donor clock style.
    right = AlignmentSegment(
        base_start=100.0,
        base_end=200.0,
        donor_start=130.0,
        donor_end=230.0,
        speed=1.0,
        offset_seconds=30.0,
        confidence=0.9,
        reliable=True,
    )
    bridges, recovered, notes = bridge_surplus_donor_gaps(
        [left, right],
        [(100.0, 130.0)],
        base_duration=200.0,
    )
    assert bridges == []
    assert recovered == []
    assert notes == []


def test_detect_filters_bridged_surplus_from_unrelated() -> None:
    """End-to-end: agree-bridge removes surplus from unrelated before carve."""
    speed = 0.96
    # Keep 0: donor 0-100 → base via donor = speed*base + 0
    pairs_a = [
        TimePair(float(t), speed * float(t), 0.95, "video_match")
        for t in range(5, 95, 5)
    ]
    # Keep 1 after a 20s surplus gap: same affine (false surplus).
    pairs_b = [
        TimePair(float(t), speed * float(t), 0.95, "video_match")
        for t in range(125, 220, 5)
    ]
    base = AudioAnalysisArtifact(
        source_path=Path("base.mkv"),
        duration_seconds=250.0,
        sample_rate=48_000,
        channels=1,
        landmarks=[],
        energy_hop_seconds=0.25,
    )
    donor = AudioAnalysisArtifact(
        source_path=Path("donor.avi"),
        duration_seconds=240.0,
        sample_rate=48_000,
        channels=1,
        landmarks=[],
        energy_hop_seconds=0.25,
    )
    visuals = [
        FrameCorrespondence(
            base_time=p.base_time,
            donor_time=p.donor_time,
            distance=2,
            confidence=0.95,
        )
        for p in pairs_a + pairs_b
    ]
    unrelated = [
        TimelineInterval(
            start=100.0,
            end=120.0,
            timeline="donor",
            reason="curve_break_surplus_donor",
        )
    ]
    missing = [
        TimelineInterval(
            start=104.0,
            end=125.0,
            timeline="base",
            reason="curve_break_missing_base",
        )
    ]
    doc = detect_alignment_from_artifacts(
        base,
        donor,
        visual_matches=visuals,
        activity_pairs=[],
        unrelated_donor_intervals=unrelated,
        missing_base_intervals=missing,
        refine_audio_offset=False,
    )
    assert int(doc.evidence.get("bridged_surplus_chunks", 0)) >= 1
    assert not any(
        abs(item.start - 100.0) < 0.1 and abs(item.end - 120.0) < 0.1
        for item in doc.unrelated_donor_intervals
    )
    assert any(abs(seg.donor_start - 100.0) < 0.5 for seg in doc.segments)
    # Bridged base coverage should carve the overlapping missing-base region.
    assert not any(
        item.start < 110.0 < item.end for item in doc.missing_base_intervals
    )
