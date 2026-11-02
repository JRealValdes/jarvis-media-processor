"""Tests for post-visual fine audio-offset refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis_media_processor.alignment.audio_refine import (
    apply_offset_delta,
    estimate_fine_audio_offset,
    refine_alignment_audio_offset,
    warp_donor_envelope_to_base,
)
from jarvis_media_processor.alignment.models import (
    AlignmentDocument,
    AlignmentSegment,
    SyncStrategyLevel,
)
from jarvis_media_processor.audio.models import AudioAnalysisArtifact


def _artifact(env: np.ndarray, *, hop: float = 0.05) -> AudioAnalysisArtifact:
    return AudioAnalysisArtifact(
        source_path=Path("x"),
        sample_rate=16_000,
        channels=1,
        duration_seconds=float(len(env) * hop),
        landmarks=[],
        energy_hop_seconds=hop,
        energy_envelope=[float(x) for x in env.tolist()],
    )


def test_warp_donor_envelope_samples_model() -> None:
    donor = np.array([0.0, 1.0, 0.0, 0.5, 0.0], dtype=np.float64)
    # speed=1, offset=0.1, hop=0.1 → base index 0 maps to donor time 0.1 → idx 1
    warped = warp_donor_envelope_to_base(
        donor,
        donor_hop=0.1,
        base_length=3,
        base_hop=0.1,
        speed=1.0,
        offset=0.1,
    )
    assert warped[0] == pytest.approx(1.0, abs=0.05)


def test_fine_offset_recovers_residual_shift() -> None:
    hop = 0.05
    n = 400
    base = np.zeros(n, dtype=np.float64)
    for i in (40, 120, 220, 300):
        base[i] = 1.0
        base[i + 1] = 0.5
    # True map: donor = base + 1.0; visual lock used offset=1.2 (error +0.2)
    residual = 0.2
    visual_offset = 1.0 + residual
    donor = np.zeros(n, dtype=np.float64)
    shift = int(round(1.0 / hop))
    donor[shift:] = base[:-shift]

    refinement = estimate_fine_audio_offset(
        _artifact(base, hop=hop),
        _artifact(donor, hop=hop),
        speed=1.0,
        offset=visual_offset,
        window_seconds=1.0,
    )
    assert refinement.applied
    # Should nudge back toward -0.2 so visual_offset + delta ≈ 1.0
    assert refinement.delta_seconds == pytest.approx(-residual, abs=0.12)


def test_apply_offset_delta_updates_global_and_segments() -> None:
    doc = AlignmentDocument(
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.9,
        global_offset_seconds=-28.0,
        global_speed=1.0,
        segments=[
            AlignmentSegment(
                base_start=10.0,
                base_end=20.0,
                donor_start=0.0,
                donor_end=10.0,
                offset_seconds=-10.0,
                confidence=0.9,
            )
        ],
    )
    updated = apply_offset_delta(doc, -0.5)
    assert updated.global_offset_seconds == pytest.approx(-28.5)
    assert updated.segments[0].offset_seconds == pytest.approx(-10.5)
    assert updated.segments[0].donor_start == pytest.approx(-0.5) or updated.segments[
        0
    ].donor_start == pytest.approx(0.0)


def test_refine_skips_piecewise() -> None:
    doc = AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=0.6,
        segments=[
            AlignmentSegment(
                base_start=0.0,
                base_end=1.0,
                donor_start=0.0,
                donor_end=1.0,
                confidence=0.6,
            )
        ],
    )
    env = np.ones(50, dtype=np.float64)
    out, refinement = refine_alignment_audio_offset(
        doc,
        _artifact(env),
        _artifact(env),
        window_seconds=1.0,
    )
    assert not refinement.applied
    assert out.strategy == SyncStrategyLevel.PIECEWISE
