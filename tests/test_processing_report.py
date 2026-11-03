"""Tests for sync validation and processing reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis_media_processor.alignment.models import (
    AlignmentDocument,
    AlignmentSegment,
    SyncStrategyLevel,
    TimelineInterval,
)
from jarvis_media_processor.audio.wavutil import write_pcm16_wav
from jarvis_media_processor.config import Settings
from jarvis_media_processor.reconstruct.models import Placement, ReconstructionResult
from jarvis_media_processor.reconstruct.pipeline import reconstruct_from_wav
from jarvis_media_processor.report.build import (
    build_processing_report,
    write_processing_report,
)
from jarvis_media_processor.report.models import FindingSeverity
from jarvis_media_processor.report.validate import (
    base_coverage,
    duration_mismatch_finding,
    placement_overlaps,
)


def _seg(
    base_start: float,
    base_end: float,
    *,
    confidence: float = 0.9,
    reliable: bool = True,
    offset: float = 0.0,
) -> AlignmentSegment:
    return AlignmentSegment(
        base_start=base_start,
        base_end=base_end,
        donor_start=base_start + offset,
        donor_end=base_end + offset,
        speed=1.0,
        offset_seconds=offset,
        confidence=confidence,
        reliable=reliable,
    )


def _result(
    placements: list[Placement],
    *,
    duration: float = 10.0,
    skipped: list[AlignmentSegment] | None = None,
) -> ReconstructionResult:
    return ReconstructionResult(
        donor_path=Path("donor.mkv"),
        output_wav=Path("out.wav"),
        sample_rate=8000,
        channels=1,
        duration_seconds=duration,
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        placements=placements,
        skipped_segments=list(skipped or []),
        notes=[],
    )


def test_placement_overlaps_detected() -> None:
    placements = [
        Placement(base_start=0.0, base_end=5.0, donor_start=0.0, donor_end=5.0),
        Placement(base_start=4.5, base_end=8.0, donor_start=4.5, donor_end=8.0),
    ]
    findings = placement_overlaps(placements)
    assert any(f.code == "placement_overlap" for f in findings)


def test_unexplained_gap_vs_missing_base() -> None:
    placements = [
        Placement(base_start=0.0, base_end=4.0, donor_start=0.0, donor_end=4.0),
        Placement(base_start=8.0, base_end=10.0, donor_start=5.0, donor_end=7.0),
    ]
    coverage, findings = base_coverage(
        placements,
        base_duration=10.0,
        explained_gaps=[(4.0, 8.0)],
        min_gap_seconds=0.5,
    )
    assert coverage.gap_seconds == pytest.approx(4.0)
    assert findings == []

    _, unexplained = base_coverage(
        placements,
        base_duration=10.0,
        explained_gaps=[],
        min_gap_seconds=0.5,
    )
    assert any(f.code == "unexplained_base_gap" for f in unexplained)


def test_duration_mismatch() -> None:
    assert duration_mismatch_finding(
        base_duration=10.0, output_duration=10.1, tolerance_seconds=0.25
    ) is None
    bad = duration_mismatch_finding(
        base_duration=10.0, output_duration=11.0, tolerance_seconds=0.25
    )
    assert bad is not None
    assert bad.severity == FindingSeverity.ERROR


def test_report_flags_low_confidence_and_uncertain() -> None:
    doc = AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=0.8,
        segments=[_seg(0, 5, confidence=0.2), _seg(5, 10, confidence=0.9)],
        uncertain_intervals=[
            TimelineInterval(start=3.0, end=4.0, timeline="base", reason="weak")
        ],
    )
    result = _result(
        [
            Placement(base_start=0.0, base_end=5.0, donor_start=0.0, donor_end=5.0),
            Placement(base_start=5.0, base_end=10.0, donor_start=5.0, donor_end=10.0),
        ]
    )
    settings = Settings(_env_file=None)
    report = build_processing_report(doc, result, base_duration=10.0, settings=settings)
    assert report.needs_manual_review
    assert "low_confidence_segments" in report.review_reasons
    assert "uncertain_intervals" in report.review_reasons
    assert any(f.code == "low_confidence_segment" for f in report.findings)
    assert any(f.code == "uncertain_interval" for f in report.findings)


def test_surplus_info_does_not_force_review_alone() -> None:
    doc = AlignmentDocument(
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.9,
        segments=[_seg(0, 10, offset=0.0), _seg(10, 20, offset=15.0)],
        unrelated_donor_intervals=[
            TimelineInterval(
                start=10.0, end=25.0, timeline="donor", reason="curve_break_surplus_donor"
            )
        ],
    )
    result = _result(
        [
            Placement(base_start=0.0, base_end=10.0, donor_start=0.0, donor_end=10.0),
            Placement(base_start=10.0, base_end=20.0, donor_start=25.0, donor_end=35.0),
        ],
        duration=20.0,
    )
    settings = Settings(_env_file=None)
    report = build_processing_report(doc, result, base_duration=20.0, settings=settings)
    assert any(f.code == "unrelated_donor_interval" for f in report.findings)
    assert report.needs_manual_review is False


def test_write_report_and_reconstruct_from_wav(tmp_path: Path) -> None:
    sr = 8000
    donor = np.sin(np.linspace(0, 20, sr * 5)).astype(np.float64) * 0.2
    donor_wav = tmp_path / "donor.wav"
    write_pcm16_wav(donor_wav, donor, sample_rate=sr)
    base = np.sin(np.linspace(0, 30, sr * 8)).astype(np.float64) * 0.2
    base_wav = tmp_path / "base.wav"
    write_pcm16_wav(base_wav, base, sample_rate=sr)

    doc = AlignmentDocument(
        base_path=base_wav,
        donor_path=donor_wav,
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.95,
        segments=[_seg(0, 3), _seg(6, 8, offset=-3.0)],
        missing_base_intervals=[
            TimelineInterval(
                start=3.0, end=6.0, timeline="base", reason="curve_break_missing_base"
            )
        ],
    )
    out = tmp_path / "out.wav"
    result = reconstruct_from_wav(
        donor_wav,
        doc,
        out,
        base_duration=8.0,
        base_wav=base_wav,
        missing_base_fill_seconds=1.0,
    )
    settings = Settings(_env_file=None)
    report = build_processing_report(doc, result, base_duration=8.0, settings=settings)
    path = write_processing_report(report, tmp_path)
    assert path.is_file()
    assert (tmp_path / "processing_report.txt").is_file()
    assert any(f.code == "missing_base_interval" for f in report.findings)
    # Gap explained by missing-base → no unexplained gap review reason required.
    assert "unexplained_base_gaps" not in report.review_reasons
