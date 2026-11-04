"""Optional e2e reconstruct + processing report against local smoke matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_media_processor.alignment.detect import (
    _activity_pairs_from_report,
    detect_alignment_from_artifacts,
)
from jarvis_media_processor.audio.models import AudioAnalysisArtifact
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media.inspect import inspect_media
from jarvis_media_processor.reconstruct.pipeline import reconstruct_donor_audio
from jarvis_media_processor.report.models import ProcessingReport
from jarvis_media_processor.video.models import VisualMatchReport

SMOKE_ROOT = Path(".jmp/smoke-matrix")


def _align_dir(name: str) -> Path:
    root = SMOKE_ROOT / name
    sr = root / "align-sr"
    if (sr / "visual" / "visual_match_report.json").is_file():
        return sr
    return root / "align"


def _load_audio(directory: Path) -> AudioAnalysisArtifact | None:
    candidates = sorted(directory.glob("*audio_analysis.json"))
    if not candidates:
        return None
    return AudioAnalysisArtifact.model_validate_json(
        candidates[0].read_text(encoding="utf-8")
    )


def _fresh_alignment(name: str):
    align = _align_dir(name)
    report_path = align / "visual" / "visual_match_report.json"
    if not report_path.is_file():
        pytest.skip(f"missing visual report for {name}")
    report = VisualMatchReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    pairs, _sync, unrelated, missing, _locks, _notes = _activity_pairs_from_report(
        report
    )
    doc = detect_alignment_from_artifacts(
        _load_audio(align / "audio_base"),
        _load_audio(align / "audio_donor"),
        visual_matches=report.matches,
        activity_pairs=pairs,
        unrelated_donor_intervals=unrelated,
        missing_base_intervals=missing,
        refine_audio_offset=False,
    )
    root = SMOKE_ROOT / name
    base = root / "base.mkv"
    donor = root / "donor.mkv"
    if not donor.is_file():
        donor = root / "donor.avi"
    doc.base_path = base.resolve()
    doc.donor_path = donor.resolve()
    return doc, base, donor


@pytest.mark.skipif(
    not (SMOKE_ROOT / "synth2-missing-base-30s" / "base.mkv").is_file(),
    reason="smoke matrix not present",
)
def test_e2e_synth2_missing_base_report(tmp_path: Path) -> None:
    doc, _base, donor = _fresh_alignment("synth2-missing-base-30s")
    assert any("missing" in item.reason for item in doc.missing_base_intervals)
    info = inspect_media(doc.base_path, settings=get_settings())
    result = reconstruct_donor_audio(
        donor,
        doc,
        tmp_path / "recon",
        base_duration=float(info.duration_seconds),
        settings=get_settings(),
    )
    report = ProcessingReport.model_validate_json(
        (tmp_path / "recon" / "processing_report.json").read_text(encoding="utf-8")
    )
    assert result.output_wav.is_file()
    assert any(f.code == "missing_base_interval" for f in report.findings)
    assert "unexplained_base_gaps" not in report.review_reasons


@pytest.mark.skipif(
    not (SMOKE_ROOT / "synth2-black-ad-15s" / "base.mkv").is_file(),
    reason="smoke matrix not present",
)
def test_e2e_synth2_black_ad_report(tmp_path: Path) -> None:
    doc, _base, donor = _fresh_alignment("synth2-black-ad-15s")
    assert doc.unrelated_donor_intervals
    info = inspect_media(doc.base_path, settings=get_settings())
    reconstruct_donor_audio(
        donor,
        doc,
        tmp_path / "recon",
        base_duration=float(info.duration_seconds),
        settings=get_settings(),
    )
    report = ProcessingReport.model_validate_json(
        (tmp_path / "recon" / "processing_report.json").read_text(encoding="utf-8")
    )
    assert any(f.code == "unrelated_donor_interval" for f in report.findings)
    # Surplus (info) alone must not be a review reason.
    assert "unrelated_donor_intervals" not in report.review_reasons
