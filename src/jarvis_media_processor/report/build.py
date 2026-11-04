"""Assemble a ``ProcessingReport`` from alignment + reconstruction artifacts."""

from __future__ import annotations

from pathlib import Path

from jarvis_media_processor.alignment.models import AlignmentDocument
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.reconstruct.models import ReconstructionResult
from jarvis_media_processor.report.models import FindingSeverity, ProcessingReport
from jarvis_media_processor.report.validate import (
    alignment_status_findings,
    base_coverage,
    duration_mismatch_finding,
    format_abrupt_coverage_summary,
    low_confidence_findings,
    placement_overlaps,
    skipped_segment_findings,
)


def build_processing_report(
    alignment: AlignmentDocument,
    reconstruction: ReconstructionResult,
    *,
    base_duration: float,
    settings: Settings | None = None,
) -> ProcessingReport:
    """Validate sync/reconstruction and return a serializable processing report."""
    cfg = settings or get_settings()
    findings = []
    findings.extend(alignment_status_findings(alignment))
    findings.extend(placement_overlaps(reconstruction.placements))
    findings.extend(skipped_segment_findings(reconstruction))

    low_rows, low_findings = low_confidence_findings(
        alignment, threshold=cfg.report_low_confidence
    )
    findings.extend(low_findings)

    explained = [
        (item.start, item.end)
        for item in alignment.missing_base_intervals
        if item.timeline == "base"
    ]
    coverage, gap_findings = base_coverage(
        reconstruction.placements,
        base_duration=base_duration,
        explained_gaps=explained,
        min_gap_seconds=cfg.report_min_gap_seconds,
    )
    findings.extend(gap_findings)

    mismatch = duration_mismatch_finding(
        base_duration=base_duration,
        output_duration=reconstruction.duration_seconds,
        tolerance_seconds=cfg.report_duration_tolerance_seconds,
    )
    if mismatch is not None:
        findings.append(mismatch)

    review_reasons: list[str] = []
    if any(f.severity == FindingSeverity.ERROR for f in findings):
        review_reasons.append("error_findings")
    if alignment.strategy.value == "failed":
        review_reasons.append("alignment_failed")
    if alignment.confidence < cfg.report_min_global_confidence:
        review_reasons.append("low_global_confidence")
    if alignment.uncertain_intervals:
        review_reasons.append("uncertain_intervals")
    if any(f.code == "low_confidence_segment" for f in findings):
        review_reasons.append("low_confidence_segments")
    if any(f.code == "unexplained_base_gap" for f in findings):
        review_reasons.append("unexplained_base_gaps")
    # Surplus-only runs should not force review by themselves (info findings).

    notes = list(reconstruction.notes)
    silence_summary = format_abrupt_coverage_summary(
        findings,
        min_seconds=cfg.report_min_gap_seconds,
    )
    notes.extend(silence_summary.splitlines())
    notes.append(
        "Manual review rules: any ERROR finding; FAILED strategy; "
        f"global confidence < {cfg.report_min_global_confidence:.2f}; "
        "uncertain intervals; low-confidence segments; unexplained base gaps."
    )

    return ProcessingReport(
        base_path=alignment.base_path,
        donor_path=alignment.donor_path or reconstruction.donor_path,
        strategy=alignment.strategy,
        alignment_confidence=float(alignment.confidence),
        needs_manual_review=bool(review_reasons),
        review_reasons=review_reasons,
        findings=findings,
        segment_count=len(alignment.segments),
        placement_count=len(reconstruction.placements),
        skipped_segment_count=len(reconstruction.skipped_segments),
        low_confidence_segments=low_rows,
        uncertain_intervals=list(alignment.uncertain_intervals),
        missing_base_intervals=list(alignment.missing_base_intervals),
        unrelated_donor_intervals=list(alignment.unrelated_donor_intervals),
        coverage=coverage,
        output_wav=reconstruction.output_wav,
        output_duration_seconds=float(reconstruction.duration_seconds),
        base_duration_seconds=float(base_duration),
        notes=notes,
    )


def write_processing_report(report: ProcessingReport, workspace: Path | str) -> Path:
    """Write ``processing_report.json`` and ``processing_report.txt`` under workspace."""
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)
    json_path = work / "processing_report.json"
    txt_path = work / "processing_report.txt"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(report.summary_text(), encoding="utf-8")
    return json_path
