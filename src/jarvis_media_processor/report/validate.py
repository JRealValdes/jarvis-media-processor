"""Validators for alignment + reconstruction consistency."""

from __future__ import annotations

from jarvis_media_processor.alignment.models import AlignmentDocument, SyncStrategyLevel
from jarvis_media_processor.reconstruct.models import Placement, ReconstructionResult
from jarvis_media_processor.report.models import (
    CoverageSummary,
    FindingSeverity,
    ValidationFinding,
)

# Finding codes that describe abrupt silence / uncovered / excluded spans.
_ABRUPT_SPAN_LABELS: dict[str, str] = {
    "unexplained_base_gap": "unexplained",
    "base_uncovered_edge": "uncovered_edge",
    "missing_base_interval": "missing_base_filled",
    "unrelated_donor_interval": "excluded_surplus",
}


def format_hhmmss(seconds: float) -> str:
    """Format a timeline position as ``HH:MM:SS`` (rounded to nearest second)."""
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def abrupt_coverage_span_lines(
    findings: list[ValidationFinding],
    *,
    min_seconds: float = 0.5,
) -> list[str]:
    """Human-readable lines for sudden silence / uncovered / excluded spans.

    Each line looks like::

        base 01:44:48–01:45:11 (23.0s) unexplained
        donor 01:40:32–01:41:21 (49.0s) excluded_surplus
    """
    rows: list[tuple[str, float, float, str]] = []
    for finding in findings:
        label = _ABRUPT_SPAN_LABELS.get(finding.code)
        if label is None or finding.start is None or finding.end is None:
            continue
        start = float(finding.start)
        end = float(finding.end)
        if end - start < min_seconds - 1e-9:
            continue
        timeline = finding.timeline or "base"
        rows.append((timeline, start, end, label))
    rows.sort(key=lambda row: (0 if row[0] == "base" else 1, row[1], row[2]))
    lines: list[str] = []
    for timeline, start, end, label in rows:
        length = end - start
        lines.append(
            f"{timeline} {format_hhmmss(start)}–{format_hhmmss(end)} "
            f"({length:.1f}s) {label}"
        )
    return lines


def format_abrupt_coverage_summary(
    findings: list[ValidationFinding],
    *,
    min_seconds: float = 0.5,
) -> str:
    """One multi-line block suitable for INFO logs and report notes."""
    lines = abrupt_coverage_span_lines(findings, min_seconds=min_seconds)
    if not lines:
        return "Abrupt silence / uncovered spans: none"
    header = f"Abrupt silence / uncovered spans ({len(lines)}):"
    return header + "\n" + "\n".join(f"  {line}" for line in lines)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item[0])
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-9:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def placement_overlaps(
    placements: list[Placement],
    *,
    min_overlap_seconds: float = 0.05,
) -> list[ValidationFinding]:
    """Flag overlapping placements on the base timeline."""
    findings: list[ValidationFinding] = []
    ordered = sorted(placements, key=lambda p: p.base_start)
    for left, right in zip(ordered, ordered[1:], strict=False):
        overlap = min(left.base_end, right.base_end) - max(left.base_start, right.base_start)
        if overlap >= min_overlap_seconds:
            findings.append(
                ValidationFinding(
                    code="placement_overlap",
                    severity=FindingSeverity.ERROR,
                    message=(
                        f"Placements overlap by {overlap:.3f}s on the base timeline."
                    ),
                    timeline="base",
                    start=max(left.base_start, right.base_start),
                    end=min(left.base_end, right.base_end),
                )
            )
    return findings


def base_coverage(
    placements: list[Placement],
    *,
    base_duration: float,
    explained_gaps: list[tuple[float, float]] | None = None,
    min_gap_seconds: float = 0.5,
    donor_duration: float | None = None,
) -> tuple[CoverageSummary, list[ValidationFinding]]:
    """Measure placement coverage and report unexplained base gaps."""
    duration = max(0.0, float(base_duration))
    covered_spans = _merge_intervals(
        [(p.base_start, min(p.base_end, duration)) for p in placements if p.base_end > p.base_start]
    )
    covered = sum(end - start for start, end in covered_spans)
    donor_spans = _merge_intervals(
        [
            (float(p.donor_start), float(p.donor_end))
            for p in placements
            if p.donor_end > p.donor_start
        ]
    )
    donor_used = sum(end - start for start, end in donor_spans)
    donor_dur: float | None = None
    donor_ratio: float | None = None
    if donor_duration is not None and float(donor_duration) > 1e-9:
        donor_dur = max(0.0, float(donor_duration))
        donor_ratio = max(0.0, min(1.0, donor_used / donor_dur))
    explained = _merge_intervals(list(explained_gaps or []))

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in covered_spans:
        if start - cursor >= min_gap_seconds:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= min_gap_seconds:
        gaps.append((cursor, duration))

    findings: list[ValidationFinding] = []
    hull_start = covered_spans[0][0] if covered_spans else 0.0
    hull_end = covered_spans[-1][1] if covered_spans else 0.0
    for gap_start, gap_end in gaps:
        # Subtract explained (missing-base) coverage from this gap.
        remaining = [(gap_start, gap_end)]
        for exp_start, exp_end in explained:
            next_remaining: list[tuple[float, float]] = []
            for r_start, r_end in remaining:
                if exp_end <= r_start or exp_start >= r_end:
                    next_remaining.append((r_start, r_end))
                    continue
                if exp_start > r_start:
                    next_remaining.append((r_start, exp_start))
                if exp_end < r_end:
                    next_remaining.append((exp_end, r_end))
            remaining = next_remaining
        for r_start, r_end in remaining:
            length = r_end - r_start
            if length < min_gap_seconds:
                continue
            # Leading/trailing silence (donor shorter/longer offset) is expected
            # padding — info only. Interior holes need operator attention.
            edge = r_end <= hull_start + 1e-9 or r_start >= hull_end - 1e-9
            findings.append(
                ValidationFinding(
                    code="base_uncovered_edge" if edge else "unexplained_base_gap",
                    severity=(
                        FindingSeverity.INFO if edge else FindingSeverity.WARNING
                    ),
                    message=(
                        f"Base timeline has {length:.2f}s without donor placement "
                        + (
                            "(outside placement hull; often shorter donor)."
                            if edge
                            else "and without a missing-base interval."
                        )
                    ),
                    timeline="base",
                    start=r_start,
                    end=r_end,
                )
            )

    gap_seconds = max(0.0, duration - covered)
    ratio = 0.0 if duration <= 1e-9 else max(0.0, min(1.0, covered / duration))
    summary = CoverageSummary(
        base_duration_seconds=duration,
        covered_seconds=covered,
        gap_seconds=gap_seconds,
        coverage_ratio=ratio,
        donor_duration_seconds=donor_dur,
        donor_used_seconds=float(donor_used) if donor_spans or donor_dur is not None else None,
        donor_utilization_ratio=donor_ratio,
    )
    return summary, findings

def low_confidence_findings(
    doc: AlignmentDocument,
    *,
    threshold: float,
) -> tuple[list[dict[str, float | bool]], list[ValidationFinding]]:
    """List segments below the confidence threshold."""
    rows: list[dict[str, float | bool]] = []
    findings: list[ValidationFinding] = []
    for segment in doc.segments:
        if segment.confidence >= threshold and segment.reliable:
            continue
        rows.append(
            {
                "base_start": segment.base_start,
                "base_end": segment.base_end,
                "confidence": segment.confidence,
                "reliable": segment.reliable,
            }
        )
        if segment.confidence < threshold:
            findings.append(
                ValidationFinding(
                    code="low_confidence_segment",
                    severity=FindingSeverity.WARNING,
                    message=(
                        f"Segment confidence {segment.confidence:.3f} "
                        f"is below threshold {threshold:.3f}."
                    ),
                    timeline="base",
                    start=segment.base_start,
                    end=segment.base_end,
                )
            )
        if not segment.reliable:
            findings.append(
                ValidationFinding(
                    code="unreliable_segment",
                    severity=FindingSeverity.WARNING,
                    message="Segment marked unreliable.",
                    timeline="base",
                    start=segment.base_start,
                    end=segment.base_end,
                )
            )
    return rows, findings


def duration_mismatch_finding(
    *,
    base_duration: float,
    output_duration: float,
    tolerance_seconds: float,
) -> ValidationFinding | None:
    """Flag reconstructed WAV length vs requested base duration."""
    delta = abs(float(output_duration) - float(base_duration))
    if delta <= tolerance_seconds:
        return None
    return ValidationFinding(
        code="duration_mismatch",
        severity=FindingSeverity.ERROR,
        message=(
            f"Output duration {output_duration:.3f}s differs from base duration "
            f"{base_duration:.3f}s by {delta:.3f}s "
            f"(tolerance {tolerance_seconds:.3f}s)."
        ),
    )


def alignment_status_findings(doc: AlignmentDocument) -> list[ValidationFinding]:
    """Findings from alignment strategy / uncertain regions."""
    findings: list[ValidationFinding] = []
    if doc.strategy == SyncStrategyLevel.FAILED:
        findings.append(
            ValidationFinding(
                code="alignment_failed",
                severity=FindingSeverity.ERROR,
                message="Alignment strategy is failed.",
            )
        )
    for interval in doc.uncertain_intervals:
        findings.append(
            ValidationFinding(
                code="uncertain_interval",
                severity=FindingSeverity.WARNING,
                message=f"Uncertain region ({interval.reason}).",
                timeline=interval.timeline,
                start=interval.start,
                end=interval.end,
            )
        )
    for interval in doc.missing_base_intervals:
        findings.append(
            ValidationFinding(
                code="missing_base_interval",
                severity=FindingSeverity.INFO,
                message=(
                    f"Missing base content ({interval.reason}); "
                    "fill from base track 0 if long enough."
                ),
                timeline=interval.timeline,
                start=interval.start,
                end=interval.end,
            )
        )
    for interval in doc.unrelated_donor_intervals:
        findings.append(
            ValidationFinding(
                code="unrelated_donor_interval",
                severity=FindingSeverity.INFO,
                message=f"Unrelated donor span excluded ({interval.reason}).",
                timeline=interval.timeline,
                start=interval.start,
                end=interval.end,
            )
        )
    return findings


def skipped_segment_findings(
    result: ReconstructionResult,
) -> list[ValidationFinding]:
    """Note segments dropped during placement planning."""
    if not result.skipped_segments:
        return []
    return [
        ValidationFinding(
            code="skipped_segments",
            severity=FindingSeverity.WARNING,
            message=f"{len(result.skipped_segments)} alignment segment(s) were not placed.",
        )
    ]
