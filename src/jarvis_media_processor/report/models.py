"""Serializable processing / sync-validation report for operators and job APIs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis_media_processor.alignment.models import SyncStrategyLevel, TimelineInterval


class FindingSeverity(StrEnum):
    """How serious a validation finding is for operators."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationFinding(BaseModel):
    """One validation note tied optionally to a timeline span."""

    code: str
    severity: FindingSeverity
    message: str
    timeline: str | None = None
    start: float | None = None
    end: float | None = None


class CoverageSummary(BaseModel):
    """How much of the base/donor timelines are covered by placements."""

    base_duration_seconds: float
    covered_seconds: float
    gap_seconds: float
    coverage_ratio: float = Field(ge=0, le=1)
    donor_duration_seconds: float | None = None
    donor_used_seconds: float | None = None
    donor_utilization_ratio: float | None = Field(default=None, ge=0, le=1)


class ProcessingReport(BaseModel):
    """End-of-run sync validation and reconstruction summary.

    Suitable for later job APIs: JSON-serializable, stable field names, no UI.
    """

    base_path: Path | None = None
    donor_path: Path | None = None
    strategy: SyncStrategyLevel
    alignment_confidence: float = Field(ge=0, le=1)
    needs_manual_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    findings: list[ValidationFinding] = Field(default_factory=list)
    segment_count: int = 0
    placement_count: int = 0
    skipped_segment_count: int = 0
    low_confidence_segments: list[dict[str, float | bool]] = Field(default_factory=list)
    uncertain_intervals: list[TimelineInterval] = Field(default_factory=list)
    missing_base_intervals: list[TimelineInterval] = Field(default_factory=list)
    unrelated_donor_intervals: list[TimelineInterval] = Field(default_factory=list)
    coverage: CoverageSummary | None = None
    output_wav: Path | None = None
    output_duration_seconds: float | None = None
    base_duration_seconds: float | None = None
    notes: list[str] = Field(default_factory=list)

    def summary_text(self) -> str:
        """Human-readable one-screen summary for CLI / log tails."""
        lines = [
            f"strategy={self.strategy.value} confidence={self.alignment_confidence:.3f}",
            f"needs_manual_review={self.needs_manual_review}",
        ]
        if self.review_reasons:
            lines.append("review_reasons: " + "; ".join(self.review_reasons))
        if self.coverage is not None:
            cov = self.coverage
            line = (
                f"base_covered={cov.coverage_ratio:.1%} "
                f"({cov.covered_seconds:.1f}/{cov.base_duration_seconds:.1f}s; "
                f"gaps={cov.gap_seconds:.2f}s)"
            )
            if (
                cov.donor_utilization_ratio is not None
                and cov.donor_used_seconds is not None
                and cov.donor_duration_seconds is not None
            ):
                line += (
                    f" donor_used={cov.donor_utilization_ratio:.1%} "
                    f"({cov.donor_used_seconds:.1f}/{cov.donor_duration_seconds:.1f}s)"
                )
            lines.append(line)
        lines.append(
            f"segments={self.segment_count} placements={self.placement_count} "
            f"skipped={self.skipped_segment_count}"
        )
        lines.append(
            f"missing_base={len(self.missing_base_intervals)} "
            f"unrelated_donor={len(self.unrelated_donor_intervals)} "
            f"uncertain={len(self.uncertain_intervals)}"
        )
        errors = sum(1 for f in self.findings if f.severity == FindingSeverity.ERROR)
        warnings = sum(
            1 for f in self.findings if f.severity == FindingSeverity.WARNING
        )
        lines.append(
            f"findings errors={errors} warnings={warnings} total={len(self.findings)}"
        )
        # Prefer the abrupt-silence note block when present (added by build).
        silence_notes = [
            note
            for note in self.notes
            if note.startswith("Abrupt silence") or note.startswith("  base ") or note.startswith("  donor ")
        ]
        if silence_notes:
            lines.extend(silence_notes)
        if self.output_wav is not None:
            lines.append(f"output={self.output_wav}")
        for finding in self.findings[:12]:
            span = ""
            if finding.start is not None and finding.end is not None:
                span = f" [{finding.start:.1f}-{finding.end:.1f}s]"
            lines.append(
                f"  - {finding.severity.value}:{finding.code}{span}: {finding.message}"
            )
        if len(self.findings) > 12:
            lines.append(f"  … {len(self.findings) - 12} more findings")
        return "\n".join(lines) + "\n"
