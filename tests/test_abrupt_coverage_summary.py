"""Unit tests for abrupt silence / uncovered-span report summary."""

from __future__ import annotations

from jarvis_media_processor.report.models import FindingSeverity, ValidationFinding
from jarvis_media_processor.report.validate import (
    abrupt_coverage_span_lines,
    format_abrupt_coverage_summary,
    format_hhmmss,
)


def test_format_hhmmss_rounds_spider_man_silence() -> None:
    assert format_hhmmss(6288.06) == "01:44:48"
    assert format_hhmmss(6311.0) == "01:45:11"


def test_abrupt_coverage_span_lines_labels_and_order() -> None:
    findings = [
        ValidationFinding(
            code="unrelated_donor_interval",
            severity=FindingSeverity.INFO,
            message="surplus",
            timeline="donor",
            start=6031.56,
            end=6080.56,
        ),
        ValidationFinding(
            code="unexplained_base_gap",
            severity=FindingSeverity.WARNING,
            message="hole",
            timeline="base",
            start=6288.06,
            end=6311.0,
        ),
        ValidationFinding(
            code="missing_base_interval",
            severity=FindingSeverity.INFO,
            message="fill",
            timeline="base",
            start=6311.0,
            end=6365.0,
        ),
        ValidationFinding(
            code="placement_overlap",
            severity=FindingSeverity.ERROR,
            message="ignore",
            timeline="base",
            start=10.0,
            end=20.0,
        ),
    ]
    lines = abrupt_coverage_span_lines(findings, min_seconds=0.5)
    assert lines == [
        "base 01:44:48–01:45:11 (22.9s) unexplained",
        "base 01:45:11–01:46:05 (54.0s) missing_base_filled",
        "donor 01:40:32–01:41:21 (49.0s) excluded_surplus",
    ]


def test_format_abrupt_coverage_summary_none() -> None:
    assert (
        format_abrupt_coverage_summary([])
        == "Abrupt silence / uncovered spans: none"
    )


def test_format_abrupt_coverage_summary_block() -> None:
    findings = [
        ValidationFinding(
            code="unexplained_base_gap",
            severity=FindingSeverity.WARNING,
            message="hole",
            timeline="base",
            start=100.0,
            end=112.5,
        )
    ]
    text = format_abrupt_coverage_summary(findings)
    assert text.startswith("Abrupt silence / uncovered spans (1):")
    assert "base 00:01:40–00:01:52 (12.5s) unexplained" in text
