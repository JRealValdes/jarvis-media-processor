"""Batch discovery results, dispositions, and summaries."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class PairLayout(StrEnum):
    """How a base/donor pair was discovered on disk."""

    SUBDIR = "subdir"
    FLAT = "flat"
    SPLIT = "split"


class PairDisposition(StrEnum):
    """What the batch runner plans to do with a discovered pair."""

    RUN = "run"
    RETRY = "retry"
    SKIP_COMPLETED = "skip_completed"
    SKIP_IN_FLIGHT = "skip_in_flight"
    SKIP_FAILED = "skip_failed"


class BatchItemOutcome(StrEnum):
    """Final outcome for one pair in a batch run."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    PLANNED = "planned"  # dry-run only


class MediaPair(BaseModel):
    """One base/donor pair with a deterministic output path."""

    pair_id: str
    base_path: Path
    donor_path: Path
    output_path: Path
    layout: PairLayout


class BatchItemResult(BaseModel):
    """Per-pair result inside a batch summary."""

    pair_id: str
    disposition: PairDisposition
    outcome: BatchItemOutcome
    reason: str = ""
    job_id: str | None = None
    base_path: Path
    donor_path: Path
    output_path: Path
    layout: PairLayout


class BatchSummary(BaseModel):
    """Aggregate status for a batch enqueue/run."""

    id: str
    input_dir: Path
    output_dir: Path
    created_at: datetime = Field(default_factory=utc_now)
    dry_run: bool = False
    retry: bool = False
    items: list[BatchItemResult] = Field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for i in self.items if i.outcome == BatchItemOutcome.SUCCEEDED)

    @property
    def failed(self) -> int:
        return sum(1 for i in self.items if i.outcome == BatchItemOutcome.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for i in self.items if i.outcome == BatchItemOutcome.SKIPPED)

    @property
    def planned(self) -> int:
        return sum(1 for i in self.items if i.outcome == BatchItemOutcome.PLANNED)

    def counts(self) -> dict[str, int]:
        """Compact counts for CLI/JSON summaries."""
        return {
            "total": len(self.items),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "planned": self.planned,
        }
