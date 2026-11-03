"""Job records, states, and progress for standalone execution."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Lifecycle states for a processing job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class JobProgress(BaseModel):
    """Structured progress snapshot."""

    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    stage: str = ""
    message: str = ""


class JobError(BaseModel):
    """Structured failure payload."""

    code: str
    message: str
    stage: str | None = None


class JobParams(BaseModel):
    """Immutable submission parameters."""

    base_path: Path
    donor_path: Path
    output_path: Path
    use_video: bool = True
    fine_warp: bool = True
    donor_language: str = "spa"
    donor_title: str = "Spanish (reconstructed)"
    idempotency_key: str | None = None


class JobRecord(BaseModel):
    """Persisted job status (JSON on the local filesystem)."""

    id: str
    params: JobParams
    state: JobState = JobState.PENDING
    progress: JobProgress = Field(default_factory=JobProgress)
    error: JobError | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    workspace: Path | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def touch(self) -> None:
        """Bump ``updated_at``."""
        self.updated_at = utc_now()

    def is_terminal(self) -> bool:
        """Return True when the job will not run further."""
        return self.state in TERMINAL_STATES
