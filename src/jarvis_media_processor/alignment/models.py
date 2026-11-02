"""Serializable temporal alignment between base and donor timelines."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class SyncStrategyLevel(StrEnum):
    """Escalating sync hypothesis that was accepted (or required)."""

    ALREADY_SYNCED = "already_synced"
    CONSTANT_OFFSET = "constant_offset"
    CONSTANT_SPEED = "constant_speed"
    PIECEWISE = "piecewise"
    FAILED = "failed"


class AlignmentSegment(BaseModel):
    """One mapped interval from donor time onto the base timeline."""

    base_start: float = Field(ge=0)
    base_end: float = Field(gt=0)
    donor_start: float = Field(ge=0)
    donor_end: float = Field(gt=0)
    speed: float = Field(default=1.0, gt=0)
    offset_seconds: float = Field(
        default=0.0,
        description="donor_time ≈ speed * base_time + offset_seconds for this segment.",
    )
    confidence: float = Field(ge=0, le=1)
    reliable: bool = True


class TimelineInterval(BaseModel):
    """Interval on the base or donor timeline."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    timeline: str = Field(description="'base' or 'donor'")
    reason: str


class AlignmentDocument(BaseModel):
    """Machine-readable alignment report for a base/donor pair."""

    base_path: Path | None = None
    donor_path: Path | None = None
    strategy: SyncStrategyLevel
    confidence: float = Field(ge=0, le=1)
    global_offset_seconds: float | None = Field(
        default=None,
        description="Offset in donor ≈ speed * base + offset (seconds).",
    )
    global_speed: float | None = Field(
        default=None,
        description="Linear map donor_time ≈ global_speed * base_time + offset.",
    )
    segments: list[AlignmentSegment] = Field(default_factory=list)
    uncertain_intervals: list[TimelineInterval] = Field(default_factory=list)
    missing_base_intervals: list[TimelineInterval] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    evidence: dict[str, float | int | str] = Field(default_factory=dict)
