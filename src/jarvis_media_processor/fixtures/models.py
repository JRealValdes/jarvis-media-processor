"""Ground-truth models for synthetic alignment fixtures."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SyncStrategyLevel(StrEnum):
    """Which escalating sync level should suffice for a scenario."""

    ALREADY_SYNCED = "already_synced"
    CONSTANT_OFFSET = "constant_offset"
    CONSTANT_SPEED = "constant_speed"
    PIECEWISE = "piecewise"


class SegmentMapping(BaseModel):
    """One aligned interval mapping donor time onto the base timeline."""

    base_start: float = Field(ge=0)
    base_end: float = Field(gt=0)
    donor_start: float = Field(ge=0)
    donor_end: float = Field(gt=0)
    speed: float = Field(default=1.0, gt=0)
    confidence: float = Field(default=1.0, ge=0, le=1)


class Landmark(BaseModel):
    """Shared non-dialogue acoustic landmark for tests."""

    kind: str
    base_time: float = Field(ge=0)
    donor_time: float = Field(ge=0)
    note: str | None = None


class MediaRef(BaseModel):
    """Expected media file metadata within a scenario output directory."""

    filename: str
    duration_seconds: float = Field(gt=0)


class TimeInterval(BaseModel):
    """Half-open-ish interval on a timeline (start inclusive, end exclusive in intent)."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)


class GroundTruth(BaseModel):
    """Documented expected alignment for a synthetic base/donor pair."""

    scenario_id: str
    description: str
    expected_strategy: SyncStrategyLevel
    base: MediaRef
    donor: MediaRef
    segments: list[SegmentMapping] = Field(default_factory=list)
    unrelated_donor_intervals: list[TimeInterval] = Field(default_factory=list)
    missing_base_intervals: list[TimeInterval] = Field(default_factory=list)
    landmarks: list[Landmark] = Field(default_factory=list)
    global_offset_seconds: float | None = Field(
        default=None,
        description="Donor lag vs base when a single offset suffices (donor = base + offset).",
    )
    global_speed: float | None = Field(
        default=None,
        description="Donor playback speed relative to base when constant (e.g. 1.1).",
    )
