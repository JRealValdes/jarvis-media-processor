"""Models for sampled frames and visual correspondences."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SampledFrame(BaseModel):
    """One extracted frame with timeline metadata."""

    time_seconds: float = Field(ge=0)
    path: Path
    phash: str
    role: str | None = None


class FrameCorrespondence(BaseModel):
    """Candidate match between a base frame and a donor frame."""

    base_time: float = Field(ge=0)
    donor_time: float = Field(ge=0)
    distance: int = Field(ge=0, description="Perceptual hash Hamming distance.")
    confidence: float = Field(ge=0, le=1)
    base_frame: Path | None = None
    donor_frame: Path | None = None


class VisualMatchReport(BaseModel):
    """Ranked visual correspondences for a base/donor pair."""

    base_path: Path
    donor_path: Path
    base_frames: list[SampledFrame] = Field(default_factory=list)
    donor_frames: list[SampledFrame] = Field(default_factory=list)
    matches: list[FrameCorrespondence] = Field(default_factory=list)
    params: dict[str, float | int | str] = Field(default_factory=dict)
