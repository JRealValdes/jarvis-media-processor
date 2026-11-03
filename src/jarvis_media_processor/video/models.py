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
    shot_anchor: bool = Field(
        default=False,
        description="True when base and/or donor frame is a detected shot cut.",
    )


class VisualMatchReport(BaseModel):
    """Ranked visual correspondences for a base/donor pair."""

    base_path: Path
    donor_path: Path
    base_frames: list[SampledFrame] = Field(default_factory=list)
    donor_frames: list[SampledFrame] = Field(default_factory=list)
    matches: list[FrameCorrespondence] = Field(default_factory=list)
    base_shot_times: list[float] = Field(
        default_factory=list,
        description="Detected shot-boundary times on the base timeline.",
    )
    donor_shot_times: list[float] = Field(
        default_factory=list,
        description="Detected shot-boundary times on the donor timeline.",
    )
    base_activity_times: list[float] = Field(
        default_factory=list,
        description="Sample times for the base visual activity envelope.",
    )
    base_activity_values: list[float] = Field(
        default_factory=list,
        description=(
            "Fused consecutive-frame activity on the base timeline "
            "(max of normalized pHash-diff and weighted structural change)."
        ),
    )
    donor_activity_times: list[float] = Field(
        default_factory=list,
        description="Sample times for the donor visual activity envelope.",
    )
    donor_activity_values: list[float] = Field(
        default_factory=list,
        description=(
            "Fused consecutive-frame activity on the donor timeline "
            "(max of normalized pHash-diff and weighted structural change)."
        ),
    )
    params: dict[str, float | int | str] = Field(default_factory=dict)
