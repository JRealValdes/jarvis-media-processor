"""Artifacts for donor-audio reconstruction onto the base timeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from jarvis_media_processor.alignment.models import AlignmentSegment, SyncStrategyLevel


class Placement(BaseModel):
    """One donor slice placed onto the base timeline."""

    base_start: float
    base_end: float
    donor_start: float
    donor_end: float
    speed: float = 1.0
    crossfade_seconds: float = 0.0
    reliable: bool = True


class ReconstructionResult(BaseModel):
    """Result of reconstructing donor audio on the base timeline."""

    donor_path: Path
    output_wav: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    strategy: SyncStrategyLevel
    placements: list[Placement] = Field(default_factory=list)
    skipped_segments: list[AlignmentSegment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
