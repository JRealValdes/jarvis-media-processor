"""Serializable audio analysis artifacts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AcousticLandmark(BaseModel):
    """A sparse acoustic event usable as a sync anchor."""

    time_seconds: float = Field(ge=0)
    strength: float = Field(ge=0)
    kind: str = "energy_peak"


class AudioAnalysisArtifact(BaseModel):
    """Analysis output for one media file (base or donor)."""

    source_path: Path
    role: str | None = Field(
        default=None,
        description="Optional role label such as 'base' or 'donor'.",
    )
    wav_path: Path | None = None
    sample_rate: int
    channels: int
    duration_seconds: float
    landmarks: list[AcousticLandmark] = Field(default_factory=list)
    energy_hop_seconds: float
    energy_envelope: list[float] = Field(
        default_factory=list,
        description="Per-hop RMS energy (normalized) for debugging/plots.",
    )
    params: dict[str, float | int | str] = Field(default_factory=dict)
