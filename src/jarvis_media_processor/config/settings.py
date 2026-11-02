"""Typed settings loaded from environment variables and optional ``.env``."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Jarvis Media Processor.

    Environment variables use the ``JMP_`` prefix (see ``.env.example``).
    """

    model_config = SettingsConfigDict(
        env_prefix="JMP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = Field(default="INFO", description="Logging level name.")
    work_dir: Path = Field(
        default=Path(".jmp"),
        description="Root directory for temporary workspaces and job artifacts.",
    )
    ffmpeg_path: str = Field(
        default="ffmpeg",
        description="FFmpeg executable name or path (must be on PATH unless absolute).",
    )
    ffprobe_path: str = Field(
        default="ffprobe",
        description="FFprobe executable name or path (must be on PATH unless absolute).",
    )

    # Audio analysis / landmarks
    audio_analysis_sample_rate: int = Field(
        default=16_000,
        description="Sample rate for analysis WAV extraction.",
    )
    audio_analysis_channels: int = Field(
        default=1,
        description="Channel count for analysis WAV extraction.",
    )
    audio_landmark_hop_seconds: float = Field(
        default=0.01,
        description="Hop size for RMS energy envelope.",
    )
    audio_landmark_frame_seconds: float = Field(
        default=0.04,
        description="Frame size for RMS energy envelope.",
    )
    audio_landmark_min_distance_seconds: float = Field(
        default=0.25,
        description="Minimum spacing between energy-peak landmarks.",
    )
    audio_landmark_prominence: float = Field(
        default=0.15,
        description="Minimum normalized prominence for energy peaks.",
    )
    audio_landmark_max_count: int = Field(
        default=200,
        description="Maximum landmarks retained per file.",
    )

    # Video sampling / visual matching
    video_sample_fps: float = Field(
        default=1.0,
        description="Frames per second to sample for visual matching.",
    )
    video_phash_size: int = Field(default=8, description="Perceptual hash size.")
    video_normalize_size: int = Field(
        default=256,
        description="Edge size used before computing pHash.",
    )
    video_crop_letterbox: bool = Field(
        default=True,
        description="Crop near-black letterbox/pillarbox before hashing.",
    )
    video_match_max_distance: int = Field(
        default=12,
        description="Maximum pHash Hamming distance accepted as a candidate match.",
    )
    video_match_max_per_base: int = Field(
        default=3,
        description="Maximum donor candidates retained per base frame.",
    )
    video_shot_min_distance: int = Field(
        default=20,
        description="Min consecutive-frame pHash distance to mark a shot cut.",
    )

    # Alignment / escalating sync
    alignment_inlier_tolerance_seconds: float = Field(
        default=0.12,
        description="Max timing residual to count an evidence pair as an inlier.",
    )
    alignment_visual_inlier_tolerance_seconds: float = Field(
        default=1.5,
        description=(
            "Inlier tolerance for visual-only timing models "
            "(looser than audio; accounts for frame sampling)."
        ),
    )
    alignment_visual_min_pairs: int = Field(
        default=8,
        description="Minimum visual pairs before preferring a visual-first model.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        Cached ``Settings`` instance loaded from the environment.
    """
    return Settings()
