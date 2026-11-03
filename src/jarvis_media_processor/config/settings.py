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
    alignment_visual_min_inlier_ratio: float = Field(
        default=0.55,
        description=(
            "Minimum inlier coverage for visual-only global models. Lower than the "
            "audio ratio because frame sampling quantizes times (~1/fps)."
        ),
    )
    alignment_speed_min: float = Field(
        default=0.85,
        description="Minimum speed factor when fitting donor ≈ speed * base + offset.",
    )
    alignment_speed_max: float = Field(
        default=1.2,
        description="Maximum speed factor when fitting donor ≈ speed * base + offset.",
    )
    alignment_audio_downweight: float = Field(
        default=0.2,
        description=(
            "Weight multiplier for acoustic pairs when fused after a weak visual-only "
            "fit (avoids cross-language landmark noise dominating)."
        ),
    )
    alignment_audio_refine_window_seconds: float = Field(
        default=2.0,
        description=(
            "After a visual global model, search ±this many seconds for a fine "
            "audio-envelope offset nudge."
        ),
    )
    alignment_audio_refine_min_score: float = Field(
        default=0.05,
        description="Minimum cross-corr score required to apply the audio fine-offset.",
    )
    alignment_unrelated_window_seconds: float = Field(
        default=12.0,
        description="Half-window (seconds) when comparing local activity curves.",
    )
    alignment_unrelated_hop_seconds: float = Field(
        default=2.0,
        description="Step (seconds) while walking activity curves forward.",
    )
    alignment_unrelated_search_seconds: float = Field(
        default=1200.0,
        description=(
            "Deprecated alias retained for env compatibility; prefer "
            "alignment_unrelated_max_surplus_seconds."
        ),
    )
    alignment_unrelated_min_score: float = Field(
        default=0.3,
        description="Minimum local curve-match score to accept a lock or resume.",
    )
    alignment_unrelated_lag_jump_seconds: float = Field(
        default=2.0,
        description="Legacy setting; curve walk expands offsets itself.",
    )
    alignment_unrelated_min_span_seconds: float = Field(
        default=8.0,
        description="Ignore surplus-donor spans shorter than this.",
    )
    alignment_unrelated_max_surplus_seconds: float = Field(
        default=1200.0,
        description=(
            "Maximum donor surplus / ad length to search when resuming after a "
            "curve break (default 20 minutes)."
        ),
    )
    alignment_curve_confirm_seconds: float = Field(
        default=6.0,
        description="How long curves must disagree before declaring a break.",
    )
    alignment_missing_base_fill_seconds: float = Field(
        default=3.0,
        description=(
            "Missing-base gaps at least this long are filled from the base "
            "file's first audio track at reconstruct time."
        ),
    )

    # Post-reconstruct windowed lag field (base track 0 vs reconstructed WAV)
    fine_warp_max_lag_seconds: float = Field(
        default=0.15,
        description="Maximum |lag| searched in each analysis window.",
    )
    fine_warp_window_seconds: float = Field(
        default=3.0,
        description="Envelope window length when estimating local lag.",
    )
    fine_warp_step_seconds: float = Field(
        default=1.0,
        description="Step between analysis window centers.",
    )
    fine_warp_neighbor_radius_seconds: float = Field(
        default=4.0,
        description="Neighborhood radius when checking if a lag improves overall.",
    )
    fine_warp_neighbor_step_seconds: float = Field(
        default=1.0,
        description="Step between neighbor windows in the improvement check.",
    )
    fine_warp_min_score: float = Field(
        default=0.35,
        description="Minimum local envelope correlation to consider a lag.",
    )
    fine_warp_min_score_margin: float = Field(
        default=0.05,
        description="Best lag must beat lag=0 by at least this correlation margin.",
    )
    fine_warp_min_improvement: float = Field(
        default=0.02,
        description=(
            "Minimum neighborhood mean-correlation gain vs lag=0 to accept a knot."
        ),
    )
    fine_warp_max_lag_jump_seconds: float = Field(
        default=0.04,
        description="Drop knots whose lag jumps more than this vs the previous kept knot.",
    )
    fine_warp_interp_gap_seconds: float = Field(
        default=3.0,
        description="Interpolate lag only between knots at most this far apart.",
    )

    # Processing report / manual-review thresholds
    report_low_confidence: float = Field(
        default=0.45,
        description="Segment confidence below this is listed and can force review.",
    )
    report_min_global_confidence: float = Field(
        default=0.35,
        description="Alignment confidence below this sets needs_manual_review.",
    )
    report_duration_tolerance_seconds: float = Field(
        default=0.25,
        description="Allowed |output_duration - base_duration| before a finding.",
    )
    report_min_gap_seconds: float = Field(
        default=0.5,
        description="Base coverage holes shorter than this are ignored.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        Cached ``Settings`` instance loaded from the environment.
    """
    return Settings()
