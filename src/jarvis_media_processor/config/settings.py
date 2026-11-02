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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        Cached ``Settings`` instance loaded from the environment.
    """
    return Settings()
