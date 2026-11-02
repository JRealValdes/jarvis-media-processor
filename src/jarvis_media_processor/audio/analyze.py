"""High-level audio analysis: extract + landmarks → artifact."""

from __future__ import annotations

import logging
from pathlib import Path

from jarvis_media_processor.audio.extract import extract_analysis_wav
from jarvis_media_processor.audio.landmarks import detect_energy_landmarks
from jarvis_media_processor.audio.models import AudioAnalysisArtifact
from jarvis_media_processor.audio.wavutil import read_pcm16_wav
from jarvis_media_processor.config import Settings, get_settings

logger = logging.getLogger(__name__)


def analyze_media_audio(
    source: Path | str,
    workspace: Path | str,
    *,
    role: str | None = None,
    settings: Settings | None = None,
    keep_wav: bool = True,
) -> AudioAnalysisArtifact:
    """Extract analysis audio from ``source`` and compute acoustic landmarks.

    Args:
        source: Input media path (untouched).
        workspace: Directory for intermediate WAV / artifacts.
        role: Optional ``base`` / ``donor`` label.
        settings: Optional settings override.
        keep_wav: If False, delete the extracted WAV after analysis.

    Returns:
        Serializable ``AudioAnalysisArtifact``.
    """
    cfg = settings or get_settings()
    src = Path(source)
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)

    stem = src.stem
    role_suffix = f"_{role}" if role else ""
    wav_path = work / f"{stem}{role_suffix}_analysis.wav"

    extract_analysis_wav(src, wav_path, settings=cfg)
    samples, sample_rate = read_pcm16_wav(wav_path)
    landmarks, envelope, hop = detect_energy_landmarks(
        samples,
        sample_rate,
        hop_seconds=cfg.audio_landmark_hop_seconds,
        frame_seconds=cfg.audio_landmark_frame_seconds,
        min_distance_seconds=cfg.audio_landmark_min_distance_seconds,
        prominence=cfg.audio_landmark_prominence,
        max_landmarks=cfg.audio_landmark_max_count,
    )

    duration = float(samples.size / sample_rate) if sample_rate else 0.0
    artifact = AudioAnalysisArtifact(
        source_path=src.resolve(),
        role=role,
        wav_path=wav_path.resolve() if keep_wav else None,
        sample_rate=sample_rate,
        channels=cfg.audio_analysis_channels,
        duration_seconds=duration,
        landmarks=landmarks,
        energy_hop_seconds=hop,
        energy_envelope=[float(x) for x in envelope.tolist()],
        params={
            "hop_seconds": cfg.audio_landmark_hop_seconds,
            "frame_seconds": cfg.audio_landmark_frame_seconds,
            "min_distance_seconds": cfg.audio_landmark_min_distance_seconds,
            "prominence": cfg.audio_landmark_prominence,
            "max_landmarks": cfg.audio_landmark_max_count,
            "analysis_sample_rate": cfg.audio_analysis_sample_rate,
        },
    )

    artifact_path = work / f"{stem}{role_suffix}_audio_analysis.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Audio analysis for %s: %d landmarks -> %s",
        src.name,
        len(landmarks),
        artifact_path,
    )

    if not keep_wav and wav_path.is_file():
        wav_path.unlink()

    return artifact


def analyze_wav_file(
    wav_path: Path | str,
    *,
    source_path: Path | str | None = None,
    role: str | None = None,
    settings: Settings | None = None,
) -> AudioAnalysisArtifact:
    """Analyze an existing WAV without calling FFmpeg (useful for unit tests)."""
    cfg = settings or get_settings()
    path = Path(wav_path)
    samples, sample_rate = read_pcm16_wav(path)
    landmarks, envelope, hop = detect_energy_landmarks(
        samples,
        sample_rate,
        hop_seconds=cfg.audio_landmark_hop_seconds,
        frame_seconds=cfg.audio_landmark_frame_seconds,
        min_distance_seconds=cfg.audio_landmark_min_distance_seconds,
        prominence=cfg.audio_landmark_prominence,
        max_landmarks=cfg.audio_landmark_max_count,
    )
    duration = float(samples.size / sample_rate) if sample_rate else 0.0
    return AudioAnalysisArtifact(
        source_path=Path(source_path).resolve() if source_path else path.resolve(),
        role=role,
        wav_path=path.resolve(),
        sample_rate=sample_rate,
        channels=1,
        duration_seconds=duration,
        landmarks=landmarks,
        energy_hop_seconds=hop,
        energy_envelope=[float(x) for x in envelope.tolist()],
        params={
            "hop_seconds": cfg.audio_landmark_hop_seconds,
            "frame_seconds": cfg.audio_landmark_frame_seconds,
            "min_distance_seconds": cfg.audio_landmark_min_distance_seconds,
            "prominence": cfg.audio_landmark_prominence,
            "max_landmarks": cfg.audio_landmark_max_count,
        },
    )
