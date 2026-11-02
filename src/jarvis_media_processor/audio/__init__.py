"""Audio extraction and acoustic landmark analysis."""

from jarvis_media_processor.audio.analyze import analyze_media_audio, analyze_wav_file
from jarvis_media_processor.audio.extract import (
    FFmpegNotFoundError,
    build_extract_command,
    extract_analysis_wav,
)
from jarvis_media_processor.audio.landmarks import detect_energy_landmarks, rms_envelope
from jarvis_media_processor.audio.models import AcousticLandmark, AudioAnalysisArtifact

__all__ = [
    "AcousticLandmark",
    "AudioAnalysisArtifact",
    "FFmpegNotFoundError",
    "analyze_media_audio",
    "analyze_wav_file",
    "build_extract_command",
    "detect_energy_landmarks",
    "extract_analysis_wav",
    "rms_envelope",
]
