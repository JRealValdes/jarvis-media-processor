"""High-level alignment detection for base/donor media pairs."""

from __future__ import annotations

import logging
from pathlib import Path

from jarvis_media_processor.alignment.estimate import estimate_alignment
from jarvis_media_processor.alignment.evidence import (
    landmark_time_pairs,
    merge_evidence,
    visual_time_pairs,
)
from jarvis_media_processor.alignment.models import AlignmentDocument
from jarvis_media_processor.audio import analyze_media_audio, analyze_wav_file
from jarvis_media_processor.audio.models import AcousticLandmark, AudioAnalysisArtifact
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.video import match_media_visuals
from jarvis_media_processor.video.models import FrameCorrespondence

logger = logging.getLogger(__name__)


def detect_alignment_from_artifacts(
    base_audio: AudioAnalysisArtifact,
    donor_audio: AudioAnalysisArtifact,
    *,
    visual_matches: list[FrameCorrespondence] | None = None,
    base_path: Path | None = None,
    donor_path: Path | None = None,
    inlier_tolerance: float = 0.12,
) -> AlignmentDocument:
    """Run escalating sync using precomputed analysis artifacts."""
    pairs = merge_evidence(
        landmark_time_pairs(base_audio.landmarks, donor_audio.landmarks),
        visual_time_pairs(visual_matches or []),
    )
    doc = estimate_alignment(
        pairs,
        base_duration=base_audio.duration_seconds,
        donor_duration=donor_audio.duration_seconds,
        inlier_tolerance=inlier_tolerance,
    )
    doc.base_path = base_path or base_audio.source_path
    doc.donor_path = donor_path or donor_audio.source_path
    return doc


def detect_alignment(
    base_path: Path | str,
    donor_path: Path | str,
    workspace: Path | str,
    *,
    settings: Settings | None = None,
    use_video: bool = True,
) -> AlignmentDocument:
    """Analyze base/donor media and emit an escalating alignment document.

    Args:
        base_path: Base media (timeline authority).
        donor_path: Donor media (audio to graft).
        workspace: Working directory for intermediate artifacts.
        settings: Optional settings override.
        use_video: If True, also gather visual match evidence when possible.

    Returns:
        ``AlignmentDocument`` written under ``workspace``.
    """
    cfg = settings or get_settings()
    base = Path(base_path)
    donor = Path(donor_path)
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)

    base_audio = analyze_media_audio(base, work / "audio_base", role="base", settings=cfg)
    donor_audio = analyze_media_audio(donor, work / "audio_donor", role="donor", settings=cfg)

    visual_matches: list[FrameCorrespondence] = []
    if use_video:
        try:
            report = match_media_visuals(base, donor, work / "visual", settings=cfg)
            visual_matches = report.matches
        except Exception as exc:  # noqa: BLE001 - video is optional evidence
            logger.warning("Visual matching skipped: %s", exc)

    doc = detect_alignment_from_artifacts(
        base_audio,
        donor_audio,
        visual_matches=visual_matches,
        base_path=base.resolve(),
        donor_path=donor.resolve(),
        inlier_tolerance=cfg.alignment_inlier_tolerance_seconds,
    )

    out = work / "alignment.json"
    out.write_text(doc.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Alignment strategy=%s confidence=%.2f -> %s", doc.strategy, doc.confidence, out)
    return doc


def detect_alignment_from_landmarks(
    base_landmarks: list[AcousticLandmark],
    donor_landmarks: list[AcousticLandmark],
    *,
    base_duration: float | None = None,
    donor_duration: float | None = None,
    inlier_tolerance: float = 0.12,
) -> AlignmentDocument:
    """Convenience for unit tests: estimate alignment from landmark lists only."""
    base_audio = AudioAnalysisArtifact(
        source_path=Path("base"),
        sample_rate=16_000,
        channels=1,
        duration_seconds=base_duration or 0.0,
        landmarks=base_landmarks,
        energy_hop_seconds=0.01,
    )
    donor_audio = AudioAnalysisArtifact(
        source_path=Path("donor"),
        sample_rate=16_000,
        channels=1,
        duration_seconds=donor_duration or 0.0,
        landmarks=donor_landmarks,
        energy_hop_seconds=0.01,
    )
    return detect_alignment_from_artifacts(
        base_audio,
        donor_audio,
        inlier_tolerance=inlier_tolerance,
    )


def detect_alignment_from_wavs(
    base_wav: Path | str,
    donor_wav: Path | str,
    *,
    settings: Settings | None = None,
) -> AlignmentDocument:
    """Analyze two WAV files (no container demux) and estimate alignment."""
    cfg = settings or get_settings()
    base_audio = analyze_wav_file(base_wav, role="base", settings=cfg)
    donor_audio = analyze_wav_file(donor_wav, role="donor", settings=cfg)
    return detect_alignment_from_artifacts(
        base_audio,
        donor_audio,
        base_path=Path(base_wav).resolve(),
        donor_path=Path(donor_wav).resolve(),
        inlier_tolerance=cfg.alignment_inlier_tolerance_seconds,
    )
