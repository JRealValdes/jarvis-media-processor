"""High-level alignment detection for base/donor media pairs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from jarvis_media_processor.alignment.activity_sync import (
    ActivitySyncResult,
    activity_time_pairs,
    estimate_speed_offset_activity,
)
from jarvis_media_processor.alignment.audio_refine import refine_alignment_audio_offset
from jarvis_media_processor.alignment.estimate import (
    estimate_alignment,
    estimate_alignment_visual_first,
)
from jarvis_media_processor.alignment.evidence import (
    TimePair,
    landmark_time_pairs,
    merge_evidence,
    visual_time_pairs,
)
from jarvis_media_processor.alignment.models import AlignmentDocument
from jarvis_media_processor.audio import analyze_media_audio, analyze_wav_file
from jarvis_media_processor.audio.models import AcousticLandmark, AudioAnalysisArtifact
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.video import match_media_visuals
from jarvis_media_processor.video.activity import ActivityEnvelope
from jarvis_media_processor.video.models import FrameCorrespondence, VisualMatchReport

logger = logging.getLogger(__name__)


def _envelope_from_report(
    times: list[float],
    values: list[float],
    *,
    sample_period: float,
) -> ActivityEnvelope | None:
    if len(times) < 4 or len(times) != len(values):
        return None
    return ActivityEnvelope(
        times=np.asarray(times, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        sample_period=sample_period,
    )


def detect_alignment_from_artifacts(
    base_audio: AudioAnalysisArtifact,
    donor_audio: AudioAnalysisArtifact,
    *,
    visual_matches: list[FrameCorrespondence] | None = None,
    activity_pairs: list[TimePair] | None = None,
    base_path: Path | None = None,
    donor_path: Path | None = None,
    inlier_tolerance: float = 0.12,
    visual_inlier_tolerance: float = 1.5,
    min_visual_pairs: int = 8,
    audio_refine_window_seconds: float = 2.0,
    audio_refine_min_score: float = 0.05,
    refine_audio_offset: bool = True,
) -> AlignmentDocument:
    """Run visual-first escalating sync using precomputed analysis artifacts."""
    audio_pairs = landmark_time_pairs(base_audio.landmarks, donor_audio.landmarks)
    phash_pairs = visual_time_pairs(visual_matches or [])
    visual_pairs = merge_evidence(activity_pairs or [], phash_pairs)
    if visual_matches or activity_pairs:
        doc = estimate_alignment_visual_first(
            visual_pairs,
            audio_pairs,
            base_duration=base_audio.duration_seconds,
            donor_duration=donor_audio.duration_seconds,
            audio_inlier_tolerance=inlier_tolerance,
            visual_inlier_tolerance=visual_inlier_tolerance,
            min_visual_pairs=min_visual_pairs,
        )
    else:
        doc = estimate_alignment(
            audio_pairs,
            base_duration=base_audio.duration_seconds,
            donor_duration=donor_audio.duration_seconds,
            inlier_tolerance=inlier_tolerance,
        )
    if refine_audio_offset and (visual_matches or activity_pairs):
        doc, refinement = refine_alignment_audio_offset(
            doc,
            base_audio,
            donor_audio,
            window_seconds=audio_refine_window_seconds,
            min_score=audio_refine_min_score,
        )
        if refinement.applied:
            logger.info(
                "Audio fine-offset Δ=%+.3fs score=%.3f",
                refinement.delta_seconds,
                refinement.score,
            )
    doc.base_path = base_path or base_audio.source_path
    doc.donor_path = donor_path or donor_audio.source_path
    return doc


def _activity_pairs_from_report(
    report: VisualMatchReport,
) -> tuple[list[TimePair], ActivitySyncResult | None]:
    period = float(report.params.get("activity_sample_period", 1.0) or 1.0)
    base_env = _envelope_from_report(
        report.base_activity_times,
        report.base_activity_values,
        sample_period=period,
    )
    donor_env = _envelope_from_report(
        report.donor_activity_times,
        report.donor_activity_values,
        sample_period=period,
    )
    if base_env is None or donor_env is None:
        return [], None
    sync = estimate_speed_offset_activity(base_env, donor_env)
    pairs = activity_time_pairs(base_env, donor_env, sync=sync)
    logger.info(
        "Activity sync method=%s speed=%.4f offset=%.3fs score=%.3f pairs=%d",
        sync.method,
        sync.speed,
        sync.offset_seconds,
        sync.score,
        len(pairs),
    )
    return pairs, sync


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
    activity_pairs: list[TimePair] = []
    if use_video:
        try:
            report = match_media_visuals(base, donor, work / "visual", settings=cfg)
            visual_matches = report.matches
            activity_pairs, activity_sync = _activity_pairs_from_report(report)
            if activity_sync is not None:
                (work / "activity_sync.json").write_text(
                    json.dumps(
                        {
                            "method": activity_sync.method,
                            "speed": activity_sync.speed,
                            "offset_seconds": activity_sync.offset_seconds,
                            "score": activity_sync.score,
                            "pair_count": len(activity_pairs),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except Exception as exc:  # noqa: BLE001 - video is optional evidence
            logger.warning("Visual matching skipped: %s", exc)

    doc = detect_alignment_from_artifacts(
        base_audio,
        donor_audio,
        visual_matches=visual_matches,
        activity_pairs=activity_pairs,
        base_path=base.resolve(),
        donor_path=donor.resolve(),
        inlier_tolerance=cfg.alignment_inlier_tolerance_seconds,
        visual_inlier_tolerance=cfg.alignment_visual_inlier_tolerance_seconds,
        min_visual_pairs=cfg.alignment_visual_min_pairs,
        audio_refine_window_seconds=cfg.alignment_audio_refine_window_seconds,
        audio_refine_min_score=cfg.alignment_audio_refine_min_score,
        refine_audio_offset=bool(visual_matches or activity_pairs),
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
