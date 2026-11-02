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
    filter_pairs_excluding_donor,
    landmark_time_pairs,
    merge_evidence,
    visual_time_pairs,
)
from jarvis_media_processor.alignment.models import AlignmentDocument, TimelineInterval
from jarvis_media_processor.alignment.unrelated import (
    apply_unrelated_to_document,
    complementary_donor_intervals,
    detect_curve_gaps,
    slice_envelope,
)
from jarvis_media_processor.audio import analyze_media_audio, analyze_wav_file
from jarvis_media_processor.audio.models import AcousticLandmark, AudioAnalysisArtifact
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.video import match_media_visuals
from jarvis_media_processor.video.activity import ActivityEnvelope
from jarvis_media_processor.video.confirm import make_frame_confirmer
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
    unrelated_donor_intervals: list[TimelineInterval] | None = None,
    missing_base_intervals: list[TimelineInterval] | None = None,
    base_path: Path | None = None,
    donor_path: Path | None = None,
    inlier_tolerance: float = 0.12,
    visual_inlier_tolerance: float = 1.5,
    min_visual_pairs: int = 8,
    visual_min_inlier_ratio: float = 0.55,
    audio_downweight: float = 0.2,
    speed_min: float = 0.85,
    speed_max: float = 1.2,
    audio_refine_window_seconds: float = 2.0,
    audio_refine_min_score: float = 0.05,
    refine_audio_offset: bool = True,
) -> AlignmentDocument:
    """Run visual-first escalating sync using precomputed analysis artifacts."""
    excluded = [
        (item.start, item.end)
        for item in (unrelated_donor_intervals or [])
        if item.timeline == "donor"
    ]
    audio_pairs = filter_pairs_excluding_donor(
        landmark_time_pairs(base_audio.landmarks, donor_audio.landmarks),
        excluded,
    )
    phash_pairs = filter_pairs_excluding_donor(
        visual_time_pairs(visual_matches or []),
        excluded,
    )
    visual_pairs = merge_evidence(
        filter_pairs_excluding_donor(activity_pairs or [], excluded),
        phash_pairs,
    )
    if visual_matches or activity_pairs:
        doc = estimate_alignment_visual_first(
            visual_pairs,
            audio_pairs,
            base_duration=base_audio.duration_seconds,
            donor_duration=donor_audio.duration_seconds,
            audio_inlier_tolerance=inlier_tolerance,
            visual_inlier_tolerance=visual_inlier_tolerance,
            min_visual_pairs=min_visual_pairs,
            visual_min_inlier_ratio=visual_min_inlier_ratio,
            audio_downweight=audio_downweight,
            speed_min=speed_min,
            speed_max=speed_max,
        )
    else:
        doc = estimate_alignment(
            audio_pairs,
            base_duration=base_audio.duration_seconds,
            donor_duration=donor_audio.duration_seconds,
            inlier_tolerance=inlier_tolerance,
            speed_min=speed_min,
            speed_max=speed_max,
        )
    if unrelated_donor_intervals or missing_base_intervals:
        apply_unrelated_to_document(
            doc,
            unrelated_donor_intervals or [],
            missing_base=missing_base_intervals,
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
    *,
    settings: Settings | None = None,
) -> tuple[
    list[TimePair],
    ActivitySyncResult | None,
    list[TimelineInterval],
    list[TimelineInterval],
]:
    cfg = settings or get_settings()
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
        return [], None, [], []
    frame_confirm = None
    if report.base_frames and report.donor_frames:
        frame_confirm = make_frame_confirmer(
            report.base_frames,
            report.donor_frames,
            max_distance=int(cfg.video_match_max_distance),
            max_time_delta=max(2.0, 1.5 / max(cfg.video_sample_fps, 0.1)),
        )
    walk = detect_curve_gaps(
        base_env,
        donor_env,
        window_seconds=cfg.alignment_unrelated_window_seconds,
        hop_seconds=cfg.alignment_unrelated_hop_seconds,
        min_score=cfg.alignment_unrelated_min_score,
        min_span_seconds=cfg.alignment_unrelated_min_span_seconds,
        max_surplus_seconds=cfg.alignment_unrelated_max_surplus_seconds,
        missing_base_fill_seconds=cfg.alignment_missing_base_fill_seconds,
        confirm_seconds=cfg.alignment_curve_confirm_seconds,
        frame_confirm=frame_confirm,
    )
    unrelated = walk.unrelated_donor_intervals
    missing_base = walk.missing_base_intervals
    for note in walk.notes:
        logger.info("Curve walk: %s", note)

    # Fit activity sync per contiguous donor chunk outside ads.
    coarse = estimate_speed_offset_activity(
        base_env,
        donor_env,
        speed_min=cfg.alignment_speed_min,
        speed_max=cfg.alignment_speed_max,
    )
    donor_duration = float(donor_env.times[-1]) if donor_env.times.size else 0.0
    chunks = complementary_donor_intervals(donor_duration, unrelated)
    pairs: list[TimePair] = []
    sync = coarse
    best_chunk_score = -1.0
    for start, end in chunks:
        chunk = slice_envelope(donor_env, start, end)
        if chunk is None:
            continue
        chunk_sync = estimate_speed_offset_activity(
            base_env,
            chunk,
            speed_min=cfg.alignment_speed_min,
            speed_max=cfg.alignment_speed_max,
        )
        chunk_pairs = activity_time_pairs(base_env, chunk, sync=chunk_sync)
        pairs.extend(chunk_pairs)
        if chunk_sync.score > best_chunk_score:
            best_chunk_score = chunk_sync.score
            sync = chunk_sync
    if not chunks:
        pairs = activity_time_pairs(base_env, donor_env, sync=sync)
    logger.info(
        "Activity sync method=%s speed=%.4f offset=%.3fs score=%.3f pairs=%d "
        "unrelated=%d missing_base=%d chunks=%d",
        sync.method,
        sync.speed,
        sync.offset_seconds,
        sync.score,
        len(pairs),
        len(unrelated),
        len(missing_base),
        len(chunks),
    )
    return pairs, sync, unrelated, missing_base


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
    unrelated_donor: list[TimelineInterval] = []
    missing_base: list[TimelineInterval] = []
    if use_video:
        try:
            report = match_media_visuals(base, donor, work / "visual", settings=cfg)
            visual_matches = report.matches
            (
                activity_pairs,
                activity_sync,
                unrelated_donor,
                missing_base,
            ) = _activity_pairs_from_report(report, settings=cfg)
            if activity_sync is not None:
                (work / "activity_sync.json").write_text(
                    json.dumps(
                        {
                            "method": activity_sync.method,
                            "speed": activity_sync.speed,
                            "offset_seconds": activity_sync.offset_seconds,
                            "score": activity_sync.score,
                            "pair_count": len(activity_pairs),
                            "unrelated_donor_intervals": [
                                {
                                    "start": item.start,
                                    "end": item.end,
                                    "reason": item.reason,
                                }
                                for item in unrelated_donor
                            ],
                            "missing_base_intervals": [
                                {
                                    "start": item.start,
                                    "end": item.end,
                                    "reason": item.reason,
                                }
                                for item in missing_base
                            ],
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
        unrelated_donor_intervals=unrelated_donor,
        missing_base_intervals=missing_base,
        base_path=base.resolve(),
        donor_path=donor.resolve(),
        inlier_tolerance=cfg.alignment_inlier_tolerance_seconds,
        visual_inlier_tolerance=cfg.alignment_visual_inlier_tolerance_seconds,
        min_visual_pairs=cfg.alignment_visual_min_pairs,
        visual_min_inlier_ratio=cfg.alignment_visual_min_inlier_ratio,
        audio_downweight=cfg.alignment_audio_downweight,
        speed_min=cfg.alignment_speed_min,
        speed_max=cfg.alignment_speed_max,
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
