"""High-level visual matching for a base/donor pair."""

from __future__ import annotations

import logging
from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.video.activity import build_fused_activity_envelope
from jarvis_media_processor.video.extract import extract_sampled_frames
from jarvis_media_processor.video.match import match_frames, match_frames_two_phase
from jarvis_media_processor.video.models import VisualMatchReport
from jarvis_media_processor.video.shots import shot_boundary_times

logger = logging.getLogger(__name__)

# Prefer two-phase matches; fall back to brute pHash if too sparse.
_MIN_TWO_PHASE_MATCHES = 8


def match_media_visuals(
    base_path: Path | str,
    donor_path: Path | str,
    workspace: Path | str,
    *,
    settings: Settings | None = None,
) -> VisualMatchReport:
    """Sample frames and match in two phases: activity derivatives, then frames.

    1. Build fused activity envelopes (pHash-diff floor + structural edge change).
    2. Propose base↔donor times from a coarse activity sync + peaks/shots.
    3. Confirm each proposal with actual frame pHash (reject lookalike curves).
    4. If confirmed matches are too sparse, fall back to brute-force pHash.

    Detected shot cuts boost confidence so they act as stronger temporal anchors.
    """
    cfg = settings or get_settings()
    base = Path(base_path)
    donor = Path(donor_path)
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)

    base_frames = extract_sampled_frames(
        base,
        work / "base_frames",
        role="base",
        settings=cfg,
    )
    donor_frames = extract_sampled_frames(
        donor,
        work / "donor_frames",
        role="donor",
        settings=cfg,
    )
    base_shots = shot_boundary_times(
        base_frames, min_distance=cfg.video_shot_min_distance
    )
    donor_shots = shot_boundary_times(
        donor_frames, min_distance=cfg.video_shot_min_distance
    )
    # Fuse encode-local pHash cuts with encode-robust structural change.
    base_activity = build_fused_activity_envelope(base_frames)
    donor_activity = build_fused_activity_envelope(donor_frames)

    # Lazy import avoids video ↔ alignment circular imports at module load.
    from jarvis_media_processor.alignment.activity_sync import (
        estimate_speed_offset_activity,
    )

    coarse = estimate_speed_offset_activity(
        base_activity,
        donor_activity,
        speed_min=cfg.alignment_speed_min,
        speed_max=cfg.alignment_speed_max,
    )
    matches = match_frames_two_phase(
        base_frames,
        donor_frames,
        base_activity,
        donor_activity,
        speed=coarse.speed,
        offset_seconds=coarse.offset_seconds,
        max_distance=cfg.video_match_max_distance,
        max_matches_per_base=cfg.video_match_max_per_base,
        half_window=cfg.alignment_unrelated_window_seconds,
        min_activity_score=cfg.alignment_unrelated_min_score,
        donor_search_seconds=min(120.0, cfg.alignment_unrelated_max_surplus_seconds),
        base_shot_times=base_shots,
        donor_shot_times=donor_shots,
    )
    match_method = "activity_then_frames"
    two_phase_count = len(matches)
    if two_phase_count < _MIN_TWO_PHASE_MATCHES:
        brute = match_frames(
            base_frames,
            donor_frames,
            max_distance=cfg.video_match_max_distance,
            max_matches_per_base=cfg.video_match_max_per_base,
            base_shot_times=base_shots,
            donor_shot_times=donor_shots,
        )
        if len(brute) > two_phase_count:
            logger.info(
                "Two-phase visual match sparse (%d); using brute-force pHash (%d).",
                two_phase_count,
                len(brute),
            )
            matches = brute
            match_method = "phash_bruteforce_fallback"

    report = VisualMatchReport(
        base_path=base.resolve(),
        donor_path=donor.resolve(),
        base_frames=base_frames,
        donor_frames=donor_frames,
        matches=matches,
        base_shot_times=sorted(base_shots),
        donor_shot_times=sorted(donor_shots),
        base_activity_times=base_activity.times.tolist(),
        base_activity_values=base_activity.values.tolist(),
        donor_activity_times=donor_activity.times.tolist(),
        donor_activity_values=donor_activity.values.tolist(),
        params={
            "sample_fps": cfg.video_sample_fps,
            "phash_size": cfg.video_phash_size,
            "normalize_size": cfg.video_normalize_size,
            "crop_letterbox": int(cfg.video_crop_letterbox),
            "max_distance": cfg.video_match_max_distance,
            "max_per_base": cfg.video_match_max_per_base,
            "shot_min_distance": cfg.video_shot_min_distance,
            "base_shot_count": len(base_shots),
            "donor_shot_count": len(donor_shots),
            "activity_sample_period": base_activity.sample_period,
            "activity_method": "fused_max_phash_structural",
            "activity_structural_weight": 0.5,
            "match_method": match_method,
            "activity_sync_speed": coarse.speed,
            "activity_sync_offset": coarse.offset_seconds,
            "activity_sync_score": coarse.score,
        },
    )
    out = work / "visual_match_report.json"
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Visual match %s vs %s: %d matches (%s) -> %s",
        base.name,
        donor.name,
        len(matches),
        match_method,
        out,
    )
    return report
