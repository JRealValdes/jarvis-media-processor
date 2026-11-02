"""Match sampled frames between base and donor via perceptual hashes."""

from __future__ import annotations

import numpy as np

from jarvis_media_processor.video.activity import (
    ActivityEnvelope,
    activity_peak_times,
    local_activity_score,
)
from jarvis_media_processor.video.confirm import frame_distance_at, nearest_frame
from jarvis_media_processor.video.hashing import distance_to_confidence, phash_distance
from jarvis_media_processor.video.models import FrameCorrespondence, SampledFrame


def match_frames(
    base_frames: list[SampledFrame],
    donor_frames: list[SampledFrame],
    *,
    max_distance: int = 12,
    max_matches_per_base: int = 3,
    base_shot_times: set[float] | None = None,
    donor_shot_times: set[float] | None = None,
    shot_confidence_boost: float = 0.15,
) -> list[FrameCorrespondence]:
    """Find ranked base↔donor frame correspondences (brute-force pHash).

    Prefer :func:`match_frames_two_phase` when activity envelopes are available:
    derivatives propose times, frames confirm. This brute path remains as a
    dense fallback and for tests that only have hashes.

    For each base frame, keep up to ``max_matches_per_base`` donor frames with
    Hamming distance ``<= max_distance``, sorted by distance then time.
    Matches involving detected shot-boundary times are flagged and slightly
    boosted so they act as stronger temporal anchors downstream.
    """
    base_shots = base_shot_times or set()
    donor_shots = donor_shot_times or set()
    matches: list[FrameCorrespondence] = []
    for base in base_frames:
        candidates: list[FrameCorrespondence] = []
        for donor in donor_frames:
            distance = phash_distance(base.phash, donor.phash)
            if distance > max_distance:
                continue
            confidence = distance_to_confidence(distance, max_distance=max_distance)
            shot_anchor = (
                round(base.time_seconds, 3) in base_shots
                or round(donor.time_seconds, 3) in donor_shots
            )
            if shot_anchor:
                confidence = min(1.0, confidence + shot_confidence_boost)
            candidates.append(
                FrameCorrespondence(
                    base_time=base.time_seconds,
                    donor_time=donor.time_seconds,
                    distance=distance,
                    confidence=confidence,
                    base_frame=base.path,
                    donor_frame=donor.path,
                    shot_anchor=shot_anchor,
                )
            )
        candidates.sort(key=lambda m: (m.distance, m.donor_time))
        matches.extend(candidates[:max_matches_per_base])

    matches.sort(key=lambda m: (-m.confidence, m.base_time, m.donor_time))
    return matches


def match_frames_two_phase(
    base_frames: list[SampledFrame],
    donor_frames: list[SampledFrame],
    base_activity: ActivityEnvelope,
    donor_activity: ActivityEnvelope,
    *,
    speed: float = 1.0,
    offset_seconds: float = 0.0,
    max_distance: int = 12,
    max_matches_per_base: int = 3,
    half_window: float = 12.0,
    min_activity_score: float = 0.25,
    donor_search_seconds: float = 40.0,
    donor_search_step: float = 1.0,
    base_shot_times: set[float] | None = None,
    donor_shot_times: set[float] | None = None,
    shot_confidence_boost: float = 0.15,
) -> list[FrameCorrespondence]:
    """Two-phase visual match: activity derivatives propose, frames confirm.

    1. Walk characteristic base times (shots + activity peaks + a light grid).
    2. For each, search nearby donor times under ``donor ≈ speed·base + offset``
       using local activity-curve correlation.
    3. Keep a candidate only when the nearest sampled frames also match by pHash.

    This rejects derivative lookalikes (similar motion, different picture) that
    a curve-only lock would accept.
    """
    base_shots = base_shot_times or set()
    donor_shots = donor_shot_times or set()
    if not base_frames or not donor_frames:
        return []

    period = float(
        min(base_activity.sample_period, donor_activity.sample_period) or 1.0
    )
    hop = max(period, 2.0)
    base_start = float(base_activity.times[0]) + half_window
    base_end = float(base_activity.times[-1]) - half_window
    if base_end <= base_start:
        return []

    anchors: list[float] = []
    for t in sorted(base_shots):
        if base_start <= t <= base_end:
            anchors.append(float(t))
    for t in activity_peak_times(base_activity, min_prominence=0.12):
        if base_start <= t <= base_end and (
            not anchors or abs(t - anchors[-1]) >= hop
        ):
            anchors.append(float(t))
    # Light chronological fill so quiet stretches still get proposals.
    for t in np.arange(base_start, base_end + hop * 0.5, hop * 4):
        ft = float(t)
        if not anchors or min(abs(ft - a) for a in anchors) >= hop * 2:
            anchors.append(ft)
    anchors.sort()

    matches: list[FrameCorrespondence] = []
    for base_t in anchors:
        scored: list[tuple[float, float]] = []  # (activity_score, donor_t)
        center = speed * base_t + offset_seconds
        donor_lo = center - donor_search_seconds
        donor_hi = center + donor_search_seconds
        step = max(donor_search_step, period)
        t = donor_lo
        while t <= donor_hi:
            act = local_activity_score(
                base_activity,
                donor_activity,
                base_time=base_t,
                donor_time=float(t),
                speed=speed,
                half_window=half_window,
            )
            if act >= min_activity_score:
                scored.append((act, float(t)))
            t += step
        scored.sort(key=lambda x: (-x[0], x[1]))

        confirmed: list[FrameCorrespondence] = []
        seen_donor: set[float] = set()
        for act_score, donor_t in scored:
            if len(confirmed) >= max_matches_per_base:
                break
            distance = frame_distance_at(
                base_frames,
                donor_frames,
                base_time=base_t,
                donor_time=donor_t,
            )
            if distance is None or distance > max_distance:
                continue
            base_f = nearest_frame(base_frames, base_t)
            donor_f = nearest_frame(donor_frames, donor_t)
            if base_f is None or donor_f is None:
                continue
            donor_key = round(donor_f.time_seconds, 3)
            if donor_key in seen_donor:
                continue
            seen_donor.add(donor_key)
            confidence = distance_to_confidence(distance, max_distance=max_distance)
            # Blend a little activity agreement into confidence.
            confidence = min(1.0, 0.75 * confidence + 0.25 * act_score)
            shot_anchor = (
                round(base_f.time_seconds, 3) in base_shots
                or round(donor_f.time_seconds, 3) in donor_shots
            )
            if shot_anchor:
                confidence = min(1.0, confidence + shot_confidence_boost)
            confirmed.append(
                FrameCorrespondence(
                    base_time=base_f.time_seconds,
                    donor_time=donor_f.time_seconds,
                    distance=distance,
                    confidence=confidence,
                    base_frame=base_f.path,
                    donor_frame=donor_f.path,
                    shot_anchor=shot_anchor,
                )
            )
        matches.extend(confirmed)

    matches.sort(key=lambda m: (-m.confidence, m.base_time, m.donor_time))
    return matches
