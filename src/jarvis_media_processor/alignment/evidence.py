"""Build timing evidence from acoustic landmarks and visual matches."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis_media_processor.audio.models import AcousticLandmark
from jarvis_media_processor.video.models import FrameCorrespondence


@dataclass(frozen=True)
class TimePair:
    """A hypothesized correspondence: base_time ↔ donor_time."""

    base_time: float
    donor_time: float
    weight: float
    source: str


def landmark_time_pairs(
    base: list[AcousticLandmark],
    donor: list[AcousticLandmark],
    *,
    max_landmarks: int = 20,
) -> list[TimePair]:
    """Build candidate landmark correspondences for offset/speed voting.

    Takes the strongest landmarks on each side and forms a capped Cartesian
    product. Downstream RANSAC / grid search selects a consistent transform.
    """
    base_sorted = sorted(base, key=lambda lm: lm.strength, reverse=True)[:max_landmarks]
    donor_sorted = sorted(donor, key=lambda lm: lm.strength, reverse=True)[:max_landmarks]
    pairs: list[TimePair] = []
    for b in base_sorted:
        for d in donor_sorted:
            weight = float(min(b.strength, d.strength))
            pairs.append(
                TimePair(
                    base_time=b.time_seconds,
                    donor_time=d.time_seconds,
                    weight=max(weight, 1e-6),
                    source="audio_landmark",
                )
            )
    return pairs


def visual_time_pairs(
    matches: list[FrameCorrespondence],
    *,
    min_confidence: float = 0.4,
) -> list[TimePair]:
    """Convert visual frame matches into timed evidence pairs."""
    pairs: list[TimePair] = []
    for match in matches:
        if match.confidence < min_confidence:
            continue
        pairs.append(
            TimePair(
                base_time=match.base_time,
                donor_time=match.donor_time,
                weight=max(match.confidence, 1e-6),
                source="video_match",
            )
        )
    return pairs


def merge_evidence(*groups: list[TimePair]) -> list[TimePair]:
    """Concatenate evidence groups (audio first callers should pass audio first)."""
    merged: list[TimePair] = []
    for group in groups:
        merged.extend(group)
    return merged
