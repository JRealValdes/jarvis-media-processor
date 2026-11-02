"""Fine audio-offset nudge after a visual global timeline lock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jarvis_media_processor.alignment.models import AlignmentDocument, SyncStrategyLevel
from jarvis_media_processor.audio.models import AudioAnalysisArtifact
from jarvis_media_processor.video.activity import normalize_activity


@dataclass(frozen=True)
class AudioOffsetRefinement:
    """Result of a bounded audio-envelope offset search."""

    delta_seconds: float
    score: float
    window_seconds: float
    applied: bool


_GLOBAL = {
    SyncStrategyLevel.ALREADY_SYNCED,
    SyncStrategyLevel.CONSTANT_OFFSET,
    SyncStrategyLevel.CONSTANT_SPEED,
}


def _envelope_arrays(
    artifact: AudioAnalysisArtifact,
) -> tuple[NDArray[np.float64], float]:
    env = np.asarray(artifact.energy_envelope, dtype=np.float64)
    hop = float(artifact.energy_hop_seconds) if artifact.energy_hop_seconds > 0 else 0.01
    return env, hop


def warp_donor_envelope_to_base(
    donor_env: NDArray[np.floating],
    *,
    donor_hop: float,
    base_length: int,
    base_hop: float,
    speed: float,
    offset: float,
) -> NDArray[np.float64]:
    """Sample donor envelope at ``speed * base_time + offset`` for each base hop."""
    if base_length <= 0 or donor_env.size == 0:
        return np.zeros(max(0, base_length), dtype=np.float64)
    base_times = np.arange(base_length, dtype=np.float64) * base_hop
    donor_times = speed * base_times + offset
    donor_index = donor_times / max(donor_hop, 1e-9)
    # Extrapolate as silence outside donor support.
    warped = np.interp(
        donor_index,
        np.arange(donor_env.size, dtype=np.float64),
        np.asarray(donor_env, dtype=np.float64),
        left=0.0,
        right=0.0,
    )
    return np.asarray(warped, dtype=np.float64)


def estimate_fine_audio_offset(
    base_audio: AudioAnalysisArtifact,
    donor_audio: AudioAnalysisArtifact,
    *,
    speed: float,
    offset: float,
    window_seconds: float = 2.0,
) -> AudioOffsetRefinement:
    """Search ``Δoffset`` in ``±window_seconds`` via energy-envelope cross-correlation.

    The donor envelope is first warped onto the base timeline with the visual
    ``(speed, offset)``. Only a residual lag is estimated.
    """
    base_env, base_hop = _envelope_arrays(base_audio)
    donor_env, donor_hop = _envelope_arrays(donor_audio)
    if base_env.size < 8 or donor_env.size < 8:
        return AudioOffsetRefinement(0.0, 0.0, window_seconds, applied=False)

    warped = warp_donor_envelope_to_base(
        donor_env,
        donor_hop=donor_hop,
        base_length=base_env.size,
        base_hop=base_hop,
        speed=speed,
        offset=offset,
    )
    a = normalize_activity(base_env)
    b = normalize_activity(warped)
    if float(np.linalg.norm(a)) < 1e-9 or float(np.linalg.norm(b)) < 1e-9:
        return AudioOffsetRefinement(0.0, 0.0, window_seconds, applied=False)

    max_lag = max(1, int(round(window_seconds / max(base_hop, 1e-9))))
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(len(b) - 1), len(a), dtype=np.int64)
    # Restrict to ±window.
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return AudioOffsetRefinement(0.0, 0.0, window_seconds, applied=False)
    restricted = corr.copy()
    restricted[~mask] = -np.inf
    best_i = int(np.argmax(restricted))
    lag = int(lags[best_i])
    # Same sign convention as visual activity sync: negate correlate lag.
    delta = -float(lag) * base_hop
    score = float(
        abs(corr[best_i]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )
    score = max(0.0, min(1.0, score))
    return AudioOffsetRefinement(
        delta_seconds=delta,
        score=score,
        window_seconds=window_seconds,
        applied=True,
    )


def apply_offset_delta(doc: AlignmentDocument, delta_seconds: float) -> AlignmentDocument:
    """Return a copy of ``doc`` with ``delta`` added to the global/segment offsets.

    Model remains ``donor ≈ speed * base + offset``; only ``offset`` shifts.
    Segment donor endpoints are remapped with the updated offset (speed kept).
    """
    if abs(delta_seconds) < 1e-9:
        return doc.model_copy(deep=True)

    updated = doc.model_copy(deep=True)
    if updated.global_offset_seconds is not None:
        updated.global_offset_seconds = float(updated.global_offset_seconds + delta_seconds)

    new_segments = []
    for seg in updated.segments:
        speed = seg.speed if seg.speed > 0 else 1.0
        new_offset = float(seg.offset_seconds + delta_seconds)
        donor_start = float(speed * seg.base_start + new_offset)
        donor_end = float(speed * seg.base_end + new_offset)
        # Keep donor times non-negative; clamp span if needed.
        if donor_end < donor_start:
            donor_end = donor_start
        if donor_start < 0.0:
            # Shrink from the left on the base timeline.
            base_start = max(seg.base_start, (0.0 - new_offset) / speed)
            donor_start = 0.0
            donor_end = float(speed * seg.base_end + new_offset)
            if donor_end < donor_start:
                donor_end = donor_start + 1e-3
            new_segments.append(
                seg.model_copy(
                    update={
                        "base_start": float(base_start),
                        "offset_seconds": new_offset,
                        "donor_start": donor_start,
                        "donor_end": max(donor_end, donor_start + 1e-3),
                    }
                )
            )
        else:
            new_segments.append(
                seg.model_copy(
                    update={
                        "offset_seconds": new_offset,
                        "donor_start": donor_start,
                        "donor_end": max(donor_end, donor_start + 1e-3),
                    }
                )
            )
    updated.segments = new_segments
    return updated


def refine_alignment_audio_offset(
    doc: AlignmentDocument,
    base_audio: AudioAnalysisArtifact,
    donor_audio: AudioAnalysisArtifact,
    *,
    window_seconds: float = 2.0,
    min_score: float = 0.05,
) -> tuple[AlignmentDocument, AudioOffsetRefinement]:
    """Refine a visual global alignment with a bounded audio-envelope offset.

    No-ops for piecewise/failed strategies (visual map not a single transform).
    """
    if doc.strategy not in _GLOBAL:
        refinement = AudioOffsetRefinement(0.0, 0.0, window_seconds, applied=False)
        return doc, refinement

    speed = float(doc.global_speed if doc.global_speed and doc.global_speed > 0 else 1.0)
    offset = float(doc.global_offset_seconds if doc.global_offset_seconds is not None else 0.0)
    refinement = estimate_fine_audio_offset(
        base_audio,
        donor_audio,
        speed=speed,
        offset=offset,
        window_seconds=window_seconds,
    )
    if not refinement.applied or refinement.score < min_score:
        note = (
            f"Audio fine-offset skipped (score={refinement.score:.3f}, "
            f"window=±{window_seconds:.2f}s)."
        )
        out = doc.model_copy(deep=True)
        out.notes = [*out.notes, note]
        evidence = dict(out.evidence)
        evidence["audio_refine_delta"] = 0.0
        evidence["audio_refine_score"] = round(refinement.score, 4)
        evidence["audio_refine_applied"] = 0
        out.evidence = evidence
        skipped = AudioOffsetRefinement(
            delta_seconds=refinement.delta_seconds,
            score=refinement.score,
            window_seconds=refinement.window_seconds,
            applied=False,
        )
        return out, skipped

    refined = apply_offset_delta(doc, refinement.delta_seconds)
    refined.notes = [
        *refined.notes,
        (
            f"Audio fine-offset Δ={refinement.delta_seconds:+.3f}s "
            f"(score={refinement.score:.3f}, window=±{window_seconds:.2f}s)."
        ),
    ]
    evidence = dict(refined.evidence)
    evidence["audio_refine_delta"] = round(refinement.delta_seconds, 4)
    evidence["audio_refine_score"] = round(refinement.score, 4)
    evidence["audio_refine_applied"] = 1
    refined.evidence = evidence
    return refined, refinement
