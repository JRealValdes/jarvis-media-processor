"""Estimate sync hypotheses from timed evidence pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jarvis_media_processor.alignment.evidence import TimePair
from jarvis_media_processor.alignment.models import (
    AlignmentDocument,
    AlignmentSegment,
    SyncStrategyLevel,
    TimelineInterval,
)


@dataclass(frozen=True)
class FitResult:
    speed: float
    offset: float
    inliers: list[TimePair]
    rmse: float


def _pair_arrays(pairs: list[TimePair]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.array([p.base_time for p in pairs], dtype=np.float64)
    donor = np.array([p.donor_time for p in pairs], dtype=np.float64)
    weights = np.array([p.weight for p in pairs], dtype=np.float64)
    return base, donor, weights


def fit_offset(
    pairs: list[TimePair],
    *,
    inlier_tolerance: float,
) -> FitResult | None:
    """RANSAC-like search for a constant offset: donor ≈ base + offset."""
    if len(pairs) < 1:
        return None

    base, donor, _weights = _pair_arrays(pairs)
    candidate_offsets = donor - base
    best: FitResult | None = None

    for offset in candidate_offsets:
        residuals = np.abs(donor - (base + offset))
        inlier_mask = residuals <= inlier_tolerance
        inliers = [p for p, ok in zip(pairs, inlier_mask, strict=True) if ok]
        if not inliers:
            continue
        rmse = float(np.sqrt(np.mean(residuals[inlier_mask] ** 2)))
        score = (len(inliers), -rmse)
        if best is None or score > (len(best.inliers), -best.rmse):
            best = FitResult(speed=1.0, offset=float(offset), inliers=inliers, rmse=rmse)
    return best


def fit_speed_offset(
    pairs: list[TimePair],
    *,
    inlier_tolerance: float,
    speed_min: float = 0.85,
    speed_max: float = 1.2,
) -> FitResult | None:
    """Fit donor ≈ speed * base + offset with a small grid over speed."""
    if len(pairs) < 2:
        return None

    base, donor, weights = _pair_arrays(pairs)
    best: FitResult | None = None
    for speed in np.linspace(speed_min, speed_max, 36):
        # weighted mean offset for this speed
        offset = float(np.average(donor - speed * base, weights=weights))
        residuals = np.abs(donor - (speed * base + offset))
        inlier_mask = residuals <= inlier_tolerance
        inliers = [p for p, ok in zip(pairs, inlier_mask, strict=True) if ok]
        if len(inliers) < 2:
            continue
        rmse = float(np.sqrt(np.mean(residuals[inlier_mask] ** 2)))
        score = (len(inliers), -abs(speed - 1.0), -rmse)
        if best is None or score > (len(best.inliers), -abs(best.speed - 1.0), -best.rmse):
            best = FitResult(speed=float(speed), offset=offset, inliers=inliers, rmse=rmse)
    return best


def _unique_base_times(pairs: list[TimePair]) -> set[float]:
    return {round(p.base_time, 3) for p in pairs}


def _confidence(
    inliers: list[TimePair],
    base_anchor_count: int,
    rmse: float,
    tolerance: float,
) -> float:
    if base_anchor_count <= 0:
        return 0.0
    coverage = len(_unique_base_times(inliers)) / base_anchor_count
    rmse_score = max(0.0, 1.0 - (rmse / max(tolerance, 1e-6)))
    return float(max(0.0, min(1.0, 0.7 * coverage + 0.3 * rmse_score)))


def _accepts(inliers: list[TimePair], base_anchor_count: int, *, min_coverage: float) -> bool:
    if base_anchor_count <= 0:
        return False
    covered = len(_unique_base_times(inliers))
    return covered >= max(1, int(np.ceil(min_coverage * base_anchor_count)))


def _full_span_segment(
    pairs: list[TimePair],
    *,
    speed: float,
    offset: float,
    confidence: float,
    base_duration: float | None,
    donor_duration: float | None,
) -> AlignmentSegment:
    if pairs:
        base_start = min(p.base_time for p in pairs)
        base_end = max(p.base_time for p in pairs)
        donor_start = min(p.donor_time for p in pairs)
        donor_end = max(p.donor_time for p in pairs)
    else:
        base_start, base_end = 0.0, base_duration or 0.0
        donor_start, donor_end = 0.0, donor_duration or 0.0

    if base_duration is not None:
        base_start, base_end = 0.0, base_duration
    if donor_duration is not None and speed == 1.0:
        # For pure offset, donor span follows duration when known.
        donor_start = max(0.0, offset) if offset > 0 else 0.0
        if donor_duration is not None:
            donor_end = donor_duration

    # Map endpoints with the model when durations known.
    if base_duration is not None:
        base_start, base_end = 0.0, float(base_duration)
        donor_start = float(speed * base_start + offset)
        donor_end = float(speed * base_end + offset)
        if donor_duration is not None:
            donor_start = max(0.0, min(donor_start, donor_duration))
            donor_end = max(0.0, min(donor_end, donor_duration))

    if base_end <= base_start:
        base_end = base_start + 1e-3
    if donor_end <= donor_start:
        donor_end = donor_start + 1e-3

    return AlignmentSegment(
        base_start=base_start,
        base_end=base_end,
        donor_start=donor_start,
        donor_end=donor_end,
        speed=speed,
        offset_seconds=offset,
        confidence=confidence,
        reliable=confidence >= 0.5,
    )


def estimate_alignment(
    pairs: list[TimePair],
    *,
    base_duration: float | None = None,
    donor_duration: float | None = None,
    inlier_tolerance: float = 0.12,
    min_inlier_ratio: float = 0.6,
    already_synced_tolerance: float = 0.08,
    speed_tolerance: float = 0.03,
) -> AlignmentDocument:
    """Escalate sync hypotheses from evidence pairs.

    Order: already synced → constant offset → constant speed → piecewise stub.
    """
    total = len(pairs)
    base_anchor_count = len(_unique_base_times(pairs))
    notes: list[str] = []
    if total == 0:
        return AlignmentDocument(
            strategy=SyncStrategyLevel.FAILED,
            confidence=0.0,
            notes=["No timing evidence pairs available."],
            uncertain_intervals=[
                TimelineInterval(
                    start=0.0,
                    end=base_duration or 0.0,
                    timeline="base",
                    reason="no_evidence",
                )
            ],
            evidence={"pair_count": 0},
        )

    offset_fit = fit_offset(pairs, inlier_tolerance=inlier_tolerance)
    if offset_fit and _accepts(
        offset_fit.inliers, base_anchor_count, min_coverage=min_inlier_ratio
    ):
        conf = _confidence(
            offset_fit.inliers,
            base_anchor_count,
            offset_fit.rmse,
            inlier_tolerance,
        )
        if abs(offset_fit.offset) <= already_synced_tolerance:
            strategy = SyncStrategyLevel.ALREADY_SYNCED
            notes.append("Accepted near-zero constant offset.")
        else:
            strategy = SyncStrategyLevel.CONSTANT_OFFSET
            notes.append(f"Accepted constant offset={offset_fit.offset:.3f}s.")
        segment = _full_span_segment(
            offset_fit.inliers,
            speed=1.0,
            offset=offset_fit.offset,
            confidence=conf,
            base_duration=base_duration,
            donor_duration=donor_duration,
        )
        return AlignmentDocument(
            strategy=strategy,
            confidence=conf,
            global_offset_seconds=offset_fit.offset,
            global_speed=1.0,
            segments=[segment],
            notes=notes,
            evidence={
                "pair_count": total,
                "base_anchors": base_anchor_count,
                "inliers": len(offset_fit.inliers),
                "rmse": round(offset_fit.rmse, 4),
            },
        )

    speed_fit = fit_speed_offset(pairs, inlier_tolerance=inlier_tolerance)
    if (
        speed_fit
        and _accepts(speed_fit.inliers, base_anchor_count, min_coverage=min_inlier_ratio)
        and abs(speed_fit.speed - 1.0) > speed_tolerance
    ):
        conf = _confidence(
            speed_fit.inliers,
            base_anchor_count,
            speed_fit.rmse,
            inlier_tolerance,
        )
        segment = _full_span_segment(
            speed_fit.inliers,
            speed=speed_fit.speed,
            offset=speed_fit.offset,
            confidence=conf,
            base_duration=base_duration,
            donor_duration=donor_duration,
        )
        return AlignmentDocument(
            strategy=SyncStrategyLevel.CONSTANT_SPEED,
            confidence=conf,
            global_offset_seconds=speed_fit.offset,
            global_speed=speed_fit.speed,
            segments=[segment],
            notes=[f"Accepted speed={speed_fit.speed:.4f}, offset={speed_fit.offset:.3f}s."],
            evidence={
                "pair_count": total,
                "base_anchors": base_anchor_count,
                "inliers": len(speed_fit.inliers),
                "rmse": round(speed_fit.rmse, 4),
            },
        )

    # Piecewise stub: cluster by local offset without inventing coverage.
    notes.append(
        "No single offset/speed hypothesis met inlier thresholds; "
        "emitting piecewise stub from inlier clusters (refine in #7)."
    )
    offsets = [(p, p.donor_time - p.base_time) for p in pairs]
    offsets.sort(key=lambda item: item[0].base_time)
    clusters: list[list[TimePair]] = []
    current: list[TimePair] = []
    current_offset: float | None = None
    for pair, offset in offsets:
        if current_offset is None or abs(offset - current_offset) <= inlier_tolerance * 2:
            current.append(pair)
            current_offset = (
                offset
                if current_offset is None
                else float(np.mean([c.donor_time - c.base_time for c in current]))
            )
        else:
            if current:
                clusters.append(current)
            current = [pair]
            current_offset = offset
    if current:
        clusters.append(current)

    segments: list[AlignmentSegment] = []
    for cluster in clusters:
        if len(cluster) < 1:
            continue
        off = float(np.mean([p.donor_time - p.base_time for p in cluster]))
        conf = min(1.0, len(cluster) / max(total, 1))
        segments.append(
            AlignmentSegment(
                base_start=min(p.base_time for p in cluster),
                base_end=max(p.base_time for p in cluster) + 1e-3,
                donor_start=min(p.donor_time for p in cluster),
                donor_end=max(p.donor_time for p in cluster) + 1e-3,
                speed=1.0,
                offset_seconds=off,
                confidence=conf,
                reliable=len(cluster) >= 2,
            )
        )

    uncertain: list[TimelineInterval] = []
    if base_duration is not None and segments:
        # Flag gaps between segment endpoints on the base timeline.
        ordered = sorted(segments, key=lambda s: s.base_start)
        if ordered[0].base_start > 0.25:
            uncertain.append(
                TimelineInterval(
                    start=0.0,
                    end=ordered[0].base_start,
                    timeline="base",
                    reason="gap_before_first_segment",
                )
            )
        for left, right in zip(ordered, ordered[1:], strict=False):
            if right.base_start - left.base_end > 0.25:
                uncertain.append(
                    TimelineInterval(
                        start=left.base_end,
                        end=right.base_start,
                        timeline="base",
                        reason="gap_between_segments",
                    )
                )
        if base_duration - ordered[-1].base_end > 0.25:
            uncertain.append(
                TimelineInterval(
                    start=ordered[-1].base_end,
                    end=base_duration,
                    timeline="base",
                    reason="gap_after_last_segment",
                )
            )
    elif not segments:
        uncertain.append(
            TimelineInterval(
                start=0.0,
                end=base_duration or 0.0,
                timeline="base",
                reason="unaligned",
            )
        )

    conf = float(np.mean([s.confidence for s in segments])) if segments else 0.0
    return AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=conf,
        segments=segments,
        uncertain_intervals=uncertain,
        notes=notes,
        evidence={"pair_count": total, "segment_count": len(segments)},
    )
