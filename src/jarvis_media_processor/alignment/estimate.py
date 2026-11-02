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
from jarvis_media_processor.alignment.piecewise import (
    build_piecewise_alignment,
    select_monotonic_correspondences,
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


def _unique_donor_times(pairs: list[TimePair]) -> set[float]:
    return {round(p.donor_time, 3) for p in pairs}


def _injective_inliers(inliers: list[TimePair]) -> list[TimePair]:
    """Keep a 1:1 subset (unique base and donor times), preferring higher weight."""
    ordered = sorted(inliers, key=lambda p: p.weight, reverse=True)
    used_base: set[float] = set()
    used_donor: set[float] = set()
    selected: list[TimePair] = []
    for pair in ordered:
        b = round(pair.base_time, 3)
        d = round(pair.donor_time, 3)
        if b in used_base or d in used_donor:
            continue
        used_base.add(b)
        used_donor.add(d)
        selected.append(pair)
    return selected


def _confidence(
    inliers: list[TimePair],
    base_anchor_count: int,
    rmse: float,
    tolerance: float,
) -> float:
    if base_anchor_count <= 0:
        return 0.0
    injective = _injective_inliers(inliers)
    coverage = len(_unique_base_times(injective)) / base_anchor_count
    rmse_score = max(0.0, 1.0 - (rmse / max(tolerance, 1e-6)))
    return float(max(0.0, min(1.0, 0.7 * coverage + 0.3 * rmse_score)))


def _accepts(inliers: list[TimePair], base_anchor_count: int, *, min_coverage: float) -> bool:
    if base_anchor_count <= 0:
        return False
    injective = _injective_inliers(inliers)
    covered = len(_unique_base_times(injective))
    # Require a true 1:1 matching set (no reused donor anchors).
    if len(_unique_donor_times(injective)) != covered:
        return False
    return covered >= max(1, int(np.ceil(min_coverage * base_anchor_count)))


def _monotonic_agrees(
    pairs: list[TimePair],
    *,
    speed: float,
    offset: float,
    tolerance: float,
    min_coverage: float,
) -> bool:
    """Require the monotonic correspondence chain to agree with a global model."""
    chain = select_monotonic_correspondences(pairs)
    if not chain:
        return False
    ok = sum(
        1
        for p in chain
        if abs(p.donor_time - (speed * p.base_time + offset)) <= tolerance
    )
    return ok / len(chain) >= min_coverage


def _full_span_segment(
    pairs: list[TimePair],
    *,
    speed: float,
    offset: float,
    confidence: float,
    base_duration: float | None,
    donor_duration: float | None,
) -> AlignmentSegment:
    """Build one segment covering the valid overlap of base and donor under the model.

    Model: ``donor ≈ speed * base + offset``. Negative offsets leave leading base
    silence; positive offsets leave trailing base silence — without time-stretching
    the mapped region.
    """
    model_speed = speed if speed > 0 else 1.0

    if base_duration is not None:
        base_lo = 0.0
        base_hi = float(base_duration)
        # Restrict to donor times inside [0, donor_duration] when known.
        donor_lo = 0.0
        donor_hi = float(donor_duration) if donor_duration is not None else float("inf")
        base_start = max(base_lo, (donor_lo - offset) / model_speed)
        base_end = min(base_hi, (donor_hi - offset) / model_speed)
        if base_end < base_start:
            base_start = base_lo
            base_end = base_lo
        donor_start = float(model_speed * base_start + offset)
        donor_end = float(model_speed * base_end + offset)
        if donor_duration is not None:
            donor_start = max(0.0, min(donor_start, float(donor_duration)))
            donor_end = max(0.0, min(donor_end, float(donor_duration)))
    elif pairs:
        base_start = min(p.base_time for p in pairs)
        base_end = max(p.base_time for p in pairs)
        donor_start = float(model_speed * base_start + offset)
        donor_end = float(model_speed * base_end + offset)
    else:
        base_start, base_end = 0.0, 0.0
        donor_start = max(0.0, float(offset))
        donor_end = donor_start

    if base_end <= base_start:
        base_end = base_start + 1e-3
    if donor_end <= donor_start:
        donor_end = donor_start + 1e-3

    return AlignmentSegment(
        base_start=base_start,
        base_end=base_end,
        donor_start=donor_start,
        donor_end=donor_end,
        speed=model_speed,
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
    min_inlier_ratio: float = 0.85,
    already_synced_tolerance: float = 0.08,
    speed_tolerance: float = 0.03,
    speed_min: float = 0.85,
    speed_max: float = 1.2,
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
    if (
        offset_fit
        and _accepts(
            offset_fit.inliers,
            base_anchor_count,
            min_coverage=min_inlier_ratio,
        )
        and _monotonic_agrees(
            pairs,
            speed=1.0,
            offset=offset_fit.offset,
            tolerance=inlier_tolerance,
            min_coverage=min_inlier_ratio,
        )
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

    speed_fit = fit_speed_offset(
        pairs,
        inlier_tolerance=inlier_tolerance,
        speed_min=speed_min,
        speed_max=speed_max,
    )
    if (
        speed_fit
        and _accepts(speed_fit.inliers, base_anchor_count, min_coverage=min_inlier_ratio)
        and abs(speed_fit.speed - 1.0) > speed_tolerance
        and _monotonic_agrees(
            pairs,
            speed=speed_fit.speed,
            offset=speed_fit.offset,
            tolerance=inlier_tolerance,
            min_coverage=min_inlier_ratio,
        )
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

    notes.append("No single offset/speed hypothesis met inlier thresholds; using piecewise mapper.")
    return build_piecewise_alignment(
        pairs,
        base_duration=base_duration,
        donor_duration=donor_duration,
        offset_tolerance=max(inlier_tolerance * 2, 0.2),
        notes=notes,
    )


_GLOBAL_STRATEGIES = {
    SyncStrategyLevel.ALREADY_SYNCED,
    SyncStrategyLevel.CONSTANT_OFFSET,
    SyncStrategyLevel.CONSTANT_SPEED,
}


def estimate_alignment_visual_first(
    visual_pairs: list[TimePair],
    audio_pairs: list[TimePair],
    *,
    base_duration: float | None = None,
    donor_duration: float | None = None,
    audio_inlier_tolerance: float = 0.12,
    visual_inlier_tolerance: float = 1.5,
    min_visual_pairs: int = 8,
    visual_min_inlier_ratio: float = 0.55,
    audio_downweight: float = 0.2,
    speed_min: float = 0.85,
    speed_max: float = 1.2,
    speed_tolerance: float = 0.03,
) -> AlignmentDocument:
    """Prefer a visual timing model; do not let weak audio override it.

    Policy:
    1. If enough visual pairs exist, fit offset/speed/piecewise on **visual only**
       with a looser tolerance (frame sampling).
    2. Accept global visual models immediately; also prefer visual piecewise when
       it is at least moderately confident.
    3. Otherwise fall back to fused evidence with **downweighted** audio so dense
       cross-language landmark noise cannot dominate.
    """
    from jarvis_media_processor.alignment.evidence import downweight_pairs, merge_evidence

    notes: list[str] = []
    if len(visual_pairs) >= min_visual_pairs:
        visual_doc = estimate_alignment(
            visual_pairs,
            base_duration=base_duration,
            donor_duration=donor_duration,
            inlier_tolerance=visual_inlier_tolerance,
            min_inlier_ratio=visual_min_inlier_ratio,
            speed_tolerance=speed_tolerance,
            speed_min=speed_min,
            speed_max=speed_max,
        )
        if visual_doc.strategy in _GLOBAL_STRATEGIES:
            visual_doc.notes = [
                "Visual-first: accepted global model from visual evidence only.",
                *visual_doc.notes,
            ]
            evidence = dict(visual_doc.evidence)
            evidence["visual_pairs"] = len(visual_pairs)
            evidence["audio_pairs"] = len(audio_pairs)
            evidence["fusion"] = "visual_only_global"
            visual_doc.evidence = evidence
            return visual_doc
        if (
            visual_doc.strategy == SyncStrategyLevel.PIECEWISE
            and visual_doc.confidence >= 0.5
            and visual_doc.segments
        ):
            visual_doc.notes = [
                "Visual-first: using visual piecewise map (no global model).",
                *visual_doc.notes,
            ]
            evidence = dict(visual_doc.evidence)
            evidence["visual_pairs"] = len(visual_pairs)
            evidence["audio_pairs"] = len(audio_pairs)
            evidence["fusion"] = "visual_only_piecewise"
            visual_doc.evidence = evidence
            return visual_doc
        notes.append(
            "Visual-first: visual-only fit was weak; falling back to fused evidence "
            "with downweighted audio."
        )
    else:
        notes.append(
            f"Visual-first: fewer than {min_visual_pairs} visual pairs; "
            "using audio-heavy fused escalate."
        )

    fused = merge_evidence(
        visual_pairs,
        downweight_pairs(audio_pairs, audio_downweight),
    )
    # When visuals exist, keep a tolerance that can absorb frame quantization.
    tolerance = (
        max(audio_inlier_tolerance, visual_inlier_tolerance * 0.5)
        if visual_pairs
        else audio_inlier_tolerance
    )
    doc = estimate_alignment(
        fused,
        base_duration=base_duration,
        donor_duration=donor_duration,
        inlier_tolerance=tolerance,
    )
    doc.notes = notes + doc.notes
    evidence = dict(doc.evidence)
    evidence["visual_pairs"] = len(visual_pairs)
    evidence["audio_pairs"] = len(audio_pairs)
    evidence["fusion"] = "fused_downweighted_audio"
    doc.evidence = evidence
    return doc
