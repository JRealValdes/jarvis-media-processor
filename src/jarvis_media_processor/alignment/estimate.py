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


def _weighted_affine_ls(pairs: list[TimePair]) -> tuple[float, float] | None:
    """Closed-form weighted LS for ``donor ≈ speed * base + offset``."""
    if len(pairs) < 2:
        return None
    base, donor, weights = _pair_arrays(pairs)
    if float(np.max(base) - np.min(base)) < 1e-3:
        return None
    w = np.maximum(weights, 1e-9)
    sw = float(np.sum(w))
    swb = float(np.sum(w * base))
    swd = float(np.sum(w * donor))
    swbb = float(np.sum(w * base * base))
    swbd = float(np.sum(w * base * donor))
    det = sw * swbb - swb * swb
    if abs(det) < 1e-12:
        return None
    speed = (sw * swbd - swb * swd) / det
    offset = (swd - speed * swb) / sw
    return float(speed), float(offset)


def _inliers_for_affine(
    pairs: list[TimePair],
    *,
    speed: float,
    offset: float,
    inlier_tolerance: float,
) -> tuple[list[TimePair], float]:
    """Return (inliers, rmse) for an affine model."""
    if not pairs:
        return [], float("inf")
    base, donor, _weights = _pair_arrays(pairs)
    residuals = np.abs(donor - (speed * base + offset))
    inlier_mask = residuals <= inlier_tolerance
    inliers = [p for p, ok in zip(pairs, inlier_mask, strict=True) if ok]
    if not inliers:
        return [], float("inf")
    rmse = float(np.sqrt(np.mean(residuals[inlier_mask] ** 2)))
    return inliers, rmse


def fit_speed_offset(
    pairs: list[TimePair],
    *,
    inlier_tolerance: float,
    speed_min: float = 0.85,
    speed_max: float = 1.2,
) -> FitResult | None:
    """Fit ``donor ≈ speed * base + offset``.

    A coarse speed grid discovers a robust inlier set (title-agnostic; not tuned
    to PAL 0.96 or any single rip). Each top candidate is then refined with
    weighted least squares so long keep-chunks do not quantize to a grid point
    and accumulate seconds of drift.
    """
    if len(pairs) < 2:
        return None

    base, donor, weights = _pair_arrays(pairs)
    # ~0.005 step over the default [0.85, 1.2] band — only seeds inliers.
    grid_count = 71
    seeds: list[tuple[int, float, float, float, list[TimePair]]] = []
    for speed in np.linspace(speed_min, speed_max, grid_count):
        offset = float(np.average(donor - speed * base, weights=weights))
        inliers, rmse = _inliers_for_affine(
            pairs,
            speed=float(speed),
            offset=offset,
            inlier_tolerance=inlier_tolerance,
        )
        if len(inliers) < 2:
            continue
        # Coverage and RMSE only — do not prefer speed≈1.0 (that bias hid true
        # mild retimes such as 23.976/25 ≈ 0.959 behind the 0.96 grid point).
        seeds.append((len(inliers), -rmse, float(speed), offset, inliers))

    if not seeds:
        return None

    seeds.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best: FitResult | None = None
    best_score: tuple[int, float] | None = None
    for _n, _neg_rmse, _seed_speed, _seed_offset, seed_inliers in seeds[:5]:
        refined = seed_inliers
        speed = float(_seed_speed)
        offset = float(_seed_offset)
        for _ in range(2):
            ls = _weighted_affine_ls(refined)
            if ls is None:
                break
            speed, offset = ls
            speed = float(min(max(speed, speed_min), speed_max))
            refined, rmse = _inliers_for_affine(
                pairs,
                speed=speed,
                offset=offset,
                inlier_tolerance=inlier_tolerance,
            )
            if len(refined) < 2:
                break
        if len(refined) < 2:
            continue
        inliers, rmse = _inliers_for_affine(
            pairs,
            speed=speed,
            offset=offset,
            inlier_tolerance=inlier_tolerance,
        )
        if len(inliers) < 2:
            continue
        score = (len(inliers), -rmse)
        if best is None or best_score is None or score > best_score:
            best = FitResult(
                speed=float(speed),
                offset=float(offset),
                inliers=inliers,
                rmse=float(rmse),
            )
            best_score = score
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
    speed_fit = fit_speed_offset(
        pairs,
        inlier_tolerance=inlier_tolerance,
        speed_min=speed_min,
        speed_max=speed_max,
    )
    offset_ok = bool(
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
    )
    speed_ok = bool(
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
    )
    # Prefer a non-1.0 speed model when it clearly explains more pairs. A short
    # post-ad window can look like donor≈base-19 at speed 1.0 while the true
    # cross-encode map is speed≈0.96 with a different offset; accepting offset
    # first hid that.
    prefer_speed = False
    if offset_ok and speed_ok:
        assert offset_fit is not None and speed_fit is not None
        off_n = len(_injective_inliers(offset_fit.inliers))
        sp_n = len(_injective_inliers(speed_fit.inliers))
        prefer_speed = sp_n >= off_n + 2 or (
            sp_n >= off_n and speed_fit.rmse <= offset_fit.rmse * 0.85
        )
    elif speed_ok and not offset_ok:
        prefer_speed = True

    if prefer_speed and speed_fit is not None:
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

    if offset_ok and offset_fit is not None:
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

# Discount applied when a failed keep-chunk borrows a sibling chunk's model.
_SIBLING_FALLBACK_CONF_SCALE = 0.85
# If affine residual trends by more than this across a keep-chunk, split once.
_RESIDUAL_DRIFT_SPLIT_SECONDS = 0.75
_MIN_DRIFT_SPLIT_DONOR_SPAN = 180.0
_MIN_DRIFT_SPLIT_PAIRS = 12
_MAX_DRIFT_SPLIT_DEPTH = 2
# Bridge a surplus-donor gap when neighbor affines agree on the mapped base hole.
_BRIDGE_EDGE_MATCH_SECONDS = 2.0
_BRIDGE_MAX_MODEL_DISAGREE_SECONDS = 2.0
_BRIDGE_MIN_DONOR_SECONDS = 1.0
_BRIDGE_CONF_SCALE = 0.75
_BRIDGE_OFFSET_JUMP_NOTE_SECONDS = 2.0
# Donor span through a missing-base hole must match speed*base_span (10s≈10.05s OK; 20s not).
_MISSING_DURATION_REL_TOL = 0.05
_MISSING_DURATION_ABS_TOL_SECONDS = 0.75
_MISSING_MIN_BASE_SECONDS = 1.0
# Close base seams where donor is already continuous by slowing both neighbors
# with one common factor f=(B_L+B_R)/(B_L+B_R+G). Cap slowdown at 2% (f≥0.98):
# micro fit seams pass; a 1s hole between two 10s keeps is rejected.
_SEAM_MIN_SPEED_FACTOR = 0.98
_SEAM_DONOR_CONTINUITY_SECONDS = 1e-3
_SEAM_MIN_GAP_SECONDS = 1e-4


@dataclass(frozen=True)
class ChunkTimingHint:
    """Optional timing hint for a failed donor keep-chunk (e.g. curve-walk lock)."""

    speed: float
    offset_seconds: float
    score: float
    donor_time: float


def _map_keep_to_segment(
    d0: float,
    d1: float,
    *,
    speed: float,
    offset: float,
    confidence: float,
    base_duration: float | None,
    reliable: bool = True,
) -> AlignmentSegment | None:
    """Map a donor keep-range through ``donor ≈ speed * base + offset``."""
    model_speed = speed if speed > 0 else 1.0
    b0 = (d0 - offset) / max(model_speed, 1e-6)
    b1 = (d1 - offset) / max(model_speed, 1e-6)
    if base_duration is not None:
        b0 = min(max(0.0, b0), base_duration)
        b1 = min(max(0.0, b1), base_duration)
    if b1 <= b0 + 1e-3:
        return None
    return AlignmentSegment(
        base_start=float(b0),
        base_end=float(b1),
        donor_start=float(d0),
        donor_end=float(d1),
        speed=float(model_speed),
        offset_seconds=float(offset),
        confidence=float(confidence),
        reliable=reliable,
    )


def _best_hint_in_chunk(
    hints: list[ChunkTimingHint],
    d0: float,
    d1: float,
) -> ChunkTimingHint | None:
    """Highest-score hint whose donor time lies inside the keep-range."""
    inside = [
        hint
        for hint in hints
        if d0 - 1e-6 <= hint.donor_time <= d1 + 1e-6
    ]
    if not inside:
        return None
    return max(inside, key=lambda hint: hint.score)


def _nearest_sibling_segment(
    d0: float,
    d1: float,
    fitted: list[AlignmentSegment],
) -> AlignmentSegment | None:
    """Pick the fitted segment whose donor midpoint is closest to this keep."""
    if not fitted:
        return None
    mid = 0.5 * (d0 + d1)
    return min(
        fitted,
        key=lambda seg: abs(0.5 * (seg.donor_start + seg.donor_end) - mid),
    )


def _base_of_donor(donor_t: float, *, speed: float, offset: float) -> float:
    model_speed = speed if speed > 0 else 1.0
    return (donor_t - offset) / max(model_speed, 1e-6)


def _donor_of_base(base_t: float, *, speed: float, offset: float) -> float:
    model_speed = speed if speed > 0 else 1.0
    return model_speed * base_t + offset


def _duration_compatible(
    base_span: float,
    donor_span: float,
    *,
    speed: float,
) -> bool:
    """True when donor length matches ``speed * base_span`` within abs/rel tolerance."""
    if base_span <= 0 or donor_span <= 0:
        return False
    model_speed = speed if speed > 0 else 1.0
    expected = model_speed * base_span
    tol = max(_MISSING_DURATION_ABS_TOL_SECONDS, _MISSING_DURATION_REL_TOL * expected)
    return abs(donor_span - expected) <= tol + 1e-9


def bridge_surplus_donor_gaps(
    segments: list[AlignmentSegment],
    surplus: list[tuple[float, float]],
    *,
    base_duration: float | None = None,
    edge_match_seconds: float = _BRIDGE_EDGE_MATCH_SECONDS,
    max_model_disagree_seconds: float = _BRIDGE_MAX_MODEL_DISAGREE_SECONDS,
    min_bridge_seconds: float = _BRIDGE_MIN_DONOR_SECONDS,
    conf_scale: float = _BRIDGE_CONF_SCALE,
) -> tuple[list[AlignmentSegment], list[tuple[float, float]], list[str]]:
    """Pass-2 consistency law: retract false surplus when neighbors agree.

    Curve-walk may *propose* surplus. After piecewise fit, if an excluded donor
    span maps onto the base hole under agreeing neighbor affines, reclassify it
    as placeable content (bridge). True mid-roll ads leave disagreeing affines
    and stay excluded.
    """
    if not segments or not surplus:
        return [], [], []

    ordered = sorted(segments, key=lambda s: s.donor_start)
    bridges: list[AlignmentSegment] = []
    recovered: list[tuple[float, float]] = []
    notes: list[str] = []

    for d0, d1 in sorted(surplus):
        if d1 - d0 < min_bridge_seconds - 1e-9:
            continue
        left_candidates = [
            seg
            for seg in ordered
            if abs(seg.donor_end - d0) <= edge_match_seconds
        ]
        right_candidates = [
            seg
            for seg in ordered
            if abs(seg.donor_start - d1) <= edge_match_seconds
        ]
        if not left_candidates or not right_candidates:
            continue
        left = min(left_candidates, key=lambda seg: abs(seg.donor_end - d0))
        right = min(right_candidates, key=lambda seg: abs(seg.donor_start - d1))
        if left is right:
            continue

        left_b0 = _base_of_donor(d0, speed=left.speed, offset=left.offset_seconds)
        left_b1 = _base_of_donor(d1, speed=left.speed, offset=left.offset_seconds)
        right_b0 = _base_of_donor(d0, speed=right.speed, offset=right.offset_seconds)
        right_b1 = _base_of_donor(d1, speed=right.speed, offset=right.offset_seconds)
        if abs(left_b0 - right_b0) > max_model_disagree_seconds:
            continue
        if abs(left_b1 - right_b1) > max_model_disagree_seconds:
            continue

        # Neighbors agree on the hole → stitch with averaged endpoints / model.
        b0 = 0.5 * (left_b0 + right_b0)
        b1 = 0.5 * (left_b1 + right_b1)
        if b1 < b0:
            b0, b1 = b1, b0
        if base_duration is not None:
            b0 = min(max(0.0, b0), base_duration)
            b1 = min(max(0.0, b1), base_duration)
        if b1 - b0 < min_bridge_seconds - 1e-9:
            continue

        # Require the bridge to land in/near the base gap between neighbors.
        gap_lo = min(left.base_end, right.base_start)
        gap_hi = max(left.base_end, right.base_start)
        if gap_hi - gap_lo < 0.05:
            # Contiguous on base already — nothing to fill.
            continue
        overlap = min(b1, gap_hi) - max(b0, gap_lo)
        if overlap < min_bridge_seconds - 1e-9:
            continue

        speed = 0.5 * (left.speed + right.speed)
        offset = 0.5 * (left.offset_seconds + right.offset_seconds)
        # Mean (not min) so a slightly weaker neighbor does not push a solid
        # bridge under the reliable threshold after conf_scale.
        conf = 0.5 * (float(left.confidence) + float(right.confidence)) * conf_scale
        segment = AlignmentSegment(
            base_start=float(b0),
            base_end=float(b1),
            donor_start=float(d0),
            donor_end=float(d1),
            speed=float(speed),
            offset_seconds=float(offset),
            confidence=float(max(0.0, min(1.0, conf))),
            reliable=conf >= 0.5,
        )
        bridges.append(segment)
        recovered.append((float(d0), float(d1)))
        offset_jump = abs(left.offset_seconds - right.offset_seconds)
        note = (
            f"Bridged surplus donor {d0:.1f}-{d1:.1f}s → base {b0:.1f}-{b1:.1f}s "
            f"(neighbors agree within {max_model_disagree_seconds:.1f}s; "
            f"conf={conf:.2f})."
        )
        if offset_jump >= _BRIDGE_OFFSET_JUMP_NOTE_SECONDS:
            note += (
                f" Neighbor offsets differ by {offset_jump:.2f}s "
                "(drift-split / dual-offset clue; bridge still used because "
                "mapped endpoints agree)."
            )
        notes.append(note)

    return bridges, recovered, notes


def bridge_missing_base_gaps(
    segments: list[AlignmentSegment],
    missing: list[tuple[float, float]],
    *,
    donor_duration: float | None = None,
    edge_match_seconds: float = _BRIDGE_EDGE_MATCH_SECONDS,
    max_model_disagree_seconds: float = _BRIDGE_MAX_MODEL_DISAGREE_SECONDS,
    min_bridge_seconds: float = _MISSING_MIN_BASE_SECONDS,
    conf_scale: float = _BRIDGE_CONF_SCALE,
) -> tuple[list[AlignmentSegment], list[tuple[float, float]], list[str]]:
    """Pass-2: retract false missing_base when neighbors can place continuous donor.

    If a base hole sits between two keeps whose affines agree on the mapped donor
    span **and** that donor span has a compatible duration (``≈ speed·hole``),
    stitch Spanish through the hole instead of EN-fill.
    """
    if not segments or not missing:
        return [], [], []

    ordered = sorted(segments, key=lambda s: s.base_start)
    bridges: list[AlignmentSegment] = []
    recovered: list[tuple[float, float]] = []
    notes: list[str] = []

    for b0, b1 in sorted(missing):
        base_span = b1 - b0
        if base_span < min_bridge_seconds - 1e-9:
            continue
        left_candidates = [
            seg
            for seg in ordered
            if abs(seg.base_end - b0) <= edge_match_seconds
        ]
        right_candidates = [
            seg
            for seg in ordered
            if abs(seg.base_start - b1) <= edge_match_seconds
        ]
        # Also accept hole strictly between neighbors (small edge slack).
        if not left_candidates:
            left_candidates = [
                seg for seg in ordered if seg.base_end <= b0 + edge_match_seconds
            ]
            if left_candidates:
                left_candidates = [
                    max(left_candidates, key=lambda seg: seg.base_end)
                ]
        if not right_candidates:
            right_candidates = [
                seg for seg in ordered if seg.base_start >= b1 - edge_match_seconds
            ]
            if right_candidates:
                right_candidates = [
                    min(right_candidates, key=lambda seg: seg.base_start)
                ]
        if not left_candidates or not right_candidates:
            continue
        left = min(left_candidates, key=lambda seg: abs(seg.base_end - b0))
        right = min(right_candidates, key=lambda seg: abs(seg.base_start - b1))
        if left is right:
            continue

        left_d0 = _donor_of_base(b0, speed=left.speed, offset=left.offset_seconds)
        left_d1 = _donor_of_base(b1, speed=left.speed, offset=left.offset_seconds)
        right_d0 = _donor_of_base(b0, speed=right.speed, offset=right.offset_seconds)
        right_d1 = _donor_of_base(b1, speed=right.speed, offset=right.offset_seconds)
        if abs(left_d0 - right_d0) > max_model_disagree_seconds:
            continue
        if abs(left_d1 - right_d1) > max_model_disagree_seconds:
            continue

        d0 = 0.5 * (left_d0 + right_d0)
        d1 = 0.5 * (left_d1 + right_d1)
        if d1 < d0:
            d0, d1 = d1, d0
        if donor_duration is not None:
            d0 = min(max(0.0, d0), donor_duration)
            d1 = min(max(0.0, d1), donor_duration)
        donor_span = d1 - d0
        if donor_span < min_bridge_seconds - 1e-9:
            continue

        speed = 0.5 * (left.speed + right.speed)
        if not _duration_compatible(base_span, donor_span, speed=speed):
            notes.append(
                f"Skipped missing-base bridge {b0:.1f}-{b1:.1f}s: donor span "
                f"{donor_span:.2f}s incompatible with base {base_span:.2f}s "
                f"at speed={speed:.4f}."
            )
            continue

        # Prefer stitching the actual hole (not an oversized map).
        conf = 0.5 * (float(left.confidence) + float(right.confidence)) * conf_scale
        offset = 0.5 * (left.offset_seconds + right.offset_seconds)
        segment = AlignmentSegment(
            base_start=float(b0),
            base_end=float(b1),
            donor_start=float(d0),
            donor_end=float(d1),
            speed=float(speed),
            offset_seconds=float(offset),
            confidence=float(max(0.0, min(1.0, conf))),
            reliable=conf >= 0.5,
        )
        bridges.append(segment)
        recovered.append((float(b0), float(b1)))
        notes.append(
            f"Bridged missing-base {b0:.1f}-{b1:.1f}s → donor {d0:.1f}-{d1:.1f}s "
            f"(neighbors agree; duration ok; conf={conf:.2f})."
        )

    return bridges, recovered, notes


def close_donor_continuous_seams(
    segments: list[AlignmentSegment],
    *,
    min_speed_factor: float = _SEAM_MIN_SPEED_FACTOR,
    donor_continuity_seconds: float = _SEAM_DONOR_CONTINUITY_SECONDS,
    min_gap_seconds: float = _SEAM_MIN_GAP_SECONDS,
) -> tuple[list[AlignmentSegment], list[str]]:
    """Close base gaps between keeps when donor already abuts (no free donor).

    Independent affine fits often leave a small base discontinuity at a donor
    chunk boundary while ``left.donor_end == right.donor_start``. That is a seam,
    not missing content: stretch both neighbors by a **common** speed factor

        f = (B_L + B_R) / (B_L + B_R + G)

    so each absorbs gap proportional to its base duration (longer keep stretches
    more in absolute time; the join lands nearer the shorter neighbor). Reject
    when ``f < min_speed_factor`` (default 0.98 → at most 2% slowdown).
    """
    if len(segments) < 2:
        return list(segments), []

    ordered = sorted(segments, key=lambda s: (s.base_start, s.donor_start))
    notes: list[str] = []
    closed = 0

    for i in range(len(ordered) - 1):
        left = ordered[i]
        right = ordered[i + 1]
        gap = float(right.base_start) - float(left.base_end)
        if gap < min_gap_seconds:
            continue
        donor_gap = float(right.donor_start) - float(left.donor_end)
        if abs(donor_gap) > donor_continuity_seconds:
            continue

        b_left = float(left.base_end) - float(left.base_start)
        b_right = float(right.base_end) - float(right.base_start)
        if b_left <= 1e-9 or b_right <= 1e-9:
            continue
        denom = b_left + b_right + gap
        factor = (b_left + b_right) / denom
        if factor < min_speed_factor - 1e-15:
            notes.append(
                f"Skipped seam close base {left.base_end:.3f}-{right.base_start:.3f}s "
                f"(gap={gap:.3f}s): common speed factor {factor:.5f} < "
                f"min {min_speed_factor:.5f}."
            )
            continue

        meet = float(left.base_start) + b_left / factor
        new_left_speed = float(left.speed) * factor
        new_right_speed = float(right.speed) * factor
        new_left = AlignmentSegment(
            base_start=float(left.base_start),
            base_end=float(meet),
            donor_start=float(left.donor_start),
            donor_end=float(left.donor_end),
            speed=float(new_left_speed),
            offset_seconds=float(left.donor_start) - new_left_speed * float(left.base_start),
            confidence=float(left.confidence),
            reliable=left.reliable,
        )
        new_right = AlignmentSegment(
            base_start=float(meet),
            base_end=float(right.base_end),
            donor_start=float(right.donor_start),
            donor_end=float(right.donor_end),
            speed=float(new_right_speed),
            offset_seconds=float(right.donor_start) - new_right_speed * float(meet),
            confidence=float(right.confidence),
            reliable=right.reliable,
        )
        ordered[i] = new_left
        ordered[i + 1] = new_right
        closed += 1
        left_share = b_left * gap / (b_left + b_right)
        right_share = gap - left_share
        notes.append(
            f"Closed donor-continuous seam base {left.base_end:.3f}-{right.base_start:.3f}s "
            f"(gap={gap:.3f}s → meet={meet:.3f}s; speed×{factor:.5f}; "
            f"absorbed L={left_share:.3f}s R={right_share:.3f}s)."
        )

    if closed:
        notes.append(
            f"Closed {closed} donor-continuous base seam(s) "
            f"(min speed factor {min_speed_factor:.3f})."
        )
    return ordered, notes


def subtract_base_intervals(
    intervals: list[TimelineInterval],
    covered: list[tuple[float, float]],
    *,
    min_duration: float = 0.05,
) -> list[TimelineInterval]:
    """Remove ``covered`` base spans from missing-base intervals (no EN overwrite)."""
    if not intervals or not covered:
        return list(intervals)
    covered_merged = sorted(
        ((float(a), float(b)) for a, b in covered if b > a),
        key=lambda item: item[0],
    )
    out: list[TimelineInterval] = []
    for item in intervals:
        if item.timeline != "base":
            out.append(item)
            continue
        remaining = [(float(item.start), float(item.end))]
        for c0, c1 in covered_merged:
            next_remaining: list[tuple[float, float]] = []
            for r0, r1 in remaining:
                if c1 <= r0 or c0 >= r1:
                    next_remaining.append((r0, r1))
                    continue
                if c0 > r0:
                    next_remaining.append((r0, c0))
                if c1 < r1:
                    next_remaining.append((c1, r1))
            remaining = next_remaining
        for r0, r1 in remaining:
            if r1 - r0 < min_duration - 1e-9:
                continue
            out.append(
                TimelineInterval(
                    start=r0,
                    end=r1,
                    timeline="base",
                    reason=item.reason,
                )
            )
    return out


def _pair_residual_drift_seconds(
    pairs: list[TimePair],
    *,
    speed: float,
    offset: float,
) -> float:
    """Signed donor-residual change across the pair base span (linear trend).

    Positive means observed donor times run increasingly *later* than the affine
    model toward the end of the span (placement would play increasingly early
    relative to matches).
    """
    use = [p for p in pairs if not str(p.source).startswith("video_activity")]
    if len(use) < 8:
        use = list(pairs)
    if len(use) < 8:
        return 0.0
    base = np.asarray([p.base_time for p in use], dtype=np.float64)
    resid = np.asarray(
        [p.donor_time - (speed * p.base_time + offset) for p in use],
        dtype=np.float64,
    )
    order = np.argsort(base)
    base = base[order]
    resid = resid[order]
    span = float(base[-1] - base[0])
    if span < 1.0:
        return 0.0
    slope = float(np.polyfit(base, resid, 1)[0])
    return float(slope * span)


def _estimate_keep_chunk_document(
    chunk_pairs: list[TimePair],
    *,
    base_duration: float | None,
    donor_duration: float | None,
    inlier_tolerance: float,
    min_inlier_ratio: float,
    min_chunk_pairs: int,
    speed_min: float,
    speed_max: float,
    speed_tolerance: float,
) -> tuple[AlignmentDocument | None, list[str]]:
    """Fit one keep-chunk; may note pHash-only recovery."""
    notes: list[str] = []
    if len(chunk_pairs) < min_chunk_pairs:
        return None, notes
    sub = estimate_alignment(
        chunk_pairs,
        base_duration=base_duration,
        donor_duration=donor_duration,
        inlier_tolerance=inlier_tolerance,
        min_inlier_ratio=min_inlier_ratio,
        speed_tolerance=speed_tolerance,
        speed_min=speed_min,
        speed_max=speed_max,
    )
    if sub.strategy not in _GLOBAL_STRATEGIES:
        # Activity pairs from a bad mid-slice crosscorr can poison an otherwise
        # clean pHash cluster — retry without activity evidence.
        phash_only = [
            p
            for p in chunk_pairs
            if not str(p.source).startswith("video_activity")
        ]
        if len(phash_only) >= min_chunk_pairs and len(phash_only) < len(chunk_pairs):
            retry = estimate_alignment(
                phash_only,
                base_duration=base_duration,
                donor_duration=donor_duration,
                inlier_tolerance=inlier_tolerance,
                min_inlier_ratio=min_inlier_ratio,
                speed_tolerance=speed_tolerance,
                speed_min=speed_min,
                speed_max=speed_max,
            )
            if retry.strategy in _GLOBAL_STRATEGIES:
                sub = retry
                notes.append("recovered via pHash-only fit")
    if sub.strategy not in _GLOBAL_STRATEGIES:
        return None, notes
    return sub, notes


def _fit_keep_ranges_with_drift_split(
    pairs: list[TimePair],
    ranges: list[tuple[float, float]],
    *,
    base_duration: float | None,
    donor_duration: float | None,
    inlier_tolerance: float,
    min_inlier_ratio: float,
    min_chunk_pairs: int,
    speed_min: float,
    speed_max: float,
    speed_tolerance: float,
    depth: int = 0,
) -> tuple[list[AlignmentSegment], list[tuple[float, float]], list[str]]:
    """Fit donor keep-ranges; split when residual drift shows a bad single affine."""
    fitted: list[AlignmentSegment] = []
    failed: list[tuple[float, float]] = []
    notes: list[str] = []
    for d0, d1 in ranges:
        chunk_pairs = [
            p for p in pairs if d0 - 1e-6 <= p.donor_time <= d1 + 1e-6
        ]
        if len(chunk_pairs) < min_chunk_pairs:
            notes.append(
                f"Chunk {d0:.1f}-{d1:.1f}s: skipped ({len(chunk_pairs)} pairs)."
            )
            continue
        sub, sub_notes = _estimate_keep_chunk_document(
            chunk_pairs,
            base_duration=base_duration,
            donor_duration=donor_duration,
            inlier_tolerance=inlier_tolerance,
            min_inlier_ratio=min_inlier_ratio,
            min_chunk_pairs=min_chunk_pairs,
            speed_min=speed_min,
            speed_max=speed_max,
            speed_tolerance=speed_tolerance,
        )
        for tag in sub_notes:
            notes.append(f"Chunk {d0:.1f}-{d1:.1f}s: {tag}.")

        def _try_split(reason: str) -> bool:
            nonlocal fitted, failed, notes
            if depth >= _MAX_DRIFT_SPLIT_DEPTH:
                return False
            if (d1 - d0) < _MIN_DRIFT_SPLIT_DONOR_SPAN:
                return False
            if len(chunk_pairs) < _MIN_DRIFT_SPLIT_PAIRS:
                return False
            mid = 0.5 * (d0 + d1)
            notes.append(
                f"Chunk {d0:.1f}-{d1:.1f}s: {reason}; splitting at donor={mid:.1f}s."
            )
            child_fitted, child_failed, child_notes = _fit_keep_ranges_with_drift_split(
                pairs,
                [(d0, mid), (mid, d1)],
                base_duration=base_duration,
                donor_duration=donor_duration,
                inlier_tolerance=inlier_tolerance,
                min_inlier_ratio=min_inlier_ratio,
                min_chunk_pairs=min_chunk_pairs,
                speed_min=speed_min,
                speed_max=speed_max,
                speed_tolerance=speed_tolerance,
                depth=depth + 1,
            )
            notes.extend(child_notes)
            if not child_fitted:
                notes.append(
                    f"Chunk {d0:.1f}-{d1:.1f}s: split unfit; not adopting children."
                )
                return False
            fitted.extend(child_fitted)
            failed.extend(child_failed)
            return True

        if sub is None:
            if _try_split("no global fit"):
                continue
            notes.append(f"Chunk {d0:.1f}-{d1:.1f}s: no global fit.")
            failed.append((d0, d1))
            continue
        speed = float(sub.global_speed if sub.global_speed is not None else 1.0)
        offset = float(
            sub.global_offset_seconds if sub.global_offset_seconds is not None else 0.0
        )
        drift = _pair_residual_drift_seconds(
            chunk_pairs, speed=speed, offset=offset
        )
        if abs(drift) >= _RESIDUAL_DRIFT_SPLIT_SECONDS:
            if _try_split(f"residual drift {drift:+.2f}s"):
                continue
            # At max split depth a trending affine is worse than sibling recovery.
            if abs(drift) >= 2.0:
                notes.append(
                    f"Chunk {d0:.1f}-{d1:.1f}s: excessive residual drift "
                    f"{drift:+.2f}s; deferring to sibling recovery."
                )
                failed.append((d0, d1))
                continue
        segment = _map_keep_to_segment(
            d0,
            d1,
            speed=speed,
            offset=offset,
            confidence=float(sub.confidence),
            base_duration=base_duration,
            reliable=True,
        )
        if segment is None:
            notes.append(f"Chunk {d0:.1f}-{d1:.1f}s: empty base span after map.")
            failed.append((d0, d1))
            continue
        fitted.append(segment)
        notes.append(
            f"Chunk {d0:.1f}-{d1:.1f}s: speed={speed:.6f} offset={offset:.3f}s "
            f"(conf={sub.confidence:.2f}, drift={drift:+.2f}s)."
        )
    return fitted, failed, notes


def estimate_alignment_per_donor_chunk(
    pairs: list[TimePair],
    donor_keep: list[tuple[float, float]],
    *,
    base_duration: float | None = None,
    donor_duration: float | None = None,
    inlier_tolerance: float = 1.5,
    min_inlier_ratio: float = 0.55,
    min_chunk_pairs: int = 6,
    speed_min: float = 0.85,
    speed_max: float = 1.2,
    speed_tolerance: float = 0.03,
    timing_hints: list[ChunkTimingHint] | None = None,
) -> AlignmentDocument | None:
    """Fit a global model inside each donor keep-range, then stitch segments.

    Mid-roll ads create *multiple* legitimate offsets (pre-ad vs post-ad). A
    single RANSAC over all pairs then falls to a noisy piecewise map. Fitting
    per complementary donor chunk recovers clean segments instead.

    Long keep-chunks whose affine residual *trends* (speed quantization /
    mild retime error) are split and refit. Failed keeps recover via:

    1. Nearest fitted sibling chunk model (preferred — avoids a bad head lock)
    2. Else best-in-chunk timing hint (highest score among hints in range)
    3. Else skip (previous behavior)
    """
    if not donor_keep or not pairs:
        return None

    notes: list[str] = [
        "Fitted timing per donor chunk outside unrelated spans.",
    ]
    hints = list(timing_hints or [])
    fitted_segments, failed_keeps, fit_notes = _fit_keep_ranges_with_drift_split(
        pairs,
        list(donor_keep),
        base_duration=base_duration,
        donor_duration=donor_duration,
        inlier_tolerance=inlier_tolerance,
        min_inlier_ratio=min_inlier_ratio,
        min_chunk_pairs=min_chunk_pairs,
        speed_min=speed_min,
        speed_max=speed_max,
        speed_tolerance=speed_tolerance,
    )
    notes.extend(fit_notes)

    recovered_count = 0
    for d0, d1 in failed_keeps:
        sibling = _nearest_sibling_segment(d0, d1, fitted_segments)
        if sibling is not None:
            conf = float(sibling.confidence) * _SIBLING_FALLBACK_CONF_SCALE
            segment = _map_keep_to_segment(
                d0,
                d1,
                speed=sibling.speed,
                offset=sibling.offset_seconds,
                confidence=conf,
                base_duration=base_duration,
                reliable=conf >= 0.5,
            )
            if segment is not None:
                fitted_segments.append(segment)
                recovered_count += 1
                notes.append(
                    f"Chunk {d0:.1f}-{d1:.1f}s: fallback to sibling chunk model "
                    f"(speed={sibling.speed:.4f} offset={sibling.offset_seconds:.3f}s, "
                    f"conf={conf:.2f})."
                )
                continue
        hint = _best_hint_in_chunk(hints, d0, d1)
        if hint is not None:
            conf = float(max(0.0, min(1.0, hint.score)))
            segment = _map_keep_to_segment(
                d0,
                d1,
                speed=hint.speed,
                offset=hint.offset_seconds,
                confidence=conf,
                base_duration=base_duration,
                reliable=conf >= 0.5,
            )
            if segment is not None:
                fitted_segments.append(segment)
                recovered_count += 1
                notes.append(
                    f"Chunk {d0:.1f}-{d1:.1f}s: fallback to curve_walk_lock "
                    f"(speed={hint.speed:.4f} offset={hint.offset_seconds:.3f}s, "
                    f"conf={conf:.2f})."
                )
                continue
        notes.append(
            f"Chunk {d0:.1f}-{d1:.1f}s: no fallback (no sibling model or lock hint)."
        )

    if not fitted_segments:
        return None

    segments = sorted(fitted_segments, key=lambda s: (s.base_start, s.donor_start))
    confidences = [float(s.confidence) for s in segments]
    speeds = [float(s.speed) for s in segments]
    mean_conf = float(sum(confidences) / len(confidences))
    speed_span = max(speeds) - min(speeds) if speeds else 0.0
    clean_fitted = len(segments) - recovered_count
    # Only call the map "global" when every keep-range fitted cleanly with no
    # recovery; a single successful chunk beside a failed post-ad chunk is still
    # piecewise.
    if (
        clean_fitted == 1
        and recovered_count == 0
        and len(donor_keep) == 1
        and len(segments) == 1
    ):
        seg = segments[0]
        strategy = (
            SyncStrategyLevel.CONSTANT_SPEED
            if abs(seg.speed - 1.0) > speed_tolerance
            else SyncStrategyLevel.CONSTANT_OFFSET
        )
        return AlignmentDocument(
            strategy=strategy,
            confidence=mean_conf,
            global_offset_seconds=seg.offset_seconds,
            global_speed=seg.speed,
            segments=segments,
            notes=notes,
            evidence={
                "pair_count": len(pairs),
                "donor_chunks": len(donor_keep),
                "fitted_chunks": 1,
                "recovered_chunks": 0,
            },
        )

    return AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=mean_conf,
        global_offset_seconds=None,
        global_speed=float(sum(speeds) / len(speeds)) if speed_span < speed_tolerance else None,
        segments=segments,
        notes=notes,
        evidence={
            "pair_count": len(pairs),
            "donor_chunks": len(donor_keep),
            "fitted_chunks": len(segments),
            "recovered_chunks": recovered_count,
            "speed_span": round(speed_span, 4),
        },
    )


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
    donor_exclude: list[tuple[float, float]] | None = None,
    timing_hints: list[ChunkTimingHint] | None = None,
) -> AlignmentDocument:
    """Prefer a visual timing model; do not let weak audio override it.

    Policy:
    1. If ``donor_exclude`` spans exist, fit **per complementary donor chunk**
       first (mid-roll ads ⇒ multiple legitimate offsets).
    2. Else if enough visual pairs exist, fit offset/speed/piecewise on **visual
       only** with a looser tolerance (frame sampling).
    3. Accept global visual models immediately; also prefer visual piecewise when
       it is at least moderately confident.
    4. Otherwise fall back to fused evidence with **downweighted** audio so dense
       cross-language landmark noise cannot dominate.
    """
    from jarvis_media_processor.alignment.evidence import downweight_pairs, merge_evidence
    from jarvis_media_processor.alignment.unrelated import complementary_donor_intervals

    notes: list[str] = []
    if donor_exclude and donor_duration is not None and len(visual_pairs) >= min_visual_pairs:
        exclude_intervals = [
            TimelineInterval(start=a, end=b, timeline="donor", reason="excluded")
            for a, b in donor_exclude
            if b > a
        ]
        keep = complementary_donor_intervals(donor_duration, exclude_intervals)
        chunked = estimate_alignment_per_donor_chunk(
            visual_pairs,
            keep,
            base_duration=base_duration,
            donor_duration=donor_duration,
            inlier_tolerance=visual_inlier_tolerance,
            min_inlier_ratio=visual_min_inlier_ratio,
            min_chunk_pairs=max(4, min_visual_pairs // 2),
            speed_min=speed_min,
            speed_max=speed_max,
            speed_tolerance=speed_tolerance,
            timing_hints=timing_hints,
        )
        fitted = int(chunked.evidence.get("fitted_chunks", 0)) if chunked else 0
        keeps = int(chunked.evidence.get("donor_chunks", 0)) if chunked else 0
        if (
            chunked is not None
            and chunked.segments
            and chunked.confidence >= 0.5
            and fitted >= 1
            and (
                # All keep-ranges fitted (global or piecewise dual-offset).
                fitted >= keeps
                # Or at least two independent chunk models (mid-roll ad case).
                or (
                    chunked.strategy == SyncStrategyLevel.PIECEWISE
                    and fitted >= 2
                )
            )
        ):
            chunked.notes = [
                "Visual-first: per-chunk fit outside unrelated donor spans.",
                *chunked.notes,
            ]
            evidence = dict(chunked.evidence)
            evidence["visual_pairs"] = len(visual_pairs)
            evidence["audio_pairs"] = len(audio_pairs)
            evidence["fusion"] = "visual_chunked_exclusions"
            chunked.evidence = evidence
            return chunked
        notes.append(
            "Visual-first: per-chunk fit weak; trying whole-timeline visual model."
        )

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
