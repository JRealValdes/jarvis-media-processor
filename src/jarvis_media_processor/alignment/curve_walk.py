"""Walk activity envelopes like a human matching derivative curves.

Lock characteristic shapes on the base, find them on the donor with an
expanding offset search, follow while they track (optional mild speed), and
when they diverge locate either surplus donor material (ads) or missing-base
stretches (content only on the base timeline).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jarvis_media_processor.alignment.activity_sync import _resample_to_period
from jarvis_media_processor.alignment.models import TimelineInterval
from jarvis_media_processor.video.activity import (
    ActivityEnvelope,
    activity_peak_times,
    local_activity_score,
)


@dataclass(frozen=True)
class CurveLock:
    """Locked linear map ``donor ≈ speed * base + offset`` at a base time."""

    speed: float
    offset_seconds: float
    base_time: float
    donor_time: float
    score: float


@dataclass(frozen=True)
class CurveWalkResult:
    """Spans discovered by walking the activity curves."""

    unrelated_donor_intervals: list[TimelineInterval]
    missing_base_intervals: list[TimelineInterval]
    locks: list[CurveLock]
    notes: list[str]


def local_curve_score(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_time: float,
    speed: float,
    half_window: float,
    period: float,
) -> float:
    """Normalized correlation of local activity shapes around a hypothesized lock."""
    _ = period  # kept for call-site compatibility; period comes from envelopes
    return local_activity_score(
        base,
        donor,
        base_time=base_time,
        donor_time=donor_time,
        speed=speed,
        half_window=half_window,
    )


def _expanding_offsets(max_offset: float) -> list[float]:
    """Offsets to try, starting near zero then expanding (both signs)."""
    steps = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 600.0, 1200.0]
    out: list[float] = []
    for step in steps:
        if step > max_offset + 1e-9:
            break
        out.append(step)
        if step > 0:
            out.append(-step)
    if max_offset > 0 and max_offset not in out and -max_offset not in out:
        out.extend([max_offset, -max_offset])
    return out


def _surplus_slip_candidates(
    min_surplus_seconds: float,
    max_surplus_seconds: float,
    *,
    hop: float,
) -> list[float]:
    """Forward donor slips to try when resuming after a break.

    Coarse expanding steps cover multi-minute ads; a dense hop grid up to ~2
    minutes catches short mid-roll blocks (e.g. 15s) that fall between 10 and 20.
    """
    slips: list[float] = []
    for step in _expanding_offsets(max_surplus_seconds):
        if step >= min_surplus_seconds:
            slips.append(float(step))
    dense_cap = min(max_surplus_seconds, 120.0)
    # 1s dense grid: ad lengths are often odd (15s, 25s) and a 2s hop skips them.
    dense_hop = 1.0
    t = float(min_surplus_seconds)
    while t <= dense_cap + 1e-9:
        slips.append(float(t))
        t += dense_hop
    _ = hop  # retained for call-site compatibility
    # Unique, shortest-first then longer (ranking still prefers score).
    return sorted(set(round(s, 3) for s in slips))


def _score_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_time: float,
    speed: float,
    half_window: float,
    period: float,
    min_score: float,
) -> CurveLock | None:
    if donor.times.size == 0:
        return None
    if donor_time < float(donor.times[0]) or donor_time > float(donor.times[-1]):
        return None
    if base_time < float(base.times[0]) or base_time > float(base.times[-1]):
        return None
    score = local_curve_score(
        base,
        donor,
        base_time=base_time,
        donor_time=donor_time,
        speed=speed,
        half_window=half_window,
        period=period,
    )
    if score < min_score:
        return None
    return CurveLock(
        speed=float(speed),
        offset_seconds=float(donor_time - speed * base_time),
        base_time=float(base_time),
        donor_time=float(donor_time),
        score=float(score),
    )


def _backwards_refine_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    lock: CurveLock,
    *,
    half_window: float,
    period: float,
    hop: float,
    min_score: float,
) -> CurveLock:
    """Walk backwards from a lock to the earliest still-matching coincidence.

    Stops when the local score falls well below the lock quality. A weak
    absolute floor alone walks into leading credits / grade mismatch and can
    turn a strong mid-timeline lock into a marginal head hit.
    """
    speed = lock.speed
    offset = lock.offset_seconds
    base_t = lock.base_time
    best = lock
    # Stay near the lock's own confidence; never below the caller floor.
    quality_floor = max(float(min_score), float(lock.score) * 0.75)
    t = base_t - hop
    floor = float(base.times[0]) + half_window
    while t >= floor:
        donor_t = speed * t + offset
        scored = _score_lock(
            base,
            donor,
            base_time=t,
            donor_time=donor_t,
            speed=speed,
            half_window=half_window,
            period=period,
            min_score=quality_floor,
        )
        if scored is None:
            break
        best = scored
        t -= hop
    return best


def find_initial_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    half_window: float = 12.0,
    min_score: float = 0.35,
    max_offset_seconds: float = 1200.0,
    speed_candidates: tuple[float, ...] = (1.0, 0.96, 1.04, 0.92, 1.08),
    frame_confirm: Callable[[float, float], bool] | None = None,
) -> CurveLock | None:
    """Find the first base↔donor coincidence via chronological curve search.

    Locks on activity derivatives (cross-encode safe). Strong activity locks
    are accepted without frames. Marginal locks (score < 0.5) require picture
    support when ``frame_confirm`` is available — otherwise a weak wrong-speed
    head hit can derail the whole walk on grade-mismatched pairs.
    """
    period = float(np.median([base.sample_period, donor.sample_period])) or 1.0
    hop = max(period, 2.0)
    base_start = float(base.times[0]) + half_window
    donor_start = float(donor.times[0]) + half_window
    base_end = float(base.times[-1]) - half_window
    donor_end = float(donor.times[-1]) - half_window
    if base_end <= base_start or donor_end <= donor_start:
        return None

    def _try_and_confirm(lock: CurveLock | None) -> CurveLock | None:
        if lock is None:
            return None
        # Keep confirm shorter than half_window. A 12s confirm from the first
        # legal base sample walks into an early mid-roll ad and rejects the
        # true 1.0 lock, after which a weak wrong-speed ring hit can win.
        if not _confirm_lock(
            base,
            donor,
            lock,
            half_window=half_window,
            period=period,
            hop=hop,
            min_score=min_score * 0.85,
            confirm_seconds=6.0,
            frame_confirm=None,
        ):
            return None
        refined = _backwards_refine_lock(
            base,
            donor,
            lock,
            half_window=half_window,
            period=period,
            hop=hop,
            min_score=min_score * 0.8,
        )
        if refined is None:
            return None
        # Strong curve agreement stands alone (cross-encode safe). Marginal
        # hits need a frame vote when available so a 0.38 wrong-speed lock at
        # the donor head cannot beat a later true ~0.7 lock.
        if refined.score >= 0.5:
            return refined
        if frame_confirm is None:
            return refined
        if frame_confirm(refined.base_time, refined.donor_time):
            return refined
        return None

    def _best_of(candidates: list[CurveLock]) -> CurveLock | None:
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.score, -abs(c.speed - 1.0)))

    peaks = activity_peak_times(base, min_prominence=0.12)
    if not peaks:
        bt, bv = _resample_to_period(base, period)
        if bt.size == 0:
            return None
        order = np.argsort(bv)[::-1][:16]
        peaks = [float(bt[i]) for i in order]

    # Dense chronological base samples over the searchable head, then peaks.
    # Density recovers start-aligned / missing-credits locks without a
    # separate special-case pass.
    head_limit = min(base_end, base_start + max_offset_seconds)
    anchors: list[float] = []
    for t in np.arange(base_start, head_limit + hop * 0.5, hop):
        anchors.append(float(t))
    for t in sorted(peaks):
        if head_limit < t <= base_end and (
            not anchors or abs(t - anchors[-1]) >= hop
        ):
            anchors.append(float(t))
    if len(anchors) > 400:
        # Keep the head dense; thin only the far tail.
        keep_head = int(min(base_start + 600.0, head_limit) - base_start) // int(hop) + 1
        keep_head = max(keep_head, 60)
        head = anchors[:keep_head]
        tail = anchors[keep_head :: max(1, (len(anchors) - keep_head) // 80)]
        anchors = head + tail

    radii: list[float] = []
    for step in _expanding_offsets(max_offset_seconds):
        r = abs(step)
        if r not in radii:
            radii.append(r)

    # Ring 0 — donor head, chronological per speed. Take the *first* base
    # time that confirms (not the global max score): a later high-score false
    # peak (e.g. base=88) must not hide an earlier true lock (base=42→32).
    # Among speeds that confirm, prefer near-1.0 to avoid cascade false ads.
    ring0_hits: list[CurveLock] = []
    for speed in speed_candidates:
        for base_t in anchors:
            cand = _score_lock(
                base,
                donor,
                base_time=base_t,
                donor_time=donor_start,
                speed=speed,
                half_window=half_window,
                period=period,
                min_score=min_score,
            )
            confirmed = _try_and_confirm(cand)
            if confirmed is not None:
                ring0_hits.append(confirmed)
                break
    if ring0_hits:
        # Earliest coincidence wins; among ties, prefer speed≈1 and higher score.
        return min(
            ring0_hits,
            key=lambda c: (c.base_time, abs(c.speed - 1.0), -c.score),
        )

    # Later rings — chronological base × expanding offset.
    for base_t in anchors:
        for radius in radii:
            offsets = [0.0] if radius == 0 else [radius, -radius]
            if radius >= 10:
                offsets.extend([radius * 0.5, -radius * 0.5])
            if radius >= 40:
                offsets.extend(
                    [radius * 0.75, -radius * 0.75, radius * 0.25, -radius * 0.25]
                )
            ring_hits: list[CurveLock] = []
            for speed in speed_candidates:
                for offset in offsets:
                    donor_t = speed * base_t + offset
                    if abs(donor_t - donor_start) < hop * 0.5:
                        continue
                    cand = _score_lock(
                        base,
                        donor,
                        base_time=base_t,
                        donor_time=donor_t,
                        speed=speed,
                        half_window=half_window,
                        period=period,
                        min_score=min_score,
                    )
                    if cand is not None:
                        ring_hits.append(cand)
            confirmed = _try_and_confirm(_best_of(ring_hits))
            if confirmed is not None:
                return confirmed
    return None


def _confirm_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    lock: CurveLock,
    *,
    half_window: float,
    period: float,
    hop: float,
    min_score: float,
    confirm_seconds: float,
    frame_confirm: Callable[[float, float], bool] | None = None,
) -> bool:
    """Require the lock to keep matching while walking a short stretch.

    Uses mean score with a soft floor so a single noisy hop cannot reject a
    otherwise solid near-1.0 speed lock (cross-encode activity is jittery).

    When ``frame_confirm`` is set, also require actual frames to agree on a
    majority of the confirm hops (derivative propose → frame confirm).
    """
    t = lock.base_time
    end = min(float(base.times[-1]) - half_window, t + confirm_seconds)
    scores: list[float] = []
    frame_checks = 0
    frame_hits = 0
    while t <= end:
        donor_t = lock.speed * t + lock.offset_seconds
        if donor_t < float(donor.times[0]) or donor_t > float(donor.times[-1]):
            return False
        score = local_curve_score(
            base,
            donor,
            base_time=t,
            donor_time=donor_t,
            speed=lock.speed,
            half_window=half_window,
            period=period,
        )
        # Hard reject only total collapse; mild dips are OK if the mean holds.
        if score < min_score * 0.55:
            return False
        scores.append(score)
        if frame_confirm is not None:
            frame_checks += 1
            if frame_confirm(t, donor_t):
                frame_hits += 1
        t += hop
    if not scores:
        return False
    covered = len(scores) * hop
    if covered < confirm_seconds * 0.5:
        return False
    mean_sc = float(np.mean(scores))
    if mean_sc < min_score or min(scores) < min_score * 0.7:
        return False
    if frame_confirm is not None and frame_checks > 0:
        return frame_hits >= max(1, (frame_checks + 1) // 2)
    return True


def _speed_refine_candidates(
    current_speed: float,
    speed_candidates: tuple[float, ...],
) -> list[float]:
    """Speeds to try at a break, excluding the failing current speed."""
    deltas = (-0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08)
    out: list[float] = []
    seen: set[float] = set()
    for raw in (*speed_candidates, *(current_speed + d for d in deltas)):
        speed = round(float(raw), 4)
        if speed < 0.85 or speed > 1.2:
            continue
        if abs(speed - current_speed) < 0.008:
            continue
        if speed in seen:
            continue
        seen.add(speed)
        out.append(speed)
    return out


def _forward_map_mean_score(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    break_base: float,
    speed: float,
    offset: float,
    half_window: float,
    period: float,
    hop: float,
    horizon_seconds: float,
) -> float | None:
    """Mean local score walking ``horizon_seconds`` ahead under a linear map."""
    scores: list[float] = []
    t = float(break_base)
    end = min(float(base.times[-1]) - half_window, t + horizon_seconds)
    donor_lo = float(donor.times[0])
    donor_hi = float(donor.times[-1])
    while t <= end + 1e-9:
        donor_t = speed * t + offset
        if donor_t < donor_lo - 1 or donor_t > donor_hi + 1:
            return None
        scores.append(
            local_curve_score(
                base,
                donor,
                base_time=t,
                donor_time=donor_t,
                speed=speed,
                half_window=half_window,
                period=period,
            )
        )
        t += hop
    if len(scores) < 2:
        return None
    return float(np.mean(scores))


def _try_speed_refine_at_break(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    break_base: float,
    speed: float,
    offset: float,
    half_window: float,
    period: float,
    hop: float,
    min_score: float,
    confirm_seconds: float,
    speed_candidates: tuple[float, ...],
    donor_start: float,
    donor_end: float,
) -> CurveLock | None:
    """Hypothesis: mild break is a wrong speed, not an ad.

    The predicted donor time under the broken map may already have slipped, so
    search a *local* donor neighborhood × nearby speeds. A candidate must
    sustain over a longer horizon and beat continuing with the old speed by a
    clear margin (avoids accepting a bogus speed on self-similar pulses).
    """
    predicted = speed * break_base + offset
    if predicted < donor_start - 1 or predicted > donor_end + 1:
        return None
    # Longer than confirm_seconds: wrong speeds often look fine for 1–2 hops.
    horizon = max(confirm_seconds * 2.0, confirm_seconds + 4.0, 10.0)
    # Slip under a wrong speed grows gradually. Keep the donor search *tight*
    # so a mid-roll ad (multi-second jump) cannot be "explained" as speed.
    radius = min(max(half_window, 6.0), 10.0)
    old_mean = _forward_map_mean_score(
        base,
        donor,
        break_base=break_base,
        speed=speed,
        offset=offset,
        half_window=half_window,
        period=period,
        hop=hop,
        horizon_seconds=horizon,
    )
    old_ref = old_mean if old_mean is not None else 0.0
    best: CurveLock | None = None
    best_key: tuple[float, float] = (-1.0, float("-inf"))
    donor_lo = max(donor_start, predicted - radius)
    donor_hi = min(donor_end, predicted + radius)
    donor_grid = np.arange(donor_lo, donor_hi + hop * 0.5, hop, dtype=np.float64)
    for trial_speed in _speed_refine_candidates(speed, speed_candidates):
        for donor_t in donor_grid:
            trial_off = float(donor_t) - trial_speed * break_base
            instant = local_curve_score(
                base,
                donor,
                base_time=break_base,
                donor_time=float(donor_t),
                speed=trial_speed,
                half_window=half_window,
                period=period,
            )
            if instant < min_score:
                continue
            mean_sc = _forward_map_mean_score(
                base,
                donor,
                break_base=break_base,
                speed=trial_speed,
                offset=trial_off,
                half_window=half_window,
                period=period,
                hop=hop,
                horizon_seconds=horizon,
            )
            if mean_sc is None or mean_sc < min_score:
                continue
            # Weak forward means are typical of fitting through an ad/flat;
            # a real speed recovery should look clearly healthy.
            if mean_sc < max(0.55, min_score + 0.2):
                continue
            if mean_sc < old_ref + 0.08:
                continue
            trial_lock = CurveLock(
                speed=float(trial_speed),
                offset_seconds=float(trial_off),
                base_time=float(break_base),
                donor_time=float(donor_t),
                score=float(mean_sc),
            )
            if not _confirm_lock(
                base,
                donor,
                trial_lock,
                half_window=half_window,
                period=period,
                hop=hop,
                min_score=min_score,
                confirm_seconds=confirm_seconds,
                frame_confirm=None,
            ):
                continue
            # Prefer higher mean; on ties prefer smaller donor slip (continuity).
            key = (mean_sc, -abs(float(donor_t) - predicted))
            if key > best_key:
                best_key = key
                best = trial_lock
    return best


def walk_activity_curves(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    half_window: float = 12.0,
    hop_seconds: float = 2.0,
    min_score: float = 0.3,
    break_score: float = 0.22,
    confirm_seconds: float = 6.0,
    min_surplus_seconds: float = 8.0,
    max_surplus_seconds: float = 1200.0,
    missing_base_fill_seconds: float = 3.0,
    max_offset_seconds: float = 1200.0,
    speed_candidates: tuple[float, ...] = (1.0, 0.96, 1.04, 0.92, 1.08),
    frame_confirm: Callable[[float, float], bool] | None = None,
) -> CurveWalkResult:
    """Follow base→donor activity curves; emit surplus-donor and missing-base spans.

    Human-inspired loop:
    1. Lock a characteristic base shape on the donor (expanding offset).
    2. Optionally confirm the lock with actual frames (``frame_confirm``).
    3. Walk forward while local curves keep matching.
    4. On a clear break, compare hypotheses: (a) refine speed for continued
       content, (b) surplus donor (ad), (c) missing base — pick by score with a
       mild preference for speed refine on ties (avoid inventing ads).
    5. If the donor does not resume but a later base time rematches at the same
       donor point → missing-base interval (fill from base audio later).
    """
    notes: list[str] = []
    if base.times.size < 8 or donor.times.size < 8:
        return CurveWalkResult([], [], [], ["Curve walk skipped: envelopes too short."])

    period = float(np.median([base.sample_period, donor.sample_period])) or 1.0
    lock = find_initial_lock(
        base,
        donor,
        half_window=half_window,
        min_score=min_score,
        max_offset_seconds=max_offset_seconds,
        speed_candidates=speed_candidates,
        frame_confirm=frame_confirm,
    )
    if lock is None:
        return CurveWalkResult(
            [],
            [],
            [],
            ["Curve walk: no initial lock found."],
        )

    notes.append(
        f"Curve lock speed={lock.speed:.4f} offset={lock.offset_seconds:.3f}s "
        f"at base={lock.base_time:.1f}s donor={lock.donor_time:.1f}s "
        f"score={lock.score:.3f}."
    )
    locks = [lock]
    unrelated: list[TimelineInterval] = []
    missing_base: list[TimelineInterval] = []

    # Head gaps implied by the first coincidence (after backwards refine).
    # Only when the lock itself sits near a timeline head — a mid-file lock
    # must not invent a multi-minute "leading" gap.
    head_base = float(base.times[0])
    head_donor = float(donor.times[0])
    base_floor = head_base + half_window
    donor_floor = head_donor + half_window
    lock_near_base_head = lock.base_time - base_floor <= max_offset_seconds * 0.25
    lock_near_donor_head = lock.donor_time - donor_floor <= half_window * 2
    if (
        lock_near_donor_head
        and lock.base_time - base_floor >= missing_base_fill_seconds
    ):
        missing_base.append(
            TimelineInterval(
                start=head_base,
                end=float(lock.base_time),
                timeline="base",
                reason="leading_missing_base",
            )
        )
        notes.append(
            f"Leading missing base 0-{lock.base_time:.1f}s "
            "(donor starts matching later; fill from base track 0)."
        )
    if (
        lock_near_base_head
        and lock.donor_time - donor_floor >= min_surplus_seconds
    ):
        unrelated.append(
            TimelineInterval(
                start=head_donor,
                end=float(lock.donor_time),
                timeline="donor",
                reason="leading_surplus_donor",
            )
        )
        notes.append(
            f"Leading surplus donor 0-{lock.donor_time:.1f}s "
            "(ads / junk before first match)."
        )

    speed = lock.speed
    offset = lock.offset_seconds
    anchor_offset = lock.offset_seconds
    base_t = lock.base_time
    base_end = float(base.times[-1]) - half_window
    donor_end = float(donor.times[-1])
    donor_start = float(donor.times[0])
    bad_run = 0.0
    hop = max(hop_seconds, period)
    max_offset_drift = hop  # keep the lock map; do not accumulate phase slip
    # Adaptive floor: break when score falls well below recent good tracking.
    # Seed from the confirm stretch, not the single lock peak (often low).
    good_ema = max(lock.score, min_score, break_score + 0.15)

    while base_t <= base_end:
        donor_t = speed * base_t + offset
        if donor_t < donor_start - 1 or donor_t > donor_end + 1:
            # Ran off the donor — treat remaining base as missing if long enough.
            if base_end - base_t >= missing_base_fill_seconds:
                missing_base.append(
                    TimelineInterval(
                        start=float(base_t),
                        end=float(base_end),
                        timeline="base",
                        reason="missing_base_beyond_donor",
                    )
                )
            break

        score = local_curve_score(
            base,
            donor,
            base_time=base_t,
            donor_time=donor_t,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        # Mild ±hop only, clamped to the last solid lock offset so refine
        # cannot slowly walk away from a good map and invent surplus.
        best_off = offset
        best_sc = score
        for delta in (-hop, 0.0, hop):
            trial_off = offset + delta
            if abs(trial_off - anchor_offset) > max_offset_drift:
                continue
            trial_d = speed * base_t + trial_off
            if trial_d < donor_start - 1 or trial_d > donor_end + 1:
                continue
            sc = local_curve_score(
                base,
                donor,
                base_time=base_t,
                donor_time=trial_d,
                speed=speed,
                half_window=half_window,
                period=period,
            )
            if sc > best_sc:
                best_sc = sc
                best_off = trial_off
        score = best_sc

        track_floor = max(break_score, good_ema * 0.55)
        if score >= track_floor:
            offset = best_off
            good_ema = 0.85 * good_ema + 0.15 * max(score, good_ema)
            bad_run = 0.0
            base_t += hop
            continue

        bad_run += hop
        if bad_run < confirm_seconds:
            base_t += hop
            continue

        # Confirmed degradation — compare hypotheses at the break origin:
        # speed refine (continued content) vs surplus donor vs missing base.
        break_base = base_t - bad_run
        resume_floor = max(min_score + 0.08, good_ema * 0.75)
        break_donor = speed * break_base + offset
        break_donor = max(donor_start, min(donor_end, break_donor))

        # Speed refine uses a moderate floor: good_ema can be ~1.0 right before
        # a gradual drift break, and that bar would reject every recovery.
        speed_floor = max(min_score, break_score + 0.1)
        speed_hit = _try_speed_refine_at_break(
            base,
            donor,
            break_base=break_base,
            speed=speed,
            offset=offset,
            half_window=half_window,
            period=period,
            hop=hop,
            min_score=speed_floor,
            confirm_seconds=confirm_seconds,
            speed_candidates=speed_candidates,
            donor_start=donor_start,
            donor_end=donor_end,
        )

        # Moderate floor: a true post-ad resume can score lower than a later
        # rhythmic alias; high good_ema must not hide the earliest real resume.
        surplus_floor = max(min_score, break_score + 0.1)
        # Breaks often fire a few hops before the cut (half-window bleed). Try
        # nearby base anchors so a short early ad is not missed at the origin.
        surplus_hit = None
        surplus_hit_anchor = break_base
        for _anchor in (
            break_base,
            break_base + hop,
            break_base + 2 * hop,
            max(half_window, break_base - hop),
        ):
            if _anchor < half_window or _anchor > base_end:
                continue
            _hit = _find_forward_surplus(
                base,
                donor,
                base_time=float(_anchor),
                offset=offset,
                speed=speed,
                half_window=half_window,
                period=period,
                min_score=surplus_floor,
                min_surplus_seconds=min_surplus_seconds,
                max_surplus_seconds=max_surplus_seconds,
                break_score=break_score,
                donor_start=donor_start,
                donor_end=donor_end,
                frame_confirm=frame_confirm,
                confirm_seconds=confirm_seconds,
                hop_seconds=hop,
            )
            if _hit is None:
                continue
            surplus_hit = _hit
            surplus_hit_anchor = float(_anchor)
            break
        # Long weak resumes invent false ads; multi-minute jumps need frames.
        if surplus_hit is not None:
            _gs, _ge, _no, _rs = surplus_hit
            # If resume was found on a later base anchor, keep the gap start at
            # the break donor so early ad seconds are not left outside surplus.
            if _gs > break_donor + 1e-6:
                surplus_hit = (float(break_donor), _ge, _no, _rs)
                _gs = float(break_donor)
            _slip = _ge - _gs
            if _slip > 45.0 and _rs < 0.55:
                surplus_hit = None
            elif _slip > 90.0 and frame_confirm is not None:
                if not frame_confirm(
                    surplus_hit_anchor, speed * surplus_hit_anchor + _no
                ):
                    surplus_hit = None

        resume_min = min_score + 0.05
        # Secondary farthest-resume search only without frames. With frames the
        # primary grid already ranks earliest picture-supported slip; the
        # secondary pass otherwise latches periodic activity aliases.
        surplus = None
        surplus_anchor = break_base
        if surplus_hit is None and frame_confirm is None:
            for anchor in (
                break_base,
                max(half_window, break_base - hop),
                break_base + hop,
            ):
                if anchor < half_window or anchor > base_end:
                    continue
                cand = _search_donor_resume(
                    base,
                    donor,
                    base_time=float(anchor),
                    donor_search_start=break_donor + min_surplus_seconds,
                    speed=speed,
                    half_window=half_window,
                    period=period,
                    min_score=resume_min,
                    max_ahead=max_surplus_seconds,
                    hop=max(hop, 1.0),
                )
                if cand is None:
                    continue
                surplus_end, resume_score = cand
                if surplus_end - break_donor < min_surplus_seconds:
                    continue
                if not _donor_gap_is_unrelated(
                    base,
                    donor,
                    base_time=float(anchor),
                    donor_gap_start=break_donor,
                    donor_gap_end=surplus_end,
                    speed=speed,
                    half_window=half_window,
                    period=period,
                    break_score=break_score,
                ):
                    continue
                surplus = cand
                surplus_anchor = float(anchor)
                break
            if surplus is not None:
                surplus_end, resume_score = surplus
                _slip2 = surplus_end - break_donor
                if _slip2 > 45.0 and resume_score < 0.55:
                    surplus = None

        min_missing = max(missing_base_fill_seconds, confirm_seconds)
        missing_hit = _find_forward_missing(
            base,
            donor,
            break_base=break_base,
            break_donor=break_donor,
            speed=speed,
            offset=offset,
            half_window=half_window,
            period=period,
            min_score=resume_min,
            min_missing_seconds=min_missing,
            max_ahead=max_surplus_seconds,
            confirm_seconds=confirm_seconds,
            hop_seconds=hop,
            frame_confirm=frame_confirm,
        )
        # Picture-supported surplus is a donor insert; drop competing missing.
        if (
            missing_hit is not None
            and surplus_hit is not None
            and frame_confirm is not None
        ):
            missing_hit = None
        # Speed refine vs missing: short framed "missing" jumps on cross-encode
        # film are usually lock noise while a mild speed still explains the
        # break (e.g. ~15s alias score ~0.79 beating speed ~0.64). Real mid
        # cuts (synth ~30s) typically leave speed_hit None. Only a *long*
        # strong missing may veto speed; otherwise prefer speed and drop the
        # short missing hypothesis.
        if missing_hit is not None and speed_hit is not None:
            _miss_gap = float(missing_hit[0]) - float(break_base)
            if _miss_gap < 30.0:
                missing_hit = None
            elif frame_confirm is not None and float(missing_hit[1]) >= 0.65:
                speed_hit = None


        # Rank hypotheses: (kind, score, payload). Human order at a break:
        # speed refine (no jump) vs surplus donor vs missing base. Speed gets a
        # small edge when both fire without picture-backed jumps; surplus must
        # not win on a tiny score edge alone.
        ranked: list[tuple[str, float, object]] = []
        if speed_hit is not None:
            ranked.append(("speed", float(speed_hit.score) + 0.08, speed_hit))
        if surplus_hit is not None:
            ranked.append(("surplus_fwd", float(surplus_hit[3]), surplus_hit))
        if surplus is not None:
            ranked.append(
                ("surplus_search", float(surplus[1]), (surplus, surplus_anchor))
            )
        if missing_hit is not None:
            ranked.append(("missing", float(missing_hit[1]), missing_hit))

        if not ranked:
            notes.append(
                f"Curve break without resume at base={break_base:.1f}s "
                f"donor≈{break_donor:.1f}s; continuing."
            )
            bad_run = 0.0
            base_t += hop
            continue

        kind, _score, payload = max(ranked, key=lambda item: item[1])

        if kind == "speed":
            new_lock = payload  # type: ignore[assignment]
            assert isinstance(new_lock, CurveLock)
            speed = new_lock.speed
            offset = new_lock.offset_seconds
            anchor_offset = offset
            locks.append(new_lock)
            notes.append(
                f"Speed refine at base={break_base:.1f}s → speed={speed:.4f} "
                f"offset={offset:.3f}s (score={new_lock.score:.3f})."
            )
            good_ema = max(new_lock.score, min_score)
            bad_run = 0.0
            base_t = break_base + hop
            continue

        if kind == "surplus_fwd":
            gap_start, gap_end, new_off, resume_score = payload  # type: ignore[misc]
            unrelated.append(
                TimelineInterval(
                    start=float(gap_start),
                    end=float(gap_end),
                    timeline="donor",
                    reason="curve_break_surplus_donor",
                )
            )
            offset = float(new_off)
            anchor_offset = offset
            locks.append(
                CurveLock(
                    speed=speed,
                    offset_seconds=offset,
                    base_time=break_base,
                    donor_time=speed * break_base + offset,
                    score=float(resume_score),
                )
            )
            notes.append(
                f"Surplus donor {gap_start:.1f}-{gap_end:.1f}s "
                f"(resume score={resume_score:.3f})."
            )
            good_ema = max(float(resume_score), min_score)
            bad_run = 0.0
            base_t = break_base + hop
            continue

        if kind == "surplus_search":
            (surplus_end, resume_score), surplus_anchor = payload  # type: ignore[misc]
            unrelated.append(
                TimelineInterval(
                    start=float(break_donor),
                    end=float(surplus_end),
                    timeline="donor",
                    reason="curve_break_surplus_donor",
                )
            )
            offset = float(surplus_end) - speed * float(surplus_anchor)
            anchor_offset = offset
            locks.append(
                CurveLock(
                    speed=speed,
                    offset_seconds=offset,
                    base_time=float(surplus_anchor),
                    donor_time=float(surplus_end),
                    score=float(resume_score),
                )
            )
            notes.append(
                f"Surplus donor {break_donor:.1f}-{surplus_end:.1f}s "
                f"(resume score={resume_score:.3f})."
            )
            good_ema = max(float(resume_score), min_score)
            bad_run = 0.0
            base_t = float(surplus_anchor) + hop
            continue

        # kind == "missing"
        resume_base, resume_score, resume_donor = payload  # type: ignore[misc]
        missing_base.append(
            TimelineInterval(
                start=float(break_base),
                end=float(resume_base),
                timeline="base",
                reason="curve_break_missing_base",
            )
        )
        offset = float(resume_donor) - speed * float(resume_base)
        anchor_offset = offset
        locks.append(
            CurveLock(
                speed=speed,
                offset_seconds=offset,
                base_time=float(resume_base),
                donor_time=float(resume_donor),
                score=float(resume_score),
            )
        )
        notes.append(
            f"Missing base {break_base:.1f}-{resume_base:.1f}s "
            f"(resume score={resume_score:.3f})."
        )
        good_ema = max(float(resume_score), min_score)
        bad_run = 0.0
        base_t = float(resume_base) + hop
        continue

    return CurveWalkResult(
        unrelated_donor_intervals=_merge_same_timeline(unrelated),
        missing_base_intervals=_merge_same_timeline(missing_base),
        locks=locks,
        notes=notes,
    )


def _merge_same_timeline(intervals: list[TimelineInterval]) -> list[TimelineInterval]:
    """Merge overlapping/adjacent intervals on the same timeline."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x.timeline, x.start, x.end))
    out: list[TimelineInterval] = []
    cur = ordered[0]
    for item in ordered[1:]:
        if (
            item.timeline == cur.timeline
            and item.start <= cur.end + 1e-6
        ):
            cur = TimelineInterval(
                start=cur.start,
                end=max(cur.end, item.end),
                timeline=cur.timeline,
                reason=cur.reason if cur.reason == item.reason else item.reason,
            )
        else:
            out.append(cur)
            cur = item
    out.append(cur)
    return out


def _find_forward_surplus(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    offset: float,
    speed: float,
    half_window: float,
    period: float,
    min_score: float,
    min_surplus_seconds: float,
    max_surplus_seconds: float,
    break_score: float,
    donor_start: float,
    donor_end: float,
    frame_confirm: Callable[[float, float], bool] | None = None,
    confirm_seconds: float = 6.0,
    hop_seconds: float = 2.0,
) -> tuple[float, float, float, float] | None:
    """If a forward donor jump restores the curve, return surplus span + new offset.

    Returns ``(gap_start, gap_end, new_offset, resume_score)`` or ``None``.

    Mid-roll *film* inserts can produce short false activity peaks; require a
    sustained resume and prefer the longest adequate jump (not the loudest).
    When ``frame_confirm`` is available, prefer candidates that also pass it,
    but fall back to activity-only so cross-encode black ads still resume.
    """
    hop = max(hop_seconds, period)
    confirmed: list[tuple[float, float, float]] = []  # slip, score, trial_off
    for step in _surplus_slip_candidates(
        min_surplus_seconds, max_surplus_seconds, hop=hop
    ):
        trial_off = offset + step
        trial_d = speed * base_time + trial_off
        if trial_d < donor_start - 1 or trial_d > donor_end + 1:
            continue
        sc = local_curve_score(
            base,
            donor,
            base_time=base_time,
            donor_time=trial_d,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        if sc < min_score:
            continue
        trial_lock = CurveLock(
            speed=speed,
            offset_seconds=float(trial_off),
            base_time=float(base_time),
            donor_time=float(trial_d),
            score=float(sc),
        )
        if not _confirm_lock(
            base,
            donor,
            trial_lock,
            half_window=half_window,
            period=period,
            hop=hop,
            min_score=min_score,
            confirm_seconds=confirm_seconds,
            frame_confirm=None,  # activity sustained first; frames below
        ):
            continue
        confirmed.append((float(step), float(sc), float(trial_off)))
    if not confirmed:
        return None

    pool = confirmed

    def _frames_support(trial_off: float) -> bool:
        """True if pHash agrees near the break for this constant offset.

        Breaks often fire a few hops *before* the exact cut, so probe a short
        forward window on the base with the candidate offset held fixed.
        """
        if frame_confirm is None:
            return False
        # 1s probes: breaks often sit a few seconds early and a 2s hop can
        # skip the only frame-agreeing anchor (e.g. break at 52, cut at 55).
        probe_hop = 1.0
        limit = max(confirm_seconds, probe_hop)
        delta = 0.0
        while delta <= limit + 1e-9:
            anchor = float(base_time) + delta
            donor_t = speed * anchor + float(trial_off)
            if frame_confirm(anchor, donor_t):
                return True
            delta += probe_hop
        return False

    # Prefer earliest frame-supported resume when available. Filtering the pool
    # to frame hits and then maximizing score preferred later aliases; scanning
    # earliest-first among frame-supported slips keeps the true post-ad lock.
    # When frames exist but none agree, do not fall back to mid-strength
    # activity aliases (periodic scene repeats / missing-base inversions);
    # only a very strong activity resume is trusted without picture support.
    framed = [item for item in pool if _frames_support(item[2])]
    if framed:
        ordered = sorted(framed, key=lambda item: (item[0], -item[1]))
    elif frame_confirm is None:
        strong = [item for item in pool if item[1] >= 0.5]
        ordered = sorted(strong or pool, key=lambda item: (item[0], -item[1]))
    else:
        strong = [item for item in pool if item[1] >= 0.75]
        ordered = sorted(strong, key=lambda item: (item[0], -item[1]))
        if not ordered:
            return None
    for slip, best_sc, best_off in ordered:
        gap_start = max(donor_start, speed * base_time + offset)
        gap_end = min(donor_end, speed * base_time + best_off)
        if gap_end - gap_start < min_surplus_seconds:
            continue
        if not _donor_gap_is_unrelated(
            base,
            donor,
            base_time=base_time,
            donor_gap_start=gap_start,
            donor_gap_end=gap_end,
            speed=speed,
            half_window=half_window,
            period=period,
            break_score=break_score,
        ):
            continue
        return gap_start, gap_end, best_off, best_sc
    return None


def _donor_gap_is_unrelated(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_gap_start: float,
    donor_gap_end: float,
    speed: float,
    half_window: float,
    period: float,
    break_score: float,
) -> bool:
    """Reject surplus spans that still track under the pre-jump offset.

    A false surplus skips real continuing content: the *old* map
    ``donor ≈ speed·base + offset`` still correlates through the interval.
    A real ad / junk block does not.
    """
    span = donor_gap_end - donor_gap_start
    if span <= 0:
        return False
    offset_old = donor_gap_start - speed * base_time
    # Probe the *interior* only — windows that still overlap pre/post content at
    # the gap edges look related even when the middle is a real ad.
    inner0 = donor_gap_start + half_window
    inner1 = donor_gap_end - half_window
    if inner1 <= inner0 + period:
        return True
    inner_span = inner1 - inner0
    # Dense probes — a single lucky zero must not green-light a false surplus.
    n_probes = max(5, int(inner_span / max(period, 2.0)))
    n_probes = min(n_probes, 24)
    base_lo = float(base.times[0]) + half_window
    base_hi = float(base.times[-1]) - half_window

    def _gap_stats(trial_speed: float, trial_offset: float) -> tuple[float, float] | None:
        scores: list[float] = []
        for i in range(1, n_probes + 1):
            donor_t = inner0 + inner_span * (i / (n_probes + 1))
            pred_base = (float(donor_t) - trial_offset) / max(trial_speed, 1e-6)
            if pred_base < base_lo or pred_base > base_hi:
                continue
            scores.append(
                local_curve_score(
                    base,
                    donor,
                    base_time=pred_base,
                    donor_time=float(donor_t),
                    speed=trial_speed,
                    half_window=half_window,
                    period=period,
                )
            )
        if not scores:
            return None
        mean_sc = float(np.mean(scores))
        hit_frac = sum(1 for s in scores if s >= break_score) / len(scores)
        return mean_sc, hit_frac

    def _looks_related(stats: tuple[float, float] | None) -> bool:
        if stats is None:
            return False
        mean_sc, hit_frac = stats
        return mean_sc >= break_score * 0.75 and hit_frac > 0.2

    # Current geometry first. Do *not* ±hop the offset: on sparse pulse trains
    # half-window overlap creates false "still related" hits that reject real
    # ads and push resume into the insert.
    if _looks_related(_gap_stats(speed, offset_old)):
        return False

    # Mild speed tweaks with the map pinned at the gap start: a wrong linear
    # speed makes continuing film look like a hole under ``offset_old``, which
    # would invent surplus ads. If any nearby speed restores tracking through
    # the span, it is not unrelated donor content.
    for delta in (-0.08, -0.04, -0.02, 0.02, 0.04, 0.08):
        trial_speed = round(float(speed) + delta, 4)
        if trial_speed < 0.85 or trial_speed > 1.2:
            continue
        trial_off = float(donor_gap_start) - trial_speed * float(base_time)
        if _looks_related(_gap_stats(trial_speed, trial_off)):
            return False
    return True


def _search_donor_resume(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_search_start: float,
    speed: float,
    half_window: float,
    period: float,
    min_score: float,
    max_ahead: float,
    hop: float,
) -> tuple[float, float] | None:
    """Search forward on the donor for the base curve at ``base_time``.

    Collects all adequate hits and prefers the *farthest* near-best resume so a
    short mid-ad coincidence cannot beat the real post-ad re-lock.
    """
    donor_end = float(donor.times[-1])
    t = max(float(donor_search_start), float(donor.times[0]))
    limit = min(donor_end, t + max_ahead)
    hits: list[tuple[float, float]] = []
    while t <= limit:
        score = local_curve_score(
            base,
            donor,
            base_time=base_time,
            donor_time=t,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        if score >= min_score:
            hits.append((float(t), float(score)))
        t += hop
    if not hits:
        return None
    best_score = max(score for _, score in hits)
    # Near-best scores only; among those prefer the farthest resume so a brief
    # mid-ad coincidence cannot beat the real post-ad re-lock.
    near = [
        (donor_t, score)
        for donor_t, score in hits
        if score >= best_score - 0.08 and score >= min_score
    ]
    if not near:
        near = hits
    donor_t, score = max(near, key=lambda item: (item[0], item[1]))
    return donor_t, score


def _find_forward_missing(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    break_base: float,
    break_donor: float,
    speed: float,
    offset: float,
    half_window: float,
    period: float,
    min_score: float,
    min_missing_seconds: float,
    max_ahead: float,
    confirm_seconds: float,
    hop_seconds: float,
    frame_confirm: Callable[[float, float], bool] | None = None,
) -> tuple[float, float, float] | None:
    """If a forward base jump restores the curve, return missing span + donor time.

    Returns ``(resume_base, resume_score, resume_donor)`` or ``None``.

    Breaks often fire a few seconds *before* the donor reaches post-cut content
    (half-window bleed). Probe a short forward donor window while searching the
    base, then prefer the earliest picture-supported resume (mirroring surplus).
    """
    hop = max(hop_seconds, period)
    base_lo = float(base.times[0]) + half_window
    base_hi = float(base.times[-1]) - half_window
    donor_lo = float(donor.times[0])
    donor_hi = float(donor.times[-1])
    search_start = max(base_lo, float(break_base) + float(min_missing_seconds))
    limit = min(base_hi, float(break_base) + float(max_ahead))
    if search_start > limit:
        return None

    # Donor may still be in pre-cut content at break_donor; look a little ahead.
    donor_span = max(float(confirm_seconds), float(half_window), 12.0)
    donor_candidates = [
        float(break_donor) + float(delta)
        for delta in np.arange(0.0, donor_span + 1e-9, 1.0)
        if donor_lo - 1e-6 <= float(break_donor) + float(delta) <= donor_hi + 1e-6
    ]
    if not donor_candidates:
        donor_candidates = [float(break_donor)]

    confirmed: list[tuple[float, float, float]] = []  # base_t, score, donor_t
    base_step = 1.0
    for base_t in np.arange(search_start, limit + 1e-9, base_step):
        best_sc = -1.0
        best_d = float(break_donor)
        for d_t in donor_candidates:
            sc = local_curve_score(
                base,
                donor,
                base_time=float(base_t),
                donor_time=float(d_t),
                speed=speed,
                half_window=half_window,
                period=period,
            )
            if sc < min_score:
                continue
            trial_off = float(d_t) - speed * float(base_t)
            trial_lock = CurveLock(
                speed=speed,
                offset_seconds=trial_off,
                base_time=float(base_t),
                donor_time=float(d_t),
                score=float(sc),
            )
            if not _confirm_lock(
                base,
                donor,
                trial_lock,
                half_window=half_window,
                period=period,
                hop=hop,
                min_score=min_score,
                confirm_seconds=min(confirm_seconds, 6.0),
                frame_confirm=None,
            ):
                continue
            if sc > best_sc:
                best_sc = float(sc)
                best_d = float(d_t)
        if best_sc >= min_score:
            confirmed.append((float(base_t), best_sc, best_d))

    if not confirmed:
        # With picture available, do not invent resumes from the coarse legacy
        # nudge — that path returns weak scores and derails cross-encode walks.
        if frame_confirm is not None:
            return None
        legacy = _search_base_resume(
            base,
            donor,
            donor_time=float(break_donor),
            base_search_start=search_start,
            speed=speed,
            offset=offset,
            half_window=half_window,
            period=period,
            min_score=min_score,
            max_ahead=max_ahead,
            hop=hop,
        )
        if legacy is None:
            return None
        resume_base, resume_score = legacy
        gap = resume_base - break_base
        if gap < min_missing_seconds:
            return None
        if not _missing_gap_credible(gap, resume_score, picture_backed=False):
            return None
        return resume_base, resume_score, float(break_donor)

    def _frames_ok(base_t: float, donor_t: float) -> bool:
        if frame_confirm is None:
            return False
        # Probe a short window with constant post-gap offset.
        trial_off = float(donor_t) - speed * float(base_t)
        for delta in range(0, int(max(confirm_seconds, 1.0)) + 1):
            anchor = float(base_t) + float(delta)
            if frame_confirm(anchor, speed * anchor + trial_off):
                return True
        return False

    # Frame hits still need a real activity score — soft pHash alone invents
    # short missing gaps on grade-mismatched pairs.
    framed = [
        c for c in confirmed if c[1] >= 0.5 and _frames_ok(c[0], c[2])
    ]
    strong = [c for c in confirmed if c[1] >= 0.5]
    if framed:
        ordered = sorted(framed, key=lambda item: (item[0], -item[1]))
    elif frame_confirm is not None:
        # Picture bank present but no frame-backed resume: refuse activity-only
        # long jumps (self-similar scenes elsewhere in the film).
        return None
    else:
        ordered = sorted(strong or confirmed, key=lambda item: (item[0], -item[1]))

    picture_backed = frame_confirm is not None
    for resume_base, resume_score, resume_donor in ordered:
        gap = resume_base - break_base
        if gap < min_missing_seconds:
            continue
        if not _missing_gap_credible(
            gap, resume_score, picture_backed=picture_backed
        ):
            continue
        return resume_base, resume_score, resume_donor
    return None


def _missing_gap_credible(
    gap: float,
    resume_score: float,
    *,
    picture_backed: bool,
) -> bool:
    """Reject short/weak missing-base islands that are usually cross-encode jitter.

    Real cuts in this pipeline are tens of seconds (synthetic ~30s). Brief
    framed agreements at 6–15s routinely appear on grade-mismatched film and
    should not outrank continuing the walk / surplus / speed refine. Activity-
    only fixtures without frames keep a lower floor so unit pulses still work.
    """
    if picture_backed:
        # With frames, short picture hits are usually grade-mismatch aliases.
        # Require a substantial cut (≥20s); mid gaps still need a solid score.
        # Long cuts (synth ~30s+) may be weaker on activity and rely on frames.
        if gap < 20.0:
            return False
        if gap < 30.0 and resume_score < 0.65:
            return False
        return True
    if gap < 12.0 and resume_score < 0.65:
        return False
    if gap < 20.0 and resume_score < 0.55:
        return False
    return True


def _search_base_resume(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    donor_time: float,
    base_search_start: float,
    speed: float,
    offset: float,
    half_window: float,
    period: float,
    min_score: float,
    max_ahead: float,
    hop: float,
) -> tuple[float, float] | None:
    """Advance on the base until it matches again near ``donor_time``."""
    base_end = float(base.times[-1])
    t = max(float(base_search_start), float(base.times[0]))
    limit = min(base_end, t + max_ahead)
    # Allow a small donor nudge while searching. First adequate hit wins —
    # preferring farthest near-best invents huge missing-base spans on sparse
    # pulse trains and steals ranking from real surplus / speed refine.
    for base_t in np.arange(t, limit + hop * 0.5, hop):
        for delta in (-2 * hop, 0.0, 2 * hop):
            d_t = donor_time + delta
            score = local_curve_score(
                base,
                donor,
                base_time=float(base_t),
                donor_time=d_t,
                speed=speed,
                half_window=half_window,
                period=period,
            )
            if score >= min_score:
                return float(base_t), float(score)
        pred = speed * float(base_t) + offset
        score_pred = local_curve_score(
            base,
            donor,
            base_time=float(base_t),
            donor_time=pred,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        if score_pred >= min_score:
            return float(base_t), float(score_pred)
    return None
