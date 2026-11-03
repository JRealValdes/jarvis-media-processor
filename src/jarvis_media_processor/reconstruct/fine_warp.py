"""Windowed fine lag field: reconstructed donor vs base track 0.

Estimates a slowly varying lag ``L(t)`` from energy-envelope correlation in
sliding windows. A candidate lag is kept only when it beats lag=0 locally
*and* improves correlation across neighboring windows. Uncertain regions stay
at identity — no landmark snapping, no multi-minute rubber-banding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jarvis_media_processor.audio.landmarks import rms_envelope
from jarvis_media_processor.video.activity import normalize_activity

_WARP_BLOCK = 1_000_000


@dataclass(frozen=True)
class LagKnot:
    """Trusted lag sample on the shared base timeline."""

    time_seconds: float
    lag_seconds: float
    score: float
    improvement: float


@dataclass
class FineWarpPlan:
    """Lag-field plan plus diagnostics."""

    knots: list[LagKnot] = field(default_factory=list)
    skipped_notes: list[str] = field(default_factory=list)


def _window_corr(
    base_env: NDArray[np.floating],
    recon_env: NDArray[np.floating],
    *,
    hop: float,
    center_seconds: float,
    lag_seconds: float,
    window_seconds: float,
) -> float:
    """Pearson-like correlation of envelopes in a window at a fixed lag."""
    if hop <= 0 or window_seconds <= 0:
        return 0.0
    half = 0.5 * window_seconds
    b0 = int(round((center_seconds - half) / hop))
    b1 = int(round((center_seconds + half) / hop)) + 1
    r0 = int(round((center_seconds - half + lag_seconds) / hop))
    r1 = r0 + (b1 - b0)
    if b0 < 0 or b1 > int(base_env.size) or r0 < 0 or r1 > int(recon_env.size):
        return 0.0
    if b1 - b0 < 4:
        return 0.0
    template = normalize_activity(np.asarray(base_env[b0:b1], dtype=np.float64))
    cand = normalize_activity(np.asarray(recon_env[r0:r1], dtype=np.float64))
    if template.size != cand.size or float(np.max(np.abs(template))) < 1e-12:
        return 0.0
    if float(np.max(np.abs(cand))) < 1e-12:
        return 0.0
    return float(np.mean(template * cand))


def estimate_local_lag(
    base_env: NDArray[np.floating],
    recon_env: NDArray[np.floating],
    *,
    hop: float,
    center_seconds: float,
    window_seconds: float,
    max_lag_seconds: float,
) -> tuple[float, float, float]:
    """Return ``(best_lag, best_score, zero_lag_score)`` for one analysis window."""
    if hop <= 0:
        return 0.0, 0.0, 0.0
    max_lag_hops = max(1, int(round(max_lag_seconds / hop)))
    best_lag = 0.0
    best_score = -1.0
    zero_score = _window_corr(
        base_env,
        recon_env,
        hop=hop,
        center_seconds=center_seconds,
        lag_seconds=0.0,
        window_seconds=window_seconds,
    )
    for lag_h in range(-max_lag_hops, max_lag_hops + 1):
        lag = lag_h * hop
        score = _window_corr(
            base_env,
            recon_env,
            hop=hop,
            center_seconds=center_seconds,
            lag_seconds=lag,
            window_seconds=window_seconds,
        )
        if score > best_score:
            best_score = score
            best_lag = lag
    if best_score < 0:
        return 0.0, 0.0, float(zero_score)
    return float(best_lag), float(best_score), float(zero_score)


def neighborhood_improvement(
    base_env: NDArray[np.floating],
    recon_env: NDArray[np.floating],
    *,
    hop: float,
    center_seconds: float,
    proposed_lag: float,
    window_seconds: float,
    neighbor_radius_seconds: float,
    neighbor_step_seconds: float,
) -> float:
    """Mean neighbor-window correlation at ``proposed_lag`` minus the same at lag=0."""
    if neighbor_step_seconds <= 0:
        neighbor_step_seconds = window_seconds
    starts = center_seconds - neighbor_radius_seconds
    ends = center_seconds + neighbor_radius_seconds
    times = np.arange(starts, ends + 0.5 * neighbor_step_seconds, neighbor_step_seconds)
    if times.size == 0:
        times = np.asarray([center_seconds], dtype=np.float64)
    with_lag: list[float] = []
    with_zero: list[float] = []
    for t in times:
        with_lag.append(
            _window_corr(
                base_env,
                recon_env,
                hop=hop,
                center_seconds=float(t),
                lag_seconds=proposed_lag,
                window_seconds=window_seconds,
            )
        )
        with_zero.append(
            _window_corr(
                base_env,
                recon_env,
                hop=hop,
                center_seconds=float(t),
                lag_seconds=0.0,
                window_seconds=window_seconds,
            )
        )
    return float(np.mean(with_lag) - np.mean(with_zero))


def estimate_lag_knots(
    base_env: NDArray[np.floating],
    recon_env: NDArray[np.floating],
    *,
    hop: float,
    duration_seconds: float,
    window_seconds: float = 3.0,
    step_seconds: float = 1.0,
    max_lag_seconds: float = 0.15,
    min_score: float = 0.35,
    min_score_margin: float = 0.05,
    min_improvement: float = 0.02,
    neighbor_radius_seconds: float = 4.0,
    neighbor_step_seconds: float = 1.0,
) -> tuple[list[LagKnot], list[str]]:
    """Slide analysis windows; keep lags that improve the local neighborhood."""
    notes: list[str] = []
    if hop <= 0 or duration_seconds <= 0 or step_seconds <= 0:
        return [], ["Invalid envelope timing for lag-field estimation."]

    half = 0.5 * window_seconds
    # Stay inside envelopes for the analysis window plus max lag pad.
    pad = half + max_lag_seconds + neighbor_radius_seconds
    t0 = pad
    t1 = duration_seconds - pad
    if t1 <= t0:
        return [], ["Timeline too short for windowed fine-warp."]

    centers = np.arange(t0, t1 + 0.5 * step_seconds, step_seconds, dtype=np.float64)
    edge_eps = max(hop * 0.75, 1e-3)
    knots: list[LagKnot] = []
    rejected = 0
    for center in centers:
        lag, score, zero_score = estimate_local_lag(
            base_env,
            recon_env,
            hop=hop,
            center_seconds=float(center),
            window_seconds=window_seconds,
            max_lag_seconds=max_lag_seconds,
        )
        if score < min_score or (score - zero_score) < min_score_margin:
            rejected += 1
            continue
        if abs(lag) <= edge_eps:
            continue
        if abs(lag) > max_lag_seconds - edge_eps:
            rejected += 1
            continue
        improve = neighborhood_improvement(
            base_env,
            recon_env,
            hop=hop,
            center_seconds=float(center),
            proposed_lag=lag,
            window_seconds=window_seconds,
            neighbor_radius_seconds=neighbor_radius_seconds,
            neighbor_step_seconds=neighbor_step_seconds,
        )
        if improve < min_improvement:
            rejected += 1
            continue
        knots.append(
            LagKnot(
                time_seconds=float(center),
                lag_seconds=float(lag),
                score=float(score),
                improvement=float(improve),
            )
        )
    if rejected:
        notes.append(
            f"Rejected {rejected} windows (weak score, no margin, edge lag, "
            "or neighborhood did not improve)."
        )
    if not knots:
        notes.append("No lag knots accepted; leaving reconstruct unchanged.")
    return knots, notes


def smooth_lag_knots(
    knots: list[LagKnot],
    *,
    max_lag_jump_seconds: float = 0.04,
    median_radius: int = 1,
) -> tuple[list[LagKnot], list[str]]:
    """Median-filter accepted lags and drop residual jumps vs neighbors."""
    if len(knots) < 2:
        return list(knots), []
    ordered = sorted(knots, key=lambda k: k.time_seconds)
    lags = np.asarray([k.lag_seconds for k in ordered], dtype=np.float64)
    if median_radius > 0 and lags.size >= 3:
        smoothed = lags.copy()
        for i in range(lags.size):
            lo = max(0, i - median_radius)
            hi = min(lags.size, i + median_radius + 1)
            smoothed[i] = float(np.median(lags[lo:hi]))
        lags = smoothed

    notes: list[str] = []
    kept: list[LagKnot] = [
        LagKnot(
            time_seconds=ordered[0].time_seconds,
            lag_seconds=float(lags[0]),
            score=ordered[0].score,
            improvement=ordered[0].improvement,
        )
    ]
    for i in range(1, len(ordered)):
        prev = kept[-1]
        lag = float(lags[i])
        if abs(lag - prev.lag_seconds) > max_lag_jump_seconds:
            notes.append(
                f"Drop lag jump at {ordered[i].time_seconds:.1f}s "
                f"({prev.lag_seconds:+.3f} → {lag:+.3f})."
            )
            continue
        kept.append(
            LagKnot(
                time_seconds=ordered[i].time_seconds,
                lag_seconds=lag,
                score=ordered[i].score,
                improvement=ordered[i].improvement,
            )
        )
    return kept, notes


def lag_curve_from_knots(
    knots: list[LagKnot],
    *,
    duration_seconds: float,
    sample_rate: int,
    interp_gap_seconds: float = 3.0,
) -> NDArray[np.float64]:
    """Per-sample lag curve; identity (0) outside trusted interpolations."""
    n = max(1, int(round(duration_seconds * sample_rate)))
    curve = np.zeros(n, dtype=np.float64)
    if len(knots) == 0:
        return curve
    ordered = sorted(knots, key=lambda k: k.time_seconds)
    times = np.asarray([k.time_seconds for k in ordered], dtype=np.float64)
    lags = np.asarray([k.lag_seconds for k in ordered], dtype=np.float64)
    sample_t = np.arange(n, dtype=np.float64) / float(sample_rate)

    # Interpolate only inside runs where consecutive knots are close enough.
    i = 0
    while i < len(ordered):
        j = i
        while (
            j + 1 < len(ordered)
            and (times[j + 1] - times[j]) <= interp_gap_seconds + 1e-9
        ):
            j += 1
        if j > i:
            t0 = float(times[i])
            t1 = float(times[j])
            mask = (sample_t >= t0) & (sample_t <= t1)
            if np.any(mask):
                curve[mask] = np.interp(sample_t[mask], times[i : j + 1], lags[i : j + 1])
        elif interp_gap_seconds > 0:
            # Isolated knot: apply constant lag in a short island around it.
            half = 0.5 * min(interp_gap_seconds, 1.0)
            mask = np.abs(sample_t - times[i]) <= half
            curve[mask] = lags[i]
        i = j + 1
    return curve


def apply_lag_curve(
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    lag_curve: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Resample reconstructed audio with ``out[t] = recon[t + L(t)]``."""
    recon = np.asarray(recon_samples, dtype=np.float64)
    mono = recon.ndim == 1
    if mono:
        recon_2d = recon[np.newaxis, :]
    else:
        recon_2d = recon
    channels, n = int(recon_2d.shape[0]), int(recon_2d.shape[-1])
    if lag_curve.size != n:
        # Trim/pad lag curve to audio length.
        curve = np.zeros(n, dtype=np.float64)
        m = min(n, int(lag_curve.size))
        curve[:m] = np.asarray(lag_curve[:m], dtype=np.float64)
    else:
        curve = np.asarray(lag_curve, dtype=np.float64)

    if float(np.max(np.abs(curve))) < 1e-9:
        return np.asarray(recon_samples)

    out = np.empty((channels, n), dtype=np.float64)
    sample_t = np.arange(n, dtype=np.float64) / float(sample_rate)
    src_t = sample_t + curve
    src_index = np.clip(src_t * sample_rate, 0.0, float(n - 1))

    for start in range(0, n, _WARP_BLOCK):
        end = min(n, start + _WARP_BLOCK)
        pos = src_index[start:end]
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        frac = pos - i0.astype(np.float64)
        for c in range(channels):
            out[c, start:end] = recon_2d[c, i0] * (1.0 - frac) + recon_2d[c, i1] * frac

    if mono:
        return out[0]
    return out


def plan_lag_field(
    base_samples: NDArray[np.floating],
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    *,
    max_lag_seconds: float = 0.15,
    window_seconds: float = 3.0,
    step_seconds: float = 1.0,
    min_score: float = 0.35,
    min_score_margin: float = 0.05,
    min_improvement: float = 0.02,
    neighbor_radius_seconds: float = 4.0,
    neighbor_step_seconds: float = 1.0,
    max_lag_jump_seconds: float = 0.04,
    interp_gap_seconds: float = 3.0,
    hop_seconds: float = 0.01,
    frame_seconds: float = 0.04,
) -> tuple[NDArray[np.float64], FineWarpPlan]:
    """Estimate a lag curve from mono mixes; return curve + plan (not yet applied)."""
    base_mono = np.asarray(base_samples, dtype=np.float64)
    if base_mono.ndim == 2:
        base_mono = base_mono.mean(axis=0)
    recon_arr = np.asarray(recon_samples, dtype=np.float64)
    recon_mono = recon_arr.mean(axis=0) if recon_arr.ndim == 2 else recon_arr
    n = min(base_mono.size, recon_mono.size)
    base_mono = base_mono[:n]
    recon_mono = recon_mono[:n]
    duration = float(n / sample_rate) if sample_rate else 0.0

    base_env, hop = rms_envelope(
        base_mono, sample_rate, hop_seconds=hop_seconds, frame_seconds=frame_seconds
    )
    recon_env, _ = rms_envelope(
        recon_mono, sample_rate, hop_seconds=hop_seconds, frame_seconds=frame_seconds
    )
    hop = float(hop) if hop > 0 else hop_seconds

    raw_knots, notes = estimate_lag_knots(
        base_env,
        recon_env,
        hop=hop,
        duration_seconds=duration,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        max_lag_seconds=max_lag_seconds,
        min_score=min_score,
        min_score_margin=min_score_margin,
        min_improvement=min_improvement,
        neighbor_radius_seconds=neighbor_radius_seconds,
        neighbor_step_seconds=neighbor_step_seconds,
    )
    knots, smooth_notes = smooth_lag_knots(
        raw_knots,
        max_lag_jump_seconds=max_lag_jump_seconds,
    )
    plan = FineWarpPlan(knots=knots, skipped_notes=[*notes, *smooth_notes])
    curve = lag_curve_from_knots(
        knots,
        duration_seconds=duration,
        sample_rate=sample_rate,
        interp_gap_seconds=interp_gap_seconds,
    )
    return curve, plan


def plan_and_apply_fine_warp(
    base_samples: NDArray[np.floating],
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    *,
    max_lag_seconds: float = 0.15,
    window_seconds: float = 3.0,
    step_seconds: float = 1.0,
    min_score: float = 0.35,
    min_score_margin: float = 0.05,
    min_improvement: float = 0.02,
    neighbor_radius_seconds: float = 4.0,
    neighbor_step_seconds: float = 1.0,
    max_lag_jump_seconds: float = 0.04,
    interp_gap_seconds: float = 3.0,
) -> tuple[NDArray[np.floating], FineWarpPlan]:
    """Build a neighborhood-validated lag field and apply it to reconstructed audio."""
    recon_arr = np.asarray(recon_samples, dtype=np.float64)
    base_mono = np.asarray(base_samples, dtype=np.float64)
    if base_mono.ndim == 2:
        base_mono = base_mono.mean(axis=0)
    recon_mono = recon_arr.mean(axis=0) if recon_arr.ndim == 2 else recon_arr
    n = min(base_mono.size, recon_mono.size if recon_arr.ndim == 1 else recon_arr.shape[-1])
    if recon_arr.ndim == 2:
        recon_work = recon_arr[:, :n]
    else:
        recon_work = recon_arr[:n]

    curve, plan = plan_lag_field(
        base_mono[:n],
        recon_work,
        sample_rate,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        min_score=min_score,
        min_score_margin=min_score_margin,
        min_improvement=min_improvement,
        neighbor_radius_seconds=neighbor_radius_seconds,
        neighbor_step_seconds=neighbor_step_seconds,
        max_lag_jump_seconds=max_lag_jump_seconds,
        interp_gap_seconds=interp_gap_seconds,
    )
    warped = apply_lag_curve(recon_work, sample_rate, curve)
    return warped, plan
