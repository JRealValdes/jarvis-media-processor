"""Dense residual lag field: reconstructed donor vs base track 0.

Builds a slowly varying lag ``L(t)`` from band-limited onset envelopes so
cross-language dialogue energy does not dominate. Every analysis step gets a
lag + confidence; low-confidence regions fall back to a robust global prior
instead of snapping to zero (which caused corrected/uncorrected islands).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from jarvis_media_processor.video.activity import normalize_activity

_WARP_BLOCK = 1_000_000


@dataclass(frozen=True)
class LagKnot:
    """Lag sample on the shared base timeline (dense grid)."""

    time_seconds: float
    lag_seconds: float
    score: float
    improvement: float
    confidence: float = 0.0


@dataclass
class FineWarpPlan:
    """Lag-field plan plus diagnostics."""

    knots: list[LagKnot] = field(default_factory=list)
    global_lag_seconds: float = 0.0
    skipped_notes: list[str] = field(default_factory=list)


def _to_mono(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2:
        return arr.mean(axis=0)
    return arr.reshape(-1)


def loudness_normalize(samples: NDArray[np.floating], *, eps: float = 1e-8) -> NDArray[np.float64]:
    """Peak-RMS normalize a mono clip for fairer cross-track comparison."""
    mono = _to_mono(samples)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    if rms < eps:
        return mono
    return mono * (0.1 / rms)


def bandpass_filter(
    samples: NDArray[np.floating],
    sample_rate: int,
    *,
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
    order: int = 2,
) -> NDArray[np.float64]:
    """Butterworth band-pass to emphasize shared FX/attacks over dialogue body."""
    mono = _to_mono(samples)
    if mono.size < 16 or sample_rate <= 0:
        return mono
    nyq = 0.5 * float(sample_rate)
    lo = max(1.0, float(low_hz)) / nyq
    hi = min(float(high_hz), nyq * 0.95) / nyq
    if not (0.0 < lo < hi < 1.0):
        return mono
    b, a = butter(order, [lo, hi], btype="band")
    return np.asarray(filtfilt(b, a, mono), dtype=np.float64)


def onset_envelope(
    samples: NDArray[np.floating],
    sample_rate: int,
    *,
    hop_seconds: float = 0.01,
    frame_seconds: float = 0.04,
) -> tuple[NDArray[np.float64], float]:
    """Half-wave-rectified log-energy flux (onset strength) of a mono signal."""
    mono = _to_mono(samples)
    if mono.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float64), hop_seconds
    hop = max(1, int(round(sample_rate * hop_seconds)))
    frame = max(hop, int(round(sample_rate * frame_seconds)))
    if mono.size < frame:
        rms = float(np.sqrt(np.mean(np.square(mono))))
        return np.asarray([rms], dtype=np.float64), hop / sample_rate

    window = np.hanning(frame)
    n_hops = 1 + (mono.size - frame) // hop
    energy = np.empty(n_hops, dtype=np.float64)
    for i in range(n_hops):
        start = i * hop
        frame_samples = mono[start : start + frame] * window
        energy[i] = float(np.mean(np.square(frame_samples)))
    log_e = np.log1p(energy * 1e4)
    flux = np.diff(log_e, prepend=log_e[0])
    onset = np.maximum(flux, 0.0)
    return onset, hop / sample_rate


def comparable_onset_envelope(
    samples: NDArray[np.floating],
    sample_rate: int,
    *,
    hop_seconds: float = 0.01,
    frame_seconds: float = 0.04,
    low_hz: float = 300.0,
    high_hz: float = 6000.0,
) -> tuple[NDArray[np.float64], float]:
    """Loudness-normalize, band-limit, then build an onset envelope for lag search."""
    normed = loudness_normalize(samples)
    filtered = bandpass_filter(normed, sample_rate, low_hz=low_hz, high_hz=high_hz)
    return onset_envelope(
        filtered,
        sample_rate,
        hop_seconds=hop_seconds,
        frame_seconds=frame_seconds,
    )


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


def context_improvement(
    base_env: NDArray[np.floating],
    recon_env: NDArray[np.floating],
    *,
    hop: float,
    center_seconds: float,
    proposed_lag: float,
    context_window_seconds: float,
) -> float:
    """Wider-window correlation gain of ``proposed_lag`` vs lag=0 (big-picture check)."""
    with_lag = _window_corr(
        base_env,
        recon_env,
        hop=hop,
        center_seconds=center_seconds,
        lag_seconds=proposed_lag,
        window_seconds=context_window_seconds,
    )
    with_zero = _window_corr(
        base_env,
        recon_env,
        hop=hop,
        center_seconds=center_seconds,
        lag_seconds=0.0,
        window_seconds=context_window_seconds,
    )
    return float(with_lag - with_zero)


def _weighted_median(values: NDArray[np.floating], weights: NDArray[np.floating]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if vals.size == 0:
        return 0.0
    if float(np.sum(w)) <= 0:
        return float(np.median(vals))
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    cutoff = 0.5 * float(cdf[-1])
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(max(idx, 0), vals.size - 1)
    return float(vals[idx])


def gate_lag_with_confidence(
    smoothed_lags: NDArray[np.floating],
    confidence: NDArray[np.floating],
    *,
    global_lag: float,
    confidence_tau: float = 0.08,
) -> NDArray[np.float64]:
    """Pull low-confidence steps back to the global prior after smoothing.

    Neighbor-weighted smooth alone re-injects strong lags into weak windows
    (smooth bleed). Own-confidence gating prevents that over-correction.
    """
    sm = np.asarray(smoothed_lags, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    tau = max(float(confidence_tau), 1e-9)
    trust = conf / (conf + tau)
    return trust * sm + (1.0 - trust) * float(global_lag)


def estimate_dense_lag_field(
    base_env: NDArray[np.floating],
    recon_env: NDArray[np.floating],
    *,
    hop: float,
    duration_seconds: float,
    window_seconds: float = 4.0,
    step_seconds: float = 2.0,
    max_lag_seconds: float = 1.0,
    context_window_seconds: float = 10.0,
    prior_weight: float = 0.15,
    smooth_seconds: float = 30.0,
    confidence_tau: float = 0.08,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[LagKnot], float, list[str]]:
    """Dense lag grid blended with a global prior and temporally smoothed.

    Returns ``(times, lags, knots, global_lag, notes)``.
    """
    notes: list[str] = []
    if hop <= 0 or duration_seconds <= 0 or step_seconds <= 0:
        return (
            np.zeros(0),
            np.zeros(0),
            [],
            0.0,
            ["Invalid envelope timing for lag-field estimation."],
        )

    pad = 0.5 * max(window_seconds, context_window_seconds) + max_lag_seconds
    t0 = pad
    t1 = duration_seconds - pad
    if t1 <= t0:
        return np.zeros(0), np.zeros(0), [], 0.0, ["Timeline too short for fine-warp."]

    centers = np.arange(t0, t1 + 0.5 * step_seconds, step_seconds, dtype=np.float64)
    edge_eps = max(hop * 0.75, 1e-3)
    raw_lags = np.zeros(centers.size, dtype=np.float64)
    conf = np.zeros(centers.size, dtype=np.float64)
    scores = np.zeros(centers.size, dtype=np.float64)
    improves = np.zeros(centers.size, dtype=np.float64)

    for i, center in enumerate(centers):
        lag, score, zero_score = estimate_local_lag(
            base_env,
            recon_env,
            hop=hop,
            center_seconds=float(center),
            window_seconds=window_seconds,
            max_lag_seconds=max_lag_seconds,
        )
        margin = float(score - zero_score)
        ctx = context_improvement(
            base_env,
            recon_env,
            hop=hop,
            center_seconds=float(center),
            proposed_lag=lag,
            context_window_seconds=context_window_seconds,
        )
        scores[i] = score
        improves[i] = ctx
        raw_lags[i] = lag
        c = max(0.0, margin) * max(0.0, ctx)
        # Distrust saturations at the search edge (often wrong cross-lang peaks).
        if abs(lag) >= max_lag_seconds - edge_eps:
            c *= 0.2
        conf[i] = c

    # Global prior from confident windows only.
    confident = conf >= max(1e-6, float(np.percentile(conf, 70)) if conf.size else 1.0)
    if np.any(confident) and float(np.sum(conf[confident])) > 0:
        global_lag = _weighted_median(raw_lags[confident], conf[confident])
    elif conf.size:
        global_lag = _weighted_median(raw_lags, np.maximum(conf, 1e-6))
    else:
        global_lag = 0.0
    notes.append(f"Global lag prior={global_lag:+.3f}s from dense onset field.")

    # Pre-blend locals toward prior, then confidence-weighted temporal smooth.
    pre = np.empty_like(raw_lags)
    for i in range(raw_lags.size):
        c = float(conf[i])
        pre[i] = (c * raw_lags[i] + prior_weight * global_lag) / (c + prior_weight)

    radius = max(0, int(round(smooth_seconds / step_seconds)))
    smoothed = pre.copy()
    if radius > 0 and pre.size >= 3:
        for i in range(pre.size):
            lo = max(0, i - radius)
            hi = min(pre.size, i + radius + 1)
            w = conf[lo:hi]
            v = pre[lo:hi]
            if float(np.sum(w)) > 1e-9:
                smoothed[i] = float(np.average(v, weights=np.maximum(w, 1e-6)))
            else:
                smoothed[i] = float(global_lag)

    # Critical: neighbor smooth can still paint strong lag onto conf≈0 windows.
    # Re-gate with each step's own confidence so weak regions fall back to prior.
    blended = gate_lag_with_confidence(
        smoothed,
        conf,
        global_lag=global_lag,
        confidence_tau=confidence_tau,
    )
    notes.append(
        f"Confidence gate tau={confidence_tau:.3f} "
        f"(low-conf steps → global prior after smooth)."
    )

    knots = [
        LagKnot(
            time_seconds=float(centers[i]),
            lag_seconds=float(blended[i]),
            score=float(scores[i]),
            improvement=float(improves[i]),
            confidence=float(conf[i]),
        )
        for i in range(centers.size)
    ]
    return centers, blended, knots, float(global_lag), notes


def lag_curve_from_dense(
    times: NDArray[np.floating],
    lags: NDArray[np.floating],
    *,
    duration_seconds: float,
    sample_rate: int,
) -> NDArray[np.float64]:
    """Interpolate a dense step grid onto every audio sample (no zero gaps)."""
    n = max(1, int(round(duration_seconds * sample_rate)))
    if times.size == 0:
        return np.zeros(n, dtype=np.float64)
    sample_t = np.arange(n, dtype=np.float64) / float(sample_rate)
    # Hold endpoints outside the estimated range.
    return np.interp(
        sample_t,
        np.asarray(times, dtype=np.float64),
        np.asarray(lags, dtype=np.float64),
        left=float(lags[0]),
        right=float(lags[-1]),
    )


def apply_lag_curve(
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    lag_curve: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Resample reconstructed audio with ``out[t] = recon[t + L(t)]``."""
    recon = np.asarray(recon_samples, dtype=np.float64)
    mono = recon.ndim == 1
    recon_2d = recon[np.newaxis, :] if mono else recon
    channels, n = int(recon_2d.shape[0]), int(recon_2d.shape[-1])
    if lag_curve.size != n:
        curve = np.zeros(n, dtype=np.float64)
        m = min(n, int(lag_curve.size))
        curve[:m] = np.asarray(lag_curve[:m], dtype=np.float64)
    else:
        curve = np.asarray(lag_curve, dtype=np.float64)

    if float(np.max(np.abs(curve))) < 1e-9:
        return np.asarray(recon_samples)

    out = np.empty((channels, n), dtype=np.float64)
    sample_t = np.arange(n, dtype=np.float64) / float(sample_rate)
    src_index = np.clip((sample_t + curve) * sample_rate, 0.0, float(n - 1))

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
    max_lag_seconds: float = 1.0,
    window_seconds: float = 4.0,
    step_seconds: float = 2.0,
    context_window_seconds: float = 10.0,
    prior_weight: float = 0.15,
    smooth_seconds: float = 30.0,
    confidence_tau: float = 0.08,
    hop_seconds: float = 0.01,
    frame_seconds: float = 0.04,
    band_low_hz: float = 300.0,
    band_high_hz: float = 6000.0,
) -> tuple[NDArray[np.float64], FineWarpPlan]:
    """Estimate a dense lag curve from comparable onset envelopes."""
    base_mono = _to_mono(base_samples)
    recon_arr = np.asarray(recon_samples, dtype=np.float64)
    recon_mono = _to_mono(recon_arr)
    n = min(base_mono.size, recon_mono.size)
    base_mono = base_mono[:n]
    recon_mono = recon_mono[:n]
    duration = float(n / sample_rate) if sample_rate else 0.0

    base_env, hop = comparable_onset_envelope(
        base_mono,
        sample_rate,
        hop_seconds=hop_seconds,
        frame_seconds=frame_seconds,
        low_hz=band_low_hz,
        high_hz=band_high_hz,
    )
    recon_env, _ = comparable_onset_envelope(
        recon_mono,
        sample_rate,
        hop_seconds=hop_seconds,
        frame_seconds=frame_seconds,
        low_hz=band_low_hz,
        high_hz=band_high_hz,
    )
    hop = float(hop) if hop > 0 else hop_seconds

    times, lags, knots, global_lag, notes = estimate_dense_lag_field(
        base_env,
        recon_env,
        hop=hop,
        duration_seconds=duration,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        max_lag_seconds=max_lag_seconds,
        context_window_seconds=context_window_seconds,
        prior_weight=prior_weight,
        smooth_seconds=smooth_seconds,
        confidence_tau=confidence_tau,
    )
    plan = FineWarpPlan(
        knots=knots,
        global_lag_seconds=global_lag,
        skipped_notes=notes,
    )
    curve = lag_curve_from_dense(
        times,
        lags,
        duration_seconds=duration,
        sample_rate=sample_rate,
    )
    return curve, plan


def plan_and_apply_fine_warp(
    base_samples: NDArray[np.floating],
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    *,
    max_lag_seconds: float = 1.0,
    window_seconds: float = 4.0,
    step_seconds: float = 2.0,
    context_window_seconds: float = 10.0,
    prior_weight: float = 0.15,
    smooth_seconds: float = 30.0,
    confidence_tau: float = 0.08,
    band_low_hz: float = 300.0,
    band_high_hz: float = 6000.0,
) -> tuple[NDArray[np.floating], FineWarpPlan]:
    """Build a dense onset-based lag field and apply it to reconstructed audio."""
    recon_arr = np.asarray(recon_samples, dtype=np.float64)
    base_mono = _to_mono(base_samples)
    n = min(
        base_mono.size,
        recon_arr.shape[-1] if recon_arr.ndim >= 1 else 0,
    )
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
        context_window_seconds=context_window_seconds,
        prior_weight=prior_weight,
        smooth_seconds=smooth_seconds,
        confidence_tau=confidence_tau,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
    )
    warped = apply_lag_curve(recon_work, sample_rate, curve)
    return warped, plan
