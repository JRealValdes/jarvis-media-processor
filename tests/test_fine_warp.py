"""Unit tests for dense post-reconstruct fine-warp."""

from __future__ import annotations

import numpy as np

from jarvis_media_processor.reconstruct.fine_warp import (
    apply_lag_curve,
    comparable_onset_envelope,
    estimate_local_lag,
    interpolate_from_pillars,
    lag_curve_from_dense,
    plan_and_apply_fine_warp,
    select_lag_pillars,
)


def _bursts(
    sample_rate: int,
    times: list[float],
    *,
    duration: float,
    width: float = 0.02,
    amp: float = 0.9,
) -> np.ndarray:
    """Transient clicks (broadband) so band-limited onset envelopes lock."""
    n = int(round(duration * sample_rate))
    out = np.zeros(n, dtype=np.float64)
    half = max(1, int(round(width * sample_rate)))
    for t in times:
        center = int(round(t * sample_rate))
        a = max(0, center - half // 2)
        b = min(n, a + half)
        if b > a:
            # decaying click → high-frequency onset energy
            ramp = np.linspace(1.0, 0.0, b - a, dtype=np.float64)
            out[a:b] = amp * ramp * ((-1.0) ** np.arange(b - a))
    return out


def _shifted(signal: np.ndarray, sample_rate: int, lag_seconds: float) -> np.ndarray:
    """Positive lag ⇒ content moves later (pad leading silence)."""
    shift = int(round(lag_seconds * sample_rate))
    out = np.zeros_like(signal)
    if shift >= 0:
        if shift < signal.size:
            out[shift:] = signal[: signal.size - shift]
    else:
        out[: signal.size + shift] = signal[-shift:]
    return out


def test_comparable_onset_envelope_nonempty() -> None:
    sr = 8_000
    x = _bursts(sr, [1.0, 2.0, 3.0], duration=4.0)
    env, hop = comparable_onset_envelope(x, sr)
    assert env.size > 10
    assert hop > 0
    assert float(np.max(env)) > 0


def test_estimate_local_lag_recovers_shift_on_onset_env() -> None:
    sr = 8_000
    times = [3.0, 5.0, 7.0, 9.0, 11.0]
    base = _bursts(sr, times, duration=14.0)
    lag = 0.25
    recon = _shifted(base, sr, lag)
    base_env, hop = comparable_onset_envelope(base, sr)
    recon_env, _ = comparable_onset_envelope(recon, sr)
    est, score, zero = estimate_local_lag(
        base_env,
        recon_env,
        hop=hop,
        center_seconds=7.0,
        window_seconds=4.0,
        max_lag_seconds=0.6,
    )
    assert score > zero
    assert abs(est - lag) <= 0.04


def test_apply_lag_curve_corrects_late_recon() -> None:
    sr = 8_000
    truth_t = 1.0
    recon = _bursts(sr, [truth_t + 0.2], duration=2.5)
    curve = np.full(recon.size, 0.2, dtype=np.float64)
    warped = apply_lag_curve(recon, sr, curve)
    peak_t = int(np.argmax(np.abs(warped))) / sr
    assert abs(peak_t - truth_t) < 0.04


def test_pillars_reject_weak_score_and_interp_preserves_sign() -> None:
    lags = np.asarray([0.45, -0.7, 0.40, 0.42], dtype=np.float64)
    conf = np.asarray([0.5, 0.4, 0.5, 0.5], dtype=np.float64)
    scores = np.asarray([0.78, 0.22, 0.75, 0.76], dtype=np.float64)
    pillars = select_lag_pillars(
        conf, scores, confidence_tau=0.08, min_pillar_score=0.4
    )
    assert bool(pillars[0]) and not bool(pillars[1]) and bool(pillars[2])
    times = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    field = interpolate_from_pillars(times, lags, pillars, fallback_lag=0.0)
    # Weak middle inherits neighbors (~+0.45), not the spurious -0.7 or a negative prior.
    assert field[1] > 0.3


def test_dense_curve_has_no_zero_gaps() -> None:
    times = np.asarray([10.0, 12.0, 14.0], dtype=np.float64)
    lags = np.asarray([0.1, 0.12, 0.11], dtype=np.float64)
    curve = lag_curve_from_dense(times, lags, duration_seconds=20.0, sample_rate=100)
    # Outside knots, endpoints are held — never forced to 0.
    assert float(curve[0]) == 0.1
    assert float(curve[-1]) == 0.11
    mid = curve[int(12.0 * 100)]
    assert abs(mid - 0.12) < 1e-6


def test_plan_and_apply_reduces_large_residual_lag() -> None:
    sr = 8_000
    times = [6.0, 8.0, 10.0, 12.0, 14.0]
    base = _bursts(sr, times, duration=20.0)
    lag = 0.35
    recon = _shifted(base, sr, lag)
    warped, plan = plan_and_apply_fine_warp(
        base,
        recon,
        sr,
        max_lag_seconds=0.8,
        window_seconds=3.0,
        step_seconds=1.0,
        context_window_seconds=6.0,
        prior_weight=0.1,
        smooth_seconds=4.0,
    )
    assert plan.knots
    assert abs(plan.global_lag_seconds - lag) < 0.12

    def _peak_near(signal: np.ndarray, t: float, radius: float = 0.35) -> float:
        lo = max(0, int((t - radius) * sr))
        hi = min(signal.size, int((t + radius) * sr))
        seg = np.abs(np.asarray(signal[lo:hi], dtype=np.float64))
        if float(seg.sum()) < 1e-9:
            return t
        return (lo + int(np.argmax(seg))) / sr

    err_before = [abs(_peak_near(recon, t) - t) for t in times]
    err_after = [abs(_peak_near(np.asarray(warped), t) - t) for t in times]
    assert float(np.mean(err_after)) < float(np.mean(err_before)) * 0.5
    assert float(np.mean(err_after)) < 0.08
