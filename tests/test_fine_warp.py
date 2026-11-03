"""Unit tests for windowed post-reconstruct fine-warp."""

from __future__ import annotations

import numpy as np

from jarvis_media_processor.audio.landmarks import rms_envelope
from jarvis_media_processor.reconstruct.fine_warp import (
    LagKnot,
    apply_lag_curve,
    estimate_local_lag,
    lag_curve_from_knots,
    neighborhood_improvement,
    plan_and_apply_fine_warp,
    smooth_lag_knots,
)


def _bursts(
    sample_rate: int,
    times: list[float],
    *,
    duration: float,
    width: float = 0.04,
    amp: float = 0.9,
) -> np.ndarray:
    n = int(round(duration * sample_rate))
    out = np.zeros(n, dtype=np.float64)
    half = max(1, int(round(width * sample_rate)))
    for t in times:
        center = int(round(t * sample_rate))
        a = max(0, center - half // 2)
        b = min(n, a + half)
        if b > a:
            out[a:b] = amp
    return out


def _shifted(signal: np.ndarray, sample_rate: int, lag_seconds: float) -> np.ndarray:
    """Positive lag ⇒ content moves later (pad leading silence)."""
    shift = int(round(lag_seconds * sample_rate))
    out = np.zeros_like(signal)
    if shift >= 0:
        out[shift:] = signal[: signal.size - shift] if shift < signal.size else 0
    else:
        out[: signal.size + shift] = signal[-shift:]
    return out


def test_estimate_local_lag_recovers_constant_shift() -> None:
    sr = 8_000
    times = [2.0, 4.0, 6.0, 8.0, 10.0]
    base = _bursts(sr, times, duration=12.0)
    lag = 0.08
    recon = _shifted(base, sr, lag)
    base_env, hop = rms_envelope(base, sr)
    recon_env, _ = rms_envelope(recon, sr)
    est, score, zero = estimate_local_lag(
        base_env,
        recon_env,
        hop=hop,
        center_seconds=6.0,
        window_seconds=3.0,
        max_lag_seconds=0.15,
    )
    assert score > zero
    assert abs(est - lag) <= hop + 1e-9


def test_neighborhood_improvement_positive_for_true_lag() -> None:
    sr = 8_000
    times = [3.0, 5.0, 7.0, 9.0]
    base = _bursts(sr, times, duration=12.0)
    lag = 0.07
    recon = _shifted(base, sr, lag)
    base_env, hop = rms_envelope(base, sr)
    recon_env, _ = rms_envelope(recon, sr)
    improve = neighborhood_improvement(
        base_env,
        recon_env,
        hop=hop,
        center_seconds=6.0,
        proposed_lag=lag,
        window_seconds=2.5,
        neighbor_radius_seconds=3.0,
        neighbor_step_seconds=1.0,
    )
    assert improve > 0.01


def test_smooth_drops_lag_jumps() -> None:
    knots = [
        LagKnot(1.0, 0.05, 0.8, 0.1),
        LagKnot(2.0, 0.06, 0.8, 0.1),
        LagKnot(3.0, -0.10, 0.8, 0.1),  # jump
        LagKnot(4.0, 0.05, 0.8, 0.1),
    ]
    kept, notes = smooth_lag_knots(knots, max_lag_jump_seconds=0.04, median_radius=0)
    assert all(abs(k.lag_seconds - (-0.10)) > 1e-9 for k in kept)
    assert any("lag jump" in n for n in notes)


def test_apply_lag_curve_pulls_early_content_later() -> None:
    sr = 8_000
    # Event at 1.0 in "truth"; recon places it 80ms early (ahead).
    truth_t = 1.0
    recon = _bursts(sr, [truth_t - 0.08], duration=2.5)
    curve = np.full(recon.size, -0.08, dtype=np.float64)
    warped = apply_lag_curve(recon, sr, curve)
    peak_t = int(np.argmax(warped)) / sr
    assert abs(peak_t - truth_t) < 0.03


def test_lag_curve_identity_without_knots() -> None:
    curve = lag_curve_from_knots([], duration_seconds=2.0, sample_rate=1000)
    assert curve.shape == (2000,)
    assert np.allclose(curve, 0.0)


def test_plan_and_apply_reduces_constant_residual_lag() -> None:
    sr = 8_000
    # Keep events away from envelope edges (analysis needs pad for window+neighbors).
    times = [4.0, 5.0, 6.0, 7.0, 8.0]
    base = _bursts(sr, times, duration=12.0)
    lag = 0.08
    recon = _shifted(base, sr, lag)
    warped, plan = plan_and_apply_fine_warp(
        base,
        recon,
        sr,
        max_lag_seconds=0.15,
        window_seconds=2.5,
        step_seconds=0.75,
        min_score=0.2,
        min_score_margin=0.02,
        min_improvement=0.01,
        neighbor_radius_seconds=2.5,
        neighbor_step_seconds=0.75,
        max_lag_jump_seconds=0.05,
        interp_gap_seconds=2.5,
    )
    assert plan.knots

    def _centroid_near(signal: np.ndarray, t: float, radius: float = 0.25) -> float:
        lo = max(0, int((t - radius) * sr))
        hi = min(signal.size, int((t + radius) * sr))
        seg = np.abs(np.asarray(signal[lo:hi], dtype=np.float64))
        if float(seg.sum()) < 1e-9:
            return t
        idx = lo + int(round(float(np.average(np.arange(seg.size), weights=seg))))
        return idx / sr

    err_before = [abs(_centroid_near(recon, t) - t) for t in times]
    err_after = [abs(_centroid_near(np.asarray(warped), t) - t) for t in times]
    assert float(np.mean(err_after)) < float(np.mean(err_before)) * 0.55
    assert float(np.mean(err_after)) < 0.035
