"""Tests for audio extraction helpers and landmark detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jarvis_media_processor.audio import (
    analyze_wav_file,
    build_extract_command,
    detect_energy_landmarks,
)
from jarvis_media_processor.audio.wavutil import read_pcm16_wav, write_pcm16_wav
from jarvis_media_processor.config import Settings


def _synth_with_beeps(
    sample_rate: int = 16_000,
    duration: float = 2.0,
    beep_times: list[float] | None = None,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    samples = 0.05 * np.sin(2 * np.pi * 220 * t)
    for bt in beep_times or []:
        mask = (t >= bt) & (t < bt + 0.05)
        samples[mask] += 0.9 * np.sin(2 * np.pi * 2000 * t[mask])
    return samples


def test_build_extract_command() -> None:
    cmd = build_extract_command(
        "ffmpeg",
        Path("base.mkv"),
        Path("out.wav"),
        sample_rate=16_000,
        channels=1,
    )
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and "base.mkv" in cmd
    assert "pcm_s16le" in cmd
    assert cmd[-1].endswith("out.wav")


def test_wav_roundtrip(tmp_path: Path) -> None:
    samples = _synth_with_beeps(beep_times=[0.5, 1.2])
    path = tmp_path / "t.wav"
    write_pcm16_wav(path, samples, sample_rate=16_000)
    loaded, sr = read_pcm16_wav(path)
    assert sr == 16_000
    assert loaded.shape == samples.shape
    assert np.corrcoef(samples, loaded)[0, 1] > 0.99


def test_detect_beep_landmarks(tmp_path: Path) -> None:
    sr = 16_000
    beep_times = [0.4, 1.1]
    samples = _synth_with_beeps(sample_rate=sr, beep_times=beep_times)
    path = tmp_path / "beeps.wav"
    write_pcm16_wav(path, samples, sample_rate=sr)

    settings = Settings(
        _env_file=None,
        audio_landmark_prominence=0.2,
        audio_landmark_min_distance_seconds=0.2,
    )
    artifact = analyze_wav_file(path, role="base", settings=settings)
    assert artifact.role == "base"
    assert len(artifact.landmarks) >= 2

    detected = sorted(lm.time_seconds for lm in artifact.landmarks)
    for expected in beep_times:
        assert any(abs(t - expected) < 0.08 for t in detected), (expected, detected)


def test_detect_energy_landmarks_stable() -> None:
    sr = 16_000
    samples = _synth_with_beeps(sample_rate=sr, beep_times=[0.5, 1.5])
    a, _, _ = detect_energy_landmarks(samples, sr, prominence=0.2, min_distance_seconds=0.2)
    b, _, _ = detect_energy_landmarks(samples, sr, prominence=0.2, min_distance_seconds=0.2)
    assert [lm.time_seconds for lm in a] == [lm.time_seconds for lm in b]
    assert [lm.strength for lm in a] == [lm.strength for lm in b]


def test_quiet_signal_no_crash() -> None:
    samples = np.zeros(16_000, dtype=np.float64)
    landmarks, env, _hop = detect_energy_landmarks(samples, 16_000)
    assert landmarks == []
    assert env.size > 0
