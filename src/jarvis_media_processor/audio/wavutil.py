"""Minimal WAV helpers (PCM 16-bit) without extra audio I/O dependencies."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_pcm16_wav(
    path: Path,
    samples: NDArray[np.floating],
    *,
    sample_rate: int,
) -> None:
    """Write mono float samples in [-1, 1] as 16-bit PCM WAV.

    Args:
        path: Destination path.
        samples: 1-D float array.
        sample_rate: Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples.astype(np.float64), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_pcm16_wav(path: Path) -> tuple[NDArray[np.float64], int]:
    """Read a mono or multi-channel PCM WAV into float64 mono.

    Args:
        path: WAV file path.

    Returns:
        ``(samples, sample_rate)`` with samples in approximately [-1, 1].

    Raises:
        ValueError: If the WAV is not 16-bit PCM.
    """
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported, got sample_width={sample_width}")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate
