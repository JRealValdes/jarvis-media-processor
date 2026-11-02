"""Tests for donor audio reconstruction onto the base timeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jarvis_media_processor.alignment.models import (
    AlignmentDocument,
    AlignmentSegment,
    SyncStrategyLevel,
    TimelineInterval,
)
from jarvis_media_processor.audio.wavutil import read_pcm16_wav, write_pcm16_wav
from jarvis_media_processor.reconstruct import plan_placements, reconstruct_from_wav


def _tone(sr: int, duration: float, freq: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_plan_skips_tiny_segments() -> None:
    doc = AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=0.5,
        segments=[
            AlignmentSegment(
                base_start=0.0,
                base_end=0.001,
                donor_start=0.0,
                donor_end=0.001,
                confidence=0.5,
            ),
            AlignmentSegment(
                base_start=1.0,
                base_end=2.0,
                donor_start=3.0,
                donor_end=4.0,
                confidence=0.9,
            ),
        ],
    )
    placements, skipped = plan_placements(doc, min_segment_seconds=0.05)
    assert len(placements) == 1
    assert len(skipped) == 1
    assert placements[0].base_start == 1.0


def test_reconstruct_negative_offset_places_leading_silence(tmp_path: Path) -> None:
    sr = 8_000
    donor = _tone(sr, 1.0, 440.0)
    donor_wav = tmp_path / "donor.wav"
    write_pcm16_wav(donor_wav, donor, sample_rate=sr)

    # donor ≈ base - 1.0  ⇒ tone fills base 1..2
    alignment = AlignmentDocument(
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.9,
        global_offset_seconds=-1.0,
        global_speed=1.0,
        segments=[
            AlignmentSegment(
                base_start=1.0,
                base_end=2.0,
                donor_start=0.0,
                donor_end=1.0,
                offset_seconds=-1.0,
                confidence=0.9,
            )
        ],
    )
    out = tmp_path / "out.wav"
    reconstruct_from_wav(
        donor_wav,
        alignment,
        out,
        base_duration=3.0,
        crossfade_seconds=0.0,
    )
    rendered, _ = read_pcm16_wav(out)
    first = float(np.sqrt(np.mean(rendered[:sr] ** 2)))
    mid = float(np.sqrt(np.mean(rendered[sr : 2 * sr] ** 2)))
    last = float(np.sqrt(np.mean(rendered[2 * sr :] ** 2)))
    assert first < 0.05
    assert mid > 0.1
    assert last < 0.05


def test_reconstruct_constant_offset_no_shift(tmp_path: Path) -> None:
    sr = 8_000
    # Donor: 1s silence + 1s tone
    donor = np.concatenate([np.zeros(sr), _tone(sr, 1.0, 440.0)])
    donor_wav = tmp_path / "donor.wav"
    write_pcm16_wav(donor_wav, donor, sample_rate=sr)

    # Base timeline 3s; tone should land at 0..1 (offset donor+1 -> base 0)
    alignment = AlignmentDocument(
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.9,
        global_offset_seconds=1.0,
        global_speed=1.0,
        segments=[
            AlignmentSegment(
                base_start=0.0,
                base_end=1.0,
                donor_start=1.0,
                donor_end=2.0,
                offset_seconds=1.0,
                confidence=0.9,
            )
        ],
    )
    out = tmp_path / "out.wav"
    result = reconstruct_from_wav(
        donor_wav,
        alignment,
        out,
        base_duration=3.0,
        crossfade_seconds=0.0,
    )
    assert abs(result.duration_seconds - 3.0) < 0.05
    rendered, _ = read_pcm16_wav(out)
    # Energy in first second, silence after (gap does not shift)
    first = float(np.sqrt(np.mean(rendered[:sr] ** 2)))
    mid = float(np.sqrt(np.mean(rendered[sr : 2 * sr] ** 2)))
    last = float(np.sqrt(np.mean(rendered[2 * sr :] ** 2)))
    assert first > 0.1
    assert mid < 0.05
    assert last < 0.05


def test_missing_base_gap_filled_from_base_track(tmp_path: Path) -> None:
    sr = 8_000
    # Donor has content only in 0..1 and 2..3; base gap 1..2 should take base audio.
    donor = np.concatenate([_tone(sr, 1.0, 440.0), np.zeros(sr), _tone(sr, 1.0, 440.0)])
    base_audio = np.concatenate(
        [np.zeros(sr), _tone(sr, 1.0, 880.0), np.zeros(sr)]
    )
    donor_wav = tmp_path / "donor.wav"
    base_wav = tmp_path / "base.wav"
    write_pcm16_wav(donor_wav, donor, sample_rate=sr)
    write_pcm16_wav(base_wav, base_audio, sample_rate=sr)

    alignment = AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=0.8,
        segments=[
            AlignmentSegment(
                base_start=0.0,
                base_end=1.0,
                donor_start=0.0,
                donor_end=1.0,
                confidence=0.9,
            ),
            AlignmentSegment(
                base_start=2.0,
                base_end=3.0,
                donor_start=2.0,
                donor_end=3.0,
                confidence=0.9,
            ),
        ],
        missing_base_intervals=[
            TimelineInterval(
                start=1.0,
                end=2.0,
                timeline="base",
                reason="curve_break_missing_base",
            )
        ],
    )
    out = tmp_path / "out.wav"
    reconstruct_from_wav(
        donor_wav,
        alignment,
        out,
        base_duration=3.0,
        crossfade_seconds=0.0,
        base_wav=base_wav,
        missing_base_fill_seconds=0.5,
    )
    rendered, _ = read_pcm16_wav(out)
    mid = rendered[sr : 2 * sr]
    # 880 Hz energy present in the filled gap.
    assert float(np.sqrt(np.mean(mid**2))) > 0.1


def test_unrelated_donor_span_carved_to_silence(tmp_path: Path) -> None:
    sr = 8_000
    # Continuous tone across donor 0..3s; ad flagged at donor 1..2 must stay silent on base.
    donor = _tone(sr, 3.0, 440.0)
    donor_wav = tmp_path / "donor.wav"
    write_pcm16_wav(donor_wav, donor, sample_rate=sr)

    alignment = AlignmentDocument(
        strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        confidence=0.9,
        global_offset_seconds=0.0,
        global_speed=1.0,
        segments=[
            AlignmentSegment(
                base_start=0.0,
                base_end=3.0,
                donor_start=0.0,
                donor_end=3.0,
                offset_seconds=0.0,
                confidence=0.9,
            )
        ],
        unrelated_donor_intervals=[
            TimelineInterval(
                start=1.0,
                end=2.0,
                timeline="donor",
                reason="activity_decorrelated",
            )
        ],
    )
    out = tmp_path / "out.wav"
    reconstruct_from_wav(
        donor_wav,
        alignment,
        out,
        base_duration=3.0,
        crossfade_seconds=0.0,
    )
    rendered, _ = read_pcm16_wav(out)
    first = float(np.sqrt(np.mean(rendered[:sr] ** 2)))
    mid = float(np.sqrt(np.mean(rendered[sr : 2 * sr] ** 2)))
    last = float(np.sqrt(np.mean(rendered[2 * sr :] ** 2)))
    assert first > 0.1
    assert mid < 0.05
    assert last > 0.1


def test_missing_gap_does_not_shift_later(tmp_path: Path) -> None:
    sr = 8_000
    donor = _tone(sr, 2.0, 330.0)
    donor_wav = tmp_path / "donor.wav"
    write_pcm16_wav(donor_wav, donor, sample_rate=sr)

    alignment = AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=0.8,
        segments=[
            AlignmentSegment(
                base_start=0.0,
                base_end=0.5,
                donor_start=0.0,
                donor_end=0.5,
                confidence=0.9,
            ),
            # Gap on base 0.5..1.5 left silent
            AlignmentSegment(
                base_start=1.5,
                base_end=2.0,
                donor_start=0.5,
                donor_end=1.0,
                confidence=0.9,
            ),
        ],
    )
    out = tmp_path / "out.wav"
    reconstruct_from_wav(
        donor_wav,
        alignment,
        out,
        base_duration=2.0,
        crossfade_seconds=0.0,
    )
    rendered, _ = read_pcm16_wav(out)
    gap = float(np.sqrt(np.mean(rendered[int(0.5 * sr) : int(1.5 * sr)] ** 2)))
    late = float(np.sqrt(np.mean(rendered[int(1.5 * sr) :] ** 2)))
    assert gap < 0.05
    assert late > 0.05
