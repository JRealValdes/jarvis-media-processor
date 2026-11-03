"""End-to-end donor audio reconstruction onto the base timeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from jarvis_media_processor.alignment.models import AlignmentDocument
from jarvis_media_processor.audio.wavutil import read_pcm16_wav, write_pcm16_wav
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.reconstruct.extract import extract_donor_wav
from jarvis_media_processor.reconstruct.models import ReconstructionResult
from jarvis_media_processor.reconstruct.plan import plan_placements
from jarvis_media_processor.reconstruct.render import (
    fill_missing_base_audio,
    render_placements,
)
from jarvis_media_processor.report.build import (
    build_processing_report,
    write_processing_report,
)

logger = logging.getLogger(__name__)


def _wav_to_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    if samples.ndim == 1:
        mono = samples.astype(np.float64)
        if channels == 1:
            return mono
        return np.stack([mono] * channels, axis=0)
    # Multi-channel interleaved was averaged in read_pcm16_wav currently.
    # read returns mono always today — keep shape handling future-proof.
    return samples


def reconstruct_donor_audio(
    donor_path: Path | str,
    alignment: AlignmentDocument,
    workspace: Path | str,
    *,
    base_duration: float,
    settings: Settings | None = None,
    sample_rate: int = 48_000,
    channels: int = 2,
    min_segment_seconds: float = 0.05,
    require_reliable: bool = False,
    crossfade_seconds: float = 0.01,
    audio_stream_index: int | None = None,
) -> ReconstructionResult:
    """Extract donor audio and render it onto a base-duration timeline.

    Args:
        donor_path: Donor media (never modified).
        alignment: Alignment document from sync detection.
        workspace: Directory for intermediate WAV / JSON artifacts.
        base_duration: Output length in seconds (base timeline).
        settings: Optional settings.
        sample_rate: Reconstruction sample rate.
        channels: Output channel count.
        min_segment_seconds: Ignore tiny alignment segments.
        require_reliable: Only use ``reliable`` segments when True.
        crossfade_seconds: Edge fade to reduce clicks.
        audio_stream_index: Optional donor audio stream selector.

    Returns:
        ``ReconstructionResult`` with output WAV path and placement map.
    """
    cfg = settings or get_settings()
    donor = Path(donor_path)
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)

    donor_wav = work / "donor_source.wav"
    extract_donor_wav(
        donor,
        donor_wav,
        settings=cfg,
        sample_rate=sample_rate,
        channels=1 if channels == 1 else channels,
        audio_stream_index=audio_stream_index,
    )
    samples, sr = read_pcm16_wav(donor_wav, mix_to_mono=(channels == 1))
    if sr != sample_rate:
        sample_rate = sr

    placements, skipped = plan_placements(
        alignment,
        min_segment_seconds=min_segment_seconds,
        require_reliable=require_reliable,
        crossfade_seconds=crossfade_seconds,
    )
    donor_ch = _wav_to_channels(samples, channels)
    canvas = render_placements(
        donor_ch,
        sample_rate,
        placements,
        base_duration=base_duration,
    )

    filled = 0
    base_path = alignment.base_path
    min_fill = float(cfg.alignment_missing_base_fill_seconds)
    if base_path is not None and Path(base_path).is_file() and alignment.missing_base_intervals:
        base_wav = work / "base_track0.wav"
        extract_donor_wav(
            base_path,
            base_wav,
            settings=cfg,
            sample_rate=sample_rate,
            channels=1 if channels == 1 else channels,
            audio_stream_index=0,
        )
        base_samples, _ = read_pcm16_wav(base_wav, mix_to_mono=(channels == 1))
        before = canvas.copy()
        canvas = fill_missing_base_audio(
            canvas,
            base_samples,
            sample_rate,
            alignment.missing_base_intervals,
            min_seconds=min_fill,
            crossfade_seconds=crossfade_seconds,
        )
        filled = sum(
            1
            for item in alignment.missing_base_intervals
            if item.timeline == "base" and item.end - item.start >= min_fill - 1e-9
        )
        _ = before  # kept for clarity that we replace in-place via return

    out_wav = work / "donor_on_base.wav"
    write_pcm16_wav(out_wav, canvas, sample_rate=sample_rate)
    out_channels = int(canvas.shape[0])

    notes = [
        f"placements={len(placements)}",
        f"skipped={len(skipped)}",
        f"base_duration={base_duration:.3f}s",
        (
            f"Filled {filled} missing-base gap(s) from base track 0 "
            f"(threshold={min_fill:.1f}s)."
            if filled
            else "Missing base regions left as silence when below fill threshold."
        ),
        "Segment edges use short fades to reduce clicks.",
        "Quality path: PCM WAV extract (default 48 kHz); avoid re-encoding until mux.",
    ]

    result = ReconstructionResult(
        donor_path=donor.resolve(),
        output_wav=out_wav.resolve(),
        sample_rate=sample_rate,
        channels=out_channels,
        duration_seconds=float(canvas.shape[-1] / sample_rate),
        strategy=alignment.strategy,
        placements=placements,
        skipped_segments=skipped,
        notes=notes,
    )
    (work / "reconstruction.json").write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_processing_report(
        alignment,
        result,
        base_duration=base_duration,
        settings=cfg,
    )
    report_path = write_processing_report(report, work)
    logger.info(
        "Reconstructed donor audio -> %s (%.2fs, %d placements); report=%s review=%s",
        out_wav,
        result.duration_seconds,
        len(placements),
        report_path,
        report.needs_manual_review,
    )
    return result


def reconstruct_from_wav(
    donor_wav: Path | str,
    alignment: AlignmentDocument,
    output_wav: Path | str,
    *,
    base_duration: float,
    min_segment_seconds: float = 0.05,
    require_reliable: bool = False,
    crossfade_seconds: float = 0.01,
    base_wav: Path | str | None = None,
    missing_base_fill_seconds: float = 3.0,
) -> ReconstructionResult:
    """Reconstruct using an already-extracted donor WAV (unit-test friendly)."""
    samples, sample_rate = read_pcm16_wav(Path(donor_wav))
    placements, skipped = plan_placements(
        alignment,
        min_segment_seconds=min_segment_seconds,
        require_reliable=require_reliable,
        crossfade_seconds=crossfade_seconds,
    )
    canvas = render_placements(
        samples,
        sample_rate,
        placements,
        base_duration=base_duration,
    )
    notes = ["reconstruct_from_wav"]
    if base_wav is not None and alignment.missing_base_intervals:
        base_samples, _ = read_pcm16_wav(Path(base_wav))
        canvas = fill_missing_base_audio(
            canvas,
            base_samples,
            sample_rate,
            alignment.missing_base_intervals,
            min_seconds=missing_base_fill_seconds,
            crossfade_seconds=crossfade_seconds,
        )
        notes.append(
            f"filled_missing_base_threshold={missing_base_fill_seconds:.1f}s"
        )
    out = Path(output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    mono = canvas[0] if canvas.ndim == 2 else canvas
    write_pcm16_wav(out, mono, sample_rate=sample_rate)
    return ReconstructionResult(
        donor_path=Path(donor_wav).resolve(),
        output_wav=out.resolve(),
        sample_rate=sample_rate,
        channels=1,
        duration_seconds=float(mono.shape[-1] / sample_rate),
        strategy=alignment.strategy,
        placements=placements,
        skipped_segments=skipped,
        notes=notes,
    )
