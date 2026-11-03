"""Run windowed fine-warp on a reconstructed WAV against base track 0."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from jarvis_media_processor.audio.wavutil import read_pcm16_wav, write_pcm16_wav
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError
from jarvis_media_processor.reconstruct.extract import extract_donor_wav
from jarvis_media_processor.reconstruct.fine_warp import (
    FineWarpPlan,
    LagKnot,
    plan_and_apply_fine_warp,
)

logger = logging.getLogger(__name__)


class FineWarpResult(BaseModel):
    """Artifacts from a post-reconstruct fine-warp pass."""

    base_path: Path
    input_wav: Path
    output_wav: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    knot_count: int = 0
    knots: list[dict[str, float]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    plan_json: Path | None = None


def _knots_as_dicts(knots: list[LagKnot]) -> list[dict[str, float]]:
    return [
        {
            "time_seconds": k.time_seconds,
            "lag_seconds": k.lag_seconds,
            "score": k.score,
            "improvement": k.improvement,
        }
        for k in knots
    ]


def _write_plan_json(path: Path, plan: FineWarpPlan) -> None:
    payload = {
        "knots": _knots_as_dicts(plan.knots),
        "skipped_notes": list(plan.skipped_notes),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fine_warp_reconstructed_audio(
    base_path: Path | str,
    reconstructed_wav: Path | str,
    workspace: Path | str,
    *,
    settings: Settings | None = None,
    output_name: str = "donor_on_base_fine.wav",
) -> FineWarpResult:
    """Fine-warp reconstructed donor audio to better match base track 0.

    Uses a windowed lag field accepted only when the neighborhood improves.
    """
    cfg = settings or get_settings()
    base = Path(base_path)
    inp = Path(reconstructed_wav)
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)

    if not base.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {base}")
    if not inp.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {inp}")

    recon, sample_rate = read_pcm16_wav(inp, mix_to_mono=False)
    channels = 1 if recon.ndim == 1 else int(recon.shape[0])

    base_wav = work / "base_track0_fine_warp.wav"
    extract_donor_wav(
        base,
        base_wav,
        settings=cfg,
        sample_rate=sample_rate,
        channels=1,
        audio_stream_index=0,
    )
    base_samples, base_sr = read_pcm16_wav(base_wav, mix_to_mono=True)
    if base_sr != sample_rate:
        raise ValueError(
            f"Base extract sample rate {base_sr} != reconstructed {sample_rate}"
        )

    warped, plan = plan_and_apply_fine_warp(
        base_samples,
        recon,
        sample_rate,
        max_lag_seconds=cfg.fine_warp_max_lag_seconds,
        window_seconds=cfg.fine_warp_window_seconds,
        step_seconds=cfg.fine_warp_step_seconds,
        min_score=cfg.fine_warp_min_score,
        min_score_margin=cfg.fine_warp_min_score_margin,
        min_improvement=cfg.fine_warp_min_improvement,
        neighbor_radius_seconds=cfg.fine_warp_neighbor_radius_seconds,
        neighbor_step_seconds=cfg.fine_warp_neighbor_step_seconds,
        max_lag_jump_seconds=cfg.fine_warp_max_lag_jump_seconds,
        interp_gap_seconds=cfg.fine_warp_interp_gap_seconds,
    )

    out_wav = work / output_name
    write_pcm16_wav(out_wav, warped, sample_rate=sample_rate)
    plan_path = work / "fine_warp_plan.json"
    _write_plan_json(plan_path, plan)

    n = int(warped.shape[-1]) if getattr(warped, "ndim", 1) >= 1 else 0
    notes = [
        f"knots={len(plan.knots)}",
        *plan.skipped_notes[:20],
    ]
    if len(plan.skipped_notes) > 20:
        notes.append(f"...and {len(plan.skipped_notes) - 20} more skip notes.")
    logger.info("Fine-warp: %d lag knots → %s", len(plan.knots), out_wav)

    return FineWarpResult(
        base_path=base,
        input_wav=inp,
        output_wav=out_wav,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=float(n / sample_rate) if sample_rate else 0.0,
        knot_count=len(plan.knots),
        knots=_knots_as_dicts(plan.knots),
        notes=notes,
        plan_json=plan_path,
    )
