"""Compare pHash-diff vs structural activity on sampled frame directories.

Used to validate encode-robustness before switching the production envelope.
Not a curve-walk smoke — only shape correlation under a known linear map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from jarvis_media_processor.video.activity import (
    aligned_activity_correlation,
    build_activity_envelope,
    build_fused_activity_envelope,
    build_structural_activity_envelope,
)
from jarvis_media_processor.video.hashing import compute_phash
from jarvis_media_processor.video.models import SampledFrame


def sampled_frames_from_dir(
    directory: Path,
    *,
    role: str,
    fps: float = 1.0,
) -> list[SampledFrame]:
    """Load ``frame_*.png`` (or any images) as ``SampledFrame`` with pHashes."""
    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        paths = sorted(directory.glob("*.png"))
    frames: list[SampledFrame] = []
    for index, path in enumerate(paths):
        with Image.open(path) as img:
            phash = str(compute_phash(img))
        time_seconds = index / fps if fps > 0 else float(index)
        frames.append(
            SampledFrame(
                time_seconds=time_seconds,
                path=path.resolve(),
                phash=phash,
                role=role,
            )
        )
    return frames


def refresh_visual_report_structural(
    report_path: Path,
    *,
    settings: object | None = None,
) -> Path:
    """Rewrite a visual match report using fused activity + rematch.

    Reuses existing sampled frame files (no FFmpeg). Updates activity arrays,
    coarse sync, and two-phase matches so curve-walk smokes see the new signal.
    """
    from jarvis_media_processor.config import Settings, get_settings
    from jarvis_media_processor.video.match import match_frames, match_frames_two_phase
    from jarvis_media_processor.video.models import VisualMatchReport
    from jarvis_media_processor.video.shots import shot_boundary_times

    cfg: Settings = settings if isinstance(settings, Settings) else get_settings()
    report = VisualMatchReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    base_frames = list(report.base_frames)
    donor_frames = list(report.donor_frames)
    if not base_frames or not donor_frames:
        raise ValueError(f"Report has no frames: {report_path}")
    for frame in base_frames + donor_frames:
        if not Path(frame.path).is_file():
            raise FileNotFoundError(f"Missing frame file: {frame.path}")

    base_shots = shot_boundary_times(
        base_frames, min_distance=cfg.video_shot_min_distance
    )
    donor_shots = shot_boundary_times(
        donor_frames, min_distance=cfg.video_shot_min_distance
    )
    base_activity = build_fused_activity_envelope(base_frames)
    donor_activity = build_fused_activity_envelope(donor_frames)

    from jarvis_media_processor.alignment.activity_sync import (
        estimate_speed_offset_activity,
    )

    coarse = estimate_speed_offset_activity(
        base_activity,
        donor_activity,
        speed_min=cfg.alignment_speed_min,
        speed_max=cfg.alignment_speed_max,
    )
    matches = match_frames_two_phase(
        base_frames,
        donor_frames,
        base_activity,
        donor_activity,
        speed=coarse.speed,
        offset_seconds=coarse.offset_seconds,
        max_distance=cfg.video_match_max_distance,
        max_matches_per_base=cfg.video_match_max_per_base,
        half_window=cfg.alignment_unrelated_window_seconds,
        min_activity_score=cfg.alignment_unrelated_min_score,
        donor_search_seconds=min(120.0, cfg.alignment_unrelated_max_surplus_seconds),
        base_shot_times=base_shots,
        donor_shot_times=donor_shots,
    )
    match_method = "activity_then_frames"
    if len(matches) < 8:
        brute = match_frames(
            base_frames,
            donor_frames,
            max_distance=cfg.video_match_max_distance,
            max_matches_per_base=cfg.video_match_max_per_base,
            base_shot_times=base_shots,
            donor_shot_times=donor_shots,
        )
        if len(brute) > len(matches):
            matches = brute
            match_method = "phash_bruteforce_fallback"

    params = dict(report.params)
    params.update(
        {
            "activity_sample_period": base_activity.sample_period,
            "activity_method": "fused_max_phash_structural",
            "activity_structural_weight": 0.5,
            "match_method": match_method,
            "activity_sync_speed": coarse.speed,
            "activity_sync_offset": coarse.offset_seconds,
            "activity_sync_score": coarse.score,
            "base_shot_count": len(base_shots),
            "donor_shot_count": len(donor_shots),
        }
    )
    updated = report.model_copy(
        update={
            "matches": matches,
            "base_shot_times": sorted(base_shots),
            "donor_shot_times": sorted(donor_shots),
            "base_activity_times": base_activity.times.tolist(),
            "base_activity_values": base_activity.values.tolist(),
            "donor_activity_times": donor_activity.times.tolist(),
            "donor_activity_values": donor_activity.values.tolist(),
            "params": params,
        }
    )
    report_path.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report_path


def compare_envelopes(
    base_frames: list[SampledFrame],
    donor_frames: list[SampledFrame],
    *,
    speed: float,
    offset_seconds: float,
    base_start: float,
    base_end: float,
) -> dict[str, float]:
    """Return shape correlations for pHash-diff and structural activity."""
    phash_base = build_activity_envelope(base_frames)
    phash_donor = build_activity_envelope(donor_frames)
    struct_base = build_structural_activity_envelope(base_frames)
    struct_donor = build_structural_activity_envelope(donor_frames)
    return {
        "phash_corr": aligned_activity_correlation(
            phash_base,
            phash_donor,
            speed=speed,
            offset_seconds=offset_seconds,
            base_start=base_start,
            base_end=base_end,
        ),
        "structural_corr": aligned_activity_correlation(
            struct_base,
            struct_donor,
            speed=speed,
            offset_seconds=offset_seconds,
            base_start=base_start,
            base_end=base_end,
        ),
        "phash_peak": float(phash_base.values.max()) if phash_base.values.size else 0.0,
        "structural_peak": (
            float(struct_base.values.max()) if struct_base.values.size else 0.0
        ),
    }


def _resolve_visual_dir(smoke_root: Path) -> Path:
    for name in ("align-sr/visual", "align/visual"):
        cand = smoke_root / name
        if (cand / "base_frames").is_dir() and (cand / "donor_frames").is_dir():
            return cand
    raise FileNotFoundError(f"No visual frame dirs under {smoke_root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "smoke_root",
        type=Path,
        help="Smoke scenario dir containing align/visual/{{base,donor}}_frames",
    )
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="Rewrite visual_match_report.json with structural activity + rematch",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--base-start", type=float, default=None)
    parser.add_argument("--base-end", type=float, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args(argv)

    visual = _resolve_visual_dir(args.smoke_root)
    if args.refresh_report:
        report = visual / "visual_match_report.json"
        refresh_visual_report_structural(report)
        print(f"refreshed {report}")
        if args.base_start is None or args.base_end is None:
            return 0

    if args.base_start is None or args.base_end is None:
        parser.error("--base-start and --base-end are required unless --refresh-report")
    base_frames = sampled_frames_from_dir(
        visual / "base_frames", role="base", fps=args.fps
    )
    donor_frames = sampled_frames_from_dir(
        visual / "donor_frames", role="donor", fps=args.fps
    )
    stats = compare_envelopes(
        base_frames,
        donor_frames,
        speed=args.speed,
        offset_seconds=args.offset,
        base_start=args.base_start,
        base_end=args.base_end,
    )
    print(f"smoke={args.smoke_root.name}")
    print(
        f"span base=[{args.base_start:.1f},{args.base_end:.1f}] "
        f"speed={args.speed:.4f} offset={args.offset:.3f}"
    )
    print(f"frames base={len(base_frames)} donor={len(donor_frames)}")
    print(f"phash_corr={stats['phash_corr']:.4f}")
    print(f"structural_corr={stats['structural_corr']:.4f}")
    delta = stats["structural_corr"] - stats["phash_corr"]
    print(f"delta_structural_minus_phash={delta:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
