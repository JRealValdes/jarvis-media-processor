"""Generate synthetic base/donor media pairs with FFmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.fixtures.models import GroundTruth
from jarvis_media_processor.fixtures.scenarios import SCENARIOS, get_scenario

logger = logging.getLogger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg executable is missing or not runnable."""


def resolve_ffmpeg(ffmpeg_path: str) -> str:
    """Resolve an FFmpeg executable path or name on ``PATH``."""
    candidate = Path(ffmpeg_path)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(ffmpeg_path)
    if found:
        return found
    raise FFmpegNotFoundError(
        f"FFmpeg not found: {ffmpeg_path!r}. Install FFmpeg and ensure it is on PATH, "
        "or set JMP_FFMPEG_PATH."
    )


def _run(cmd: list[str]) -> None:
    logger.debug("Running: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"FFmpeg failed ({completed.returncode}): {stderr}")


def _render_tone_clip(
    ffmpeg: str,
    out: Path,
    *,
    duration: float,
    color: str,
    freq: int,
    beep_times: list[float] | None = None,
) -> None:
    """Render a short clip: solid color video + sine tone, optional loud beeps."""
    beep_times = beep_times or []
    input_args: list[str] = [
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=160x120:r=25:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={freq}:sample_rate=48000:duration={duration}",
    ]
    filter_parts = ["[1:a]volume=0.15[basea]"]
    mix_labels = ["[basea]"]
    next_in = 2
    for i, t in enumerate(beep_times):
        if t < 0 or t >= duration:
            continue
        input_args.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=2000:sample_rate=48000:duration=0.08",
            ]
        )
        delay_ms = int(round(t * 1000))
        filter_parts.append(
            f"[{next_in}:a]volume=0.95,adelay={delay_ms}|{delay_ms},apad=whole_dur={duration}[b{i}]"
        )
        mix_labels.append(f"[b{i}]")
        next_in += 1

    n = len(mix_labels)
    mix = "".join(mix_labels) + f"amix=inputs={n}:duration=first:dropout_transition=0[aout]"
    filter_complex = ";".join([*filter_parts, mix])

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    ]
    _run(cmd)


def _concat_clips(ffmpeg: str, clips: list[Path], out: Path) -> None:
    list_file = out.with_suffix(".concat.txt")
    lines = [f"file '{c.resolve().as_posix()}'" for c in clips]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        _run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(out),
            ]
        )
    finally:
        list_file.unlink(missing_ok=True)


def _silence_pad(ffmpeg: str, src: Path, out: Path, pad_seconds: float) -> None:
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=160x120:r=25:d={pad_seconds}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=48000:cl=stereo:d={pad_seconds}",
            "-i",
            str(src),
            "-filter_complex",
            "[0:v][1:a][2:v][2:a]concat=n=2:v=1:a=1[v][a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out),
        ]
    )


def _change_speed(ffmpeg: str, src: Path, out: Path, speed: float) -> None:
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-filter_complex",
            f"[0:v]setpts=PTS/{speed}[v];[0:a]atempo={speed}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out),
        ]
    )


def _write_truth(truth: GroundTruth, out_dir: Path) -> Path:
    path = out_dir / "ground_truth.json"
    path.write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def generate_scenario(
    scenario_id: str,
    output_dir: Path,
    *,
    settings: Settings | None = None,
) -> GroundTruth:
    """Generate media files and ``ground_truth.json`` for one scenario.

    Args:
        scenario_id: Catalog scenario id.
        output_dir: Destination directory (created if needed).
        settings: Optional settings for FFmpeg path.

    Returns:
        The scenario ground truth (also written to disk).

    Raises:
        FFmpegNotFoundError: If FFmpeg is unavailable.
        KeyError: If ``scenario_id`` is unknown.
        RuntimeError: If FFmpeg fails.
    """
    cfg = settings or get_settings()
    ffmpeg = resolve_ffmpeg(cfg.ffmpeg_path)
    truth = get_scenario(scenario_id)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        base_path = out_dir / truth.base.filename
        donor_path = out_dir / truth.donor.filename

        scenes = [
            ("red", 440, 2.0),
            ("green", 660, 2.0),
            ("blue", 880, 2.0),
        ]

        base_parts: list[Path] = []
        for i, (color, freq, dur) in enumerate(scenes):
            part = work / f"base_scene{i}.mkv"
            beeps = [
                lm.base_time - (i * 2.0)
                for lm in truth.landmarks
                if i * 2.0 <= lm.base_time < i * 2.0 + dur
            ]
            beeps = [b for b in beeps if 0 <= b < dur]
            _render_tone_clip(ffmpeg, part, duration=dur, color=color, freq=freq, beep_times=beeps)
            base_parts.append(part)
        _concat_clips(ffmpeg, base_parts, base_path)

        if scenario_id == "already_synced":
            shutil.copyfile(base_path, donor_path)
        elif scenario_id == "constant_offset":
            offset = truth.global_offset_seconds or 0.0
            tmp = work / "donor_core.mkv"
            shutil.copyfile(base_path, tmp)
            _silence_pad(ffmpeg, tmp, donor_path, offset)
        elif scenario_id == "unrelated_block":
            unrelated = work / "unrelated.mkv"
            _render_tone_clip(
                ffmpeg,
                unrelated,
                duration=2.0,
                color="yellow",
                freq=120,
                beep_times=[],
            )
            d0 = work / "donor_scene0.mkv"
            d1 = work / "donor_scene1.mkv"
            d2 = work / "donor_scene2.mkv"
            _render_tone_clip(ffmpeg, d0, duration=2.0, color="red", freq=440, beep_times=[0.5])
            _render_tone_clip(ffmpeg, d1, duration=2.0, color="green", freq=660, beep_times=[0.5])
            _render_tone_clip(ffmpeg, d2, duration=2.0, color="blue", freq=880, beep_times=[0.5])
            _concat_clips(ffmpeg, [d0, unrelated, d1, d2], donor_path)
        elif scenario_id == "constant_speed":
            speed = truth.global_speed or 1.1
            _change_speed(ffmpeg, base_path, donor_path, speed)
        elif scenario_id == "local_length_drift":
            d0 = work / "d0.mkv"
            d1 = work / "d1.mkv"
            d2 = work / "d2.mkv"
            _render_tone_clip(ffmpeg, d0, duration=2.0, color="red", freq=440, beep_times=[0.5])
            _render_tone_clip(ffmpeg, d1, duration=3.0, color="green", freq=660, beep_times=[0.5])
            _render_tone_clip(ffmpeg, d2, duration=2.0, color="blue", freq=880, beep_times=[0.5])
            _concat_clips(ffmpeg, [d0, d1, d2], donor_path)
        else:
            raise KeyError(f"No generator implementation for {scenario_id!r}")

        _write_truth(truth, out_dir)
        logger.info("Generated scenario %s in %s", scenario_id, out_dir)
        return truth
    finally:
        shutil.rmtree(work, ignore_errors=True)


def generate_scenarios(
    output_root: Path,
    scenario_ids: list[str] | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, GroundTruth]:
    """Generate multiple scenarios under ``output_root/<scenario_id>/``."""
    ids = scenario_ids or list(SCENARIOS.keys())
    results: dict[str, GroundTruth] = {}
    for scenario_id in ids:
        results[scenario_id] = generate_scenario(
            scenario_id,
            Path(output_root) / scenario_id,
            settings=settings,
        )
    return results
