"""FFprobe subprocess integration."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from jarvis_media_processor.media.errors import FFprobeError, FFprobeNotFoundError

logger = logging.getLogger(__name__)


def resolve_ffprobe(ffprobe_path: str) -> str:
    """Resolve an FFprobe executable path or name on ``PATH``.

    Args:
        ffprobe_path: Absolute path or executable name.

    Returns:
        Path string suitable for ``subprocess``.

    Raises:
        FFprobeNotFoundError: If the executable cannot be found.
    """
    candidate = Path(ffprobe_path)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(ffprobe_path)
    if found:
        return found
    raise FFprobeNotFoundError(
        f"FFprobe not found: {ffprobe_path!r}. Install FFmpeg/FFprobe and ensure it is on PATH, "
        "or set JMP_FFPROBE_PATH."
    )


def run_ffprobe_json(path: Path, *, ffprobe_path: str = "ffprobe") -> dict[str, Any]:
    """Run FFprobe and return parsed JSON for ``path``.

    Args:
        path: Existing media file to inspect.
        ffprobe_path: FFprobe executable name or absolute path.

    Returns:
        Parsed FFprobe JSON object.

    Raises:
        FFprobeNotFoundError: If FFprobe is not available.
        FFprobeError: If FFprobe fails or returns invalid JSON.
    """
    exe = resolve_ffprobe(ffprobe_path)
    cmd = [
        exe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    logger.debug("Running FFprobe: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise FFprobeNotFoundError(f"Failed to execute FFprobe at {exe!r}: {exc}") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise FFprobeError(
            f"FFprobe failed for {path}: {stderr or f'exit code {completed.returncode}'}",
            returncode=completed.returncode,
            stderr=stderr or None,
        )

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFprobeError(
            f"FFprobe returned invalid JSON for {path}: {exc}",
            returncode=completed.returncode,
            stderr=(completed.stderr or "").strip() or None,
        ) from exc

    if not isinstance(payload, dict):
        raise FFprobeError(f"FFprobe JSON root must be an object for {path}")
    return payload
