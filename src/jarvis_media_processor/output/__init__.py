"""Output MKV generation, muxing, and optimize/compress."""

from jarvis_media_processor.output.mux import (
    OutputPathError,
    SubtitleSpecError,
    SubtitleTrack,
    build_mux_mkv_command,
    mux_mkv,
    parse_subtitle_spec,
    validate_output_path,
)
from jarvis_media_processor.output.optimize import (
    OptimizeDecision,
    OptimizeProfile,
    OptimizeSpecError,
    SelectedStreams,
    build_optimize_command,
    estimate_video_bitrate_bps,
    optimize_mkv,
    parse_language_list,
    select_streams,
    should_optimize_output,
)

__all__ = [
    "OptimizeDecision",
    "OptimizeProfile",
    "OptimizeSpecError",
    "OutputPathError",
    "SelectedStreams",
    "SubtitleSpecError",
    "SubtitleTrack",
    "build_mux_mkv_command",
    "build_optimize_command",
    "estimate_video_bitrate_bps",
    "mux_mkv",
    "optimize_mkv",
    "parse_language_list",
    "parse_subtitle_spec",
    "select_streams",
    "should_optimize_output",
    "validate_output_path",
]
