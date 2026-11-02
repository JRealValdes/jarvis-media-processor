"""Video frame extraction and visual matching."""

from jarvis_media_processor.video.analyze import match_media_visuals
from jarvis_media_processor.video.extract import (
    FFmpegNotFoundError,
    build_frame_extract_command,
    extract_sampled_frames,
)
from jarvis_media_processor.video.hashing import (
    compute_phash,
    distance_to_confidence,
    phash_distance,
)
from jarvis_media_processor.video.match import match_frames
from jarvis_media_processor.video.models import FrameCorrespondence, SampledFrame, VisualMatchReport
from jarvis_media_processor.video.preprocess import crop_letterbox, normalize_frame
from jarvis_media_processor.video.shots import detect_shot_boundaries, shot_boundary_times

__all__ = [
    "FFmpegNotFoundError",
    "FrameCorrespondence",
    "SampledFrame",
    "VisualMatchReport",
    "build_frame_extract_command",
    "compute_phash",
    "crop_letterbox",
    "detect_shot_boundaries",
    "distance_to_confidence",
    "extract_sampled_frames",
    "match_frames",
    "match_media_visuals",
    "normalize_frame",
    "phash_distance",
    "shot_boundary_times",
]
