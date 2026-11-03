"""Video frame extraction and visual matching."""

from jarvis_media_processor.video.activity import (
    ActivityEnvelope,
    activity_peak_times,
    build_activity_envelope,
)
from jarvis_media_processor.video.analyze import match_media_visuals
from jarvis_media_processor.video.confirm import (
    frame_distance_at,
    frames_match_at,
    make_frame_confirmer,
    nearest_frame,
)
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
from jarvis_media_processor.video.match import match_frames, match_frames_two_phase
from jarvis_media_processor.video.models import FrameCorrespondence, SampledFrame, VisualMatchReport
from jarvis_media_processor.video.preprocess import crop_letterbox, normalize_frame
from jarvis_media_processor.video.shots import detect_shot_boundaries, shot_boundary_times

__all__ = [
    "ActivityEnvelope",
    "FFmpegNotFoundError",
    "FrameCorrespondence",
    "SampledFrame",
    "VisualMatchReport",
    "activity_peak_times",
    "build_activity_envelope",
    "build_frame_extract_command",
    "compute_phash",
    "crop_letterbox",
    "detect_shot_boundaries",
    "distance_to_confidence",
    "extract_sampled_frames",
    "frame_distance_at",
    "frames_match_at",
    "make_frame_confirmer",
    "match_frames",
    "match_frames_two_phase",
    "match_media_visuals",
    "nearest_frame",
    "normalize_frame",
    "phash_distance",
    "shot_boundary_times",
]
