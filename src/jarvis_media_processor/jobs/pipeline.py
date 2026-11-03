"""End-to-end graft pipeline invoked by the job runner."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from jarvis_media_processor.alignment.detect import detect_alignment
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.jobs.models import JobError, JobProgress, JobRecord, JobState, utc_now
from jarvis_media_processor.jobs.store import JobStore
from jarvis_media_processor.media.inspect import inspect_media
from jarvis_media_processor.output.mux import mux_mkv, validate_output_path
from jarvis_media_processor.reconstruct.fine_warp_pipeline import fine_warp_reconstructed_audio
from jarvis_media_processor.reconstruct.pipeline import reconstruct_donor_audio

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[JobProgress], None]


class JobCancelled(Exception):
    """Raised when a cooperative cancel is observed between stages."""


class JobPipelineError(RuntimeError):
    """Pipeline stage failure with a stable error code."""

    def __init__(self, code: str, message: str, *, stage: str) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage


def _check_cancel(store: JobStore, job_id: str) -> None:
    if store.is_cancel_requested(job_id):
        raise JobCancelled("Cancel requested")


def run_graft_pipeline(
    record: JobRecord,
    store: JobStore,
    *,
    settings: Settings | None = None,
    on_progress: ProgressCallback | None = None,
    cleanup_workspace_on_success: bool = False,
) -> JobRecord:
    """Run align → reconstruct → [fine-warp] → mux for ``record``.

    Updates the store after each stage. Sources are never overwritten.
    """
    cfg = settings or get_settings()
    job_id = record.id
    params = record.params

    def emit(percent: float, stage: str, message: str) -> None:
        progress = JobProgress(percent=percent, stage=stage, message=message)
        record.progress = progress
        record.touch()
        store.save(record)
        if on_progress is not None:
            on_progress(progress)

    try:
        validate_output_path(params.output_path, params.base_path, params.donor_path)
    except Exception as exc:
        raise JobPipelineError("unsafe_output", str(exc), stage="validate") from exc

    if not params.base_path.is_file():
        raise JobPipelineError(
            "base_missing",
            f"Base media not found: {params.base_path}",
            stage="validate",
        )
    if not params.donor_path.is_file():
        raise JobPipelineError(
            "donor_missing",
            f"Donor media not found: {params.donor_path}",
            stage="validate",
        )

    work = Path(record.workspace or (store.job_dir(job_id) / "workspace"))
    work.mkdir(parents=True, exist_ok=True)
    record.workspace = work
    record.state = JobState.RUNNING
    record.started_at = record.started_at or utc_now()
    store.save(record)

    try:
        _check_cancel(store, job_id)
        emit(5.0, "inspect", "Reading base duration")
        inspection = inspect_media(params.base_path, settings=cfg)
        base_duration = inspection.duration_seconds
        if base_duration is None or base_duration <= 0:
            raise JobPipelineError(
                "base_duration",
                "Could not determine base media duration.",
                stage="inspect",
            )
        record.notes.append(f"base_duration={base_duration:.3f}s")
        store.save(record)

        _check_cancel(store, job_id)
        emit(15.0, "align", "Detecting alignment")
        align_dir = work / "align"
        align_dir.mkdir(parents=True, exist_ok=True)
        alignment = detect_alignment(
            params.base_path,
            params.donor_path,
            align_dir,
            settings=cfg,
            use_video=params.use_video,
        )
        alignment_path = align_dir / "alignment.json"
        alignment_path.write_text(
            alignment.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        record.artifacts["alignment_json"] = alignment_path
        store.save(record)

        _check_cancel(store, job_id)
        emit(45.0, "reconstruct", "Reconstructing donor audio on base timeline")
        recon_dir = work / "reconstruct"
        recon = reconstruct_donor_audio(
            params.donor_path,
            alignment,
            recon_dir,
            base_duration=float(base_duration),
            settings=cfg,
        )
        record.artifacts["reconstructed_wav"] = recon.output_wav
        if recon.output_wav.with_name("processing_report.json").is_file():
            record.artifacts["processing_report"] = recon.output_wav.with_name(
                "processing_report.json"
            )
        store.save(record)

        donor_audio = recon.output_wav
        if params.fine_warp:
            _check_cancel(store, job_id)
            emit(70.0, "fine_warp", "Fine-warping reconstructed audio vs base track 0")
            fine_dir = work / "fine-warp"
            fine = fine_warp_reconstructed_audio(
                params.base_path,
                recon.output_wav,
                fine_dir,
                settings=cfg,
            )
            donor_audio = fine.output_wav
            record.artifacts["fine_warped_wav"] = fine.output_wav
            if fine.plan_json is not None:
                record.artifacts["fine_warp_plan"] = fine.plan_json
            store.save(record)
        else:
            record.notes.append("fine_warp skipped by job params")

        _check_cancel(store, job_id)
        emit(90.0, "mux", "Muxing output MKV")
        out = mux_mkv(
            params.base_path,
            donor_audio,
            params.output_path,
            settings=cfg,
            donor_language=params.donor_language,
            donor_title=params.donor_title,
        )
        record.artifacts["output_mkv"] = out
        store.save(record)

        if cleanup_workspace_on_success:
            _cleanup_intermediates(work)
            record.notes.append("Cleaned intermediate workspace files after success.")

        emit(100.0, "done", "Job succeeded")
        record.state = JobState.SUCCEEDED
        record.finished_at = utc_now()
        store.save(record)
        return record

    except JobCancelled:
        record.state = JobState.CANCELLED
        record.finished_at = utc_now()
        record.progress = JobProgress(
            percent=record.progress.percent,
            stage=record.progress.stage,
            message="Cancelled",
        )
        record.notes.append("Cancelled cooperatively between stages.")
        store.save(record)
        return record
    except JobPipelineError as exc:
        record.state = JobState.FAILED
        record.finished_at = utc_now()
        record.error = JobError(code=exc.code, message=str(exc), stage=exc.stage)
        record.progress.message = str(exc)
        store.save(record)
        return record
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        record.state = JobState.FAILED
        record.finished_at = utc_now()
        record.error = JobError(
            code="internal",
            message=str(exc),
            stage=record.progress.stage or None,
        )
        record.progress.message = str(exc)
        store.save(record)
        return record


def _cleanup_intermediates(workspace: Path) -> None:
    """Remove bulky intermediates; keep JSON plans/reports if present."""
    for name in ("align", "reconstruct", "fine-warp"):
        path = workspace / name
        if not path.is_dir():
            continue
        for pattern in ("*.wav", "*.png"):
            for item in path.rglob(pattern):
                try:
                    item.unlink(missing_ok=True)
                except OSError:
                    pass
        # Drop empty frame dirs if any
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass


def default_idempotency_key(
    base: Path,
    donor: Path,
    output: Path,
    *,
    use_video: bool,
    fine_warp: bool,
) -> str:
    """Stable key for identical graft submissions."""
    parts = [
        str(Path(base).resolve()),
        str(Path(donor).resolve()),
        str(Path(output).resolve()),
        f"video={int(use_video)}",
        f"fine={int(fine_warp)}",
    ]
    return "|".join(parts)
