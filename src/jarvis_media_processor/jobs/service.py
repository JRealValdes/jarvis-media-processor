"""High-level job submit / query API used by the CLI."""

from __future__ import annotations

from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.jobs.models import JobParams, JobRecord, JobState
from jarvis_media_processor.jobs.pipeline import default_idempotency_key, run_graft_pipeline
from jarvis_media_processor.jobs.store import JobStore


def submit_and_run(
    base: Path,
    donor: Path,
    output: Path,
    *,
    use_video: bool = True,
    fine_warp: bool = True,
    donor_language: str = "spa",
    donor_title: str = "Spanish (reconstructed)",
    idempotency_key: str | None = None,
    auto_idempotency: bool = True,
    cleanup_workspace_on_success: bool | None = None,
    settings: Settings | None = None,
    store: JobStore | None = None,
) -> JobRecord:
    """Create a job (honoring idempotency) and run the graft pipeline."""
    cfg = settings or get_settings()
    job_store = store or JobStore(settings=cfg)
    key = idempotency_key
    if key is None and auto_idempotency:
        key = default_idempotency_key(
            base, donor, output, use_video=use_video, fine_warp=fine_warp
        )
    params = JobParams(
        base_path=Path(base),
        donor_path=Path(donor),
        output_path=Path(output),
        use_video=use_video,
        fine_warp=fine_warp,
        donor_language=donor_language,
        donor_title=donor_title,
        idempotency_key=key,
    )
    record = job_store.create(params)
    if record.is_terminal():
        return record
    if record.state == JobState.RUNNING:
        # Another process may own it; just return current snapshot.
        return record

    cleanup = (
        cfg.jobs_cleanup_workspace_on_success
        if cleanup_workspace_on_success is None
        else cleanup_workspace_on_success
    )
    return run_graft_pipeline(
        record,
        job_store,
        settings=cfg,
        cleanup_workspace_on_success=cleanup,
    )


def get_job(job_id: str, *, store: JobStore | None = None) -> JobRecord:
    """Load job status."""
    return (store or JobStore()).load(job_id)


def list_jobs(*, store: JobStore | None = None) -> list[JobRecord]:
    """List known jobs."""
    return (store or JobStore()).list_jobs()


def cancel_job(job_id: str, *, store: JobStore | None = None) -> JobRecord:
    """Request cancellation (cooperative if running)."""
    return (store or JobStore()).request_cancel(job_id)
