"""``jmp job`` subcommands: submit, status, list, cancel."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.config import get_settings
from jarvis_media_processor.jobs.service import (
    cancel_job,
    get_job,
    list_jobs,
    submit_and_run,
)
from jarvis_media_processor.jobs.store import JobStoreError

job_app = typer.Typer(
    name="job",
    help="Standalone graft jobs (submit / status / list / cancel).",
    no_args_is_help=True,
)


@job_app.command("submit")
def job_submit(
    base: Annotated[Path, typer.Argument(help="Base media path.")],
    donor: Annotated[Path, typer.Argument(help="Donor media path.")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output MKV path (must not overwrite inputs)."),
    ],
    no_video: Annotated[
        bool,
        typer.Option("--no-video", help="Skip visual matching during align."),
    ] = False,
    skip_fine_warp: Annotated[
        bool,
        typer.Option("--skip-fine-warp", help="Mux reconstructed WAV without fine-warp."),
    ] = False,
    idempotency_key: Annotated[
        str | None,
        typer.Option(
            "--idempotency-key",
            help="Reuse an in-flight or succeeded job with this key.",
        ),
    ] = None,
    no_auto_idempotency: Annotated[
        bool,
        typer.Option(
            "--no-auto-idempotency",
            help="Do not derive an idempotency key from paths/options.",
        ),
    ] = False,
    cleanup: Annotated[
        bool,
        typer.Option(
            "--cleanup",
            help="Delete bulky workspace media after success.",
        ),
    ] = False,
    donor_language: Annotated[
        str,
        typer.Option("--donor-language", help="Language tag for donor track."),
    ] = "spa",
    donor_title: Annotated[
        str,
        typer.Option("--donor-title", help="Track title for donor audio."),
    ] = "Spanish (reconstructed)",
) -> None:
    """Submit a graft job and run it in-process (status persisted on disk)."""
    settings = get_settings()
    try:
        record = submit_and_run(
            base,
            donor,
            output,
            use_video=not no_video,
            fine_warp=not skip_fine_warp,
            donor_language=donor_language,
            donor_title=donor_title,
            idempotency_key=idempotency_key,
            auto_idempotency=not no_auto_idempotency,
            cleanup_workspace_on_success=cleanup or None,
            settings=settings,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(record.model_dump_json(indent=2))
    if record.state.value == "failed":
        raise typer.Exit(code=1)


@job_app.command("status")
def job_status(
    job_id: Annotated[str, typer.Argument(help="Job id (UUID).")],
) -> None:
    """Print persisted job status JSON."""
    try:
        record = get_job(job_id)
    except JobStoreError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(record.model_dump_json(indent=2))


@job_app.command("list")
def job_list() -> None:
    """List jobs under JMP_WORK_DIR/jobs (newest first)."""
    jobs = list_jobs()
    if not jobs:
        typer.echo("[]")
        return
    summary = [
        {
            "id": j.id,
            "state": j.state.value,
            "progress": j.progress.model_dump(),
            "output": str(j.params.output_path),
            "updated_at": j.updated_at.isoformat(),
        }
        for j in jobs
    ]
    import json

    typer.echo(json.dumps(summary, indent=2))


@job_app.command("cancel")
def job_cancel(
    job_id: Annotated[str, typer.Argument(help="Job id (UUID).")],
) -> None:
    """Request cancel (immediate if pending; cooperative if running)."""
    try:
        record = cancel_job(job_id)
    except JobStoreError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(record.model_dump_json(indent=2))
