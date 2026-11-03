"""``jmp batch`` subcommands: discover, run, status, list."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.batch.pairing import discover_pairs
from jarvis_media_processor.batch.service import resolve_batch_dirs, run_batch
from jarvis_media_processor.batch.store import BatchStore, BatchStoreError
from jarvis_media_processor.config import get_settings

batch_app = typer.Typer(
    name="batch",
    help="Folder-based batch discovery and graft enqueue.",
    no_args_is_help=True,
)


@batch_app.command("discover")
def batch_discover(
    input_dir: Annotated[
        Path | None,
        typer.Argument(help="Input root (default: JMP_BATCH_INPUT_DIR)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output root for deterministic paths (default: JMP_BATCH_OUTPUT_DIR).",
        ),
    ] = None,
) -> None:
    """List base/donor pairs discovered under a folder layout (no jobs created)."""
    try:
        in_dir, out_dir = resolve_batch_dirs(input_dir=input_dir, output_dir=output_dir)
        pairs = discover_pairs(in_dir, out_dir)
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    payload = [
        {
            "pair_id": p.pair_id,
            "layout": p.layout.value,
            "base": str(p.base_path),
            "donor": str(p.donor_path),
            "output": str(p.output_path),
        }
        for p in pairs
    ]
    typer.echo(json.dumps(payload, indent=2))


@batch_app.command("run")
def batch_run(
    input_dir: Annotated[
        Path | None,
        typer.Argument(help="Input root (default: JMP_BATCH_INPUT_DIR)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output root (default: JMP_BATCH_OUTPUT_DIR).",
        ),
    ] = None,
    retry: Annotated[
        bool,
        typer.Option(
            "--retry",
            help="Re-queue pairs whose prior job failed or was cancelled.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Classify pairs only; do not create or run jobs.",
        ),
    ] = False,
    no_video: Annotated[
        bool,
        typer.Option("--no-video", help="Skip visual matching during align."),
    ] = False,
    skip_fine_warp: Annotated[
        bool,
        typer.Option("--skip-fine-warp", help="Mux reconstructed WAV without fine-warp."),
    ] = False,
    cleanup: Annotated[
        bool,
        typer.Option(
            "--cleanup",
            help="Delete bulky workspace media after each successful job.",
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
    """Enqueue/run a batch from a folder layout; print a success/failure/skipped summary."""
    settings = get_settings()
    try:
        summary = run_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            retry=retry,
            dry_run=dry_run,
            use_video=not no_video,
            fine_warp=not skip_fine_warp,
            donor_language=donor_language,
            donor_title=donor_title,
            cleanup_workspace_on_success=cleanup or None,
            settings=settings,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "id": summary.id,
                "dry_run": summary.dry_run,
                "retry": summary.retry,
                "input_dir": str(summary.input_dir),
                "output_dir": str(summary.output_dir),
                "counts": summary.counts(),
                "items": [
                    {
                        "pair_id": i.pair_id,
                        "disposition": i.disposition.value,
                        "outcome": i.outcome.value,
                        "reason": i.reason,
                        "job_id": i.job_id,
                        "output": str(i.output_path),
                    }
                    for i in summary.items
                ],
            },
            indent=2,
        )
    )
    if summary.failed > 0:
        raise typer.Exit(code=1)


@batch_app.command("status")
def batch_status(
    batch_id: Annotated[str, typer.Argument(help="Batch id (UUID).")],
) -> None:
    """Print a persisted batch summary JSON."""
    try:
        summary = BatchStore().load(batch_id)
    except BatchStoreError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    payload = json.loads(summary.model_dump_json())
    payload["counts"] = summary.counts()
    typer.echo(json.dumps(payload, indent=2))


@batch_app.command("list")
def batch_list() -> None:
    """List batches under JMP_WORK_DIR/batches (newest first)."""
    batches = BatchStore().list_batches()
    if not batches:
        typer.echo("[]")
        return
    payload = [
        {
            "id": b.id,
            "created_at": b.created_at.isoformat(),
            "input_dir": str(b.input_dir),
            "output_dir": str(b.output_dir),
            "dry_run": b.dry_run,
            "counts": b.counts(),
        }
        for b in batches
    ]
    typer.echo(json.dumps(payload, indent=2))
