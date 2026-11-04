"""``jmp schedule`` subcommands: tick, loop, show, plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.batch.service import resolve_batch_dirs
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.schedule.service import (
    plan_scheduled_tick,
    run_scheduled_loop,
    run_scheduled_tick,
)

schedule_app = typer.Typer(
    name="schedule",
    help=(
        "Unattended batch ticks. Prefer OS schedulers invoking ``tick``; "
        "``loop`` is an optional in-process interval runner."
    ),
    no_args_is_help=True,
)


@schedule_app.command("show")
def schedule_show() -> None:
    """Print resolved schedule/batch settings for unattended setups."""
    settings = get_settings()
    payload: dict[str, object] = {
        "schedule_interval_seconds": settings.schedule_interval_seconds,
        "schedule_retry_failed": settings.schedule_retry_failed,
        "batch_input_dir": (str(settings.batch_input_dir) if settings.batch_input_dir else None),
        "batch_output_dir": (str(settings.batch_output_dir) if settings.batch_output_dir else None),
        "batch_temp_dir": (str(settings.batch_temp_dir) if settings.batch_temp_dir else None),
        "work_dir": str(settings.work_dir),
        "log_level": settings.log_level,
        "preferred_entry": "jmp schedule tick",
        "optional_loop": "jmp schedule loop --interval <seconds>",
        "docs": "docs/scheduled-execution.md",
    }
    try:
        in_dir, out_dir = resolve_batch_dirs(settings=settings)
        payload["resolved_input_dir"] = str(in_dir)
        payload["resolved_output_dir"] = str(out_dir)
    except ValueError as exc:
        payload["resolve_error"] = str(exc)
    typer.echo(json.dumps(payload, indent=2))


@schedule_app.command("plan")
def schedule_plan(
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
            help="Include failed/cancelled pairs in the process set.",
        ),
    ] = False,
    no_video: Annotated[
        bool,
        typer.Option("--no-video", help="Skip visual matching during align."),
    ] = False,
    skip_fine_warp: Annotated[
        bool,
        typer.Option("--skip-fine-warp", help="Mux without fine-warp."),
    ] = False,
) -> None:
    """Show dedupe plan for the next tick (no jobs created)."""
    try:
        plan = plan_scheduled_tick(
            input_dir=input_dir,
            output_dir=output_dir,
            retry=retry,
            use_video=not no_video,
            fine_warp=not skip_fine_warp,
        )
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "counts": plan.counts(),
                "items": [
                    {
                        "pair_id": p.pair.pair_id,
                        "action": p.action.value,
                        "disposition": p.disposition.value,
                        "reason": p.reason,
                        "prior_job_id": p.prior_job_id,
                        "output": str(p.pair.output_path),
                    }
                    for p in plan.pairs
                ],
            },
            indent=2,
        )
    )


@schedule_app.command("tick")
def schedule_tick(
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
        bool | None,
        typer.Option(
            "--retry/--no-retry",
            help="Override JMP_SCHEDULE_RETRY_FAILED for this tick.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Classify only; do not run jobs."),
    ] = False,
    no_video: Annotated[
        bool,
        typer.Option("--no-video", help="Skip visual matching during align."),
    ] = False,
    skip_fine_warp: Annotated[
        bool,
        typer.Option("--skip-fine-warp", help="Mux without fine-warp."),
    ] = False,
    cleanup: Annotated[
        bool,
        typer.Option("--cleanup", help="Cleanup bulky workspace media on success."),
    ] = False,
) -> None:
    """Run one scheduled batch tick (for cron / Task Scheduler / systemd)."""
    settings = get_settings()
    try:
        summary = run_scheduled_tick(
            input_dir=input_dir,
            output_dir=output_dir,
            retry=retry,
            dry_run=dry_run,
            use_video=not no_video,
            fine_warp=not skip_fine_warp,
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
                "counts": summary.counts(),
                "dry_run": summary.dry_run,
                "retry": summary.retry,
            },
            indent=2,
        )
    )
    if summary.failed > 0:
        raise typer.Exit(code=1)


@schedule_app.command("loop")
def schedule_loop(
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
    interval: Annotated[
        float | None,
        typer.Option(
            "--interval",
            help="Seconds between ticks (default: JMP_SCHEDULE_INTERVAL_SECONDS).",
        ),
    ] = None,
    max_ticks: Annotated[
        int | None,
        typer.Option(
            "--max-ticks",
            help="Stop after N ticks (default: run until interrupted).",
        ),
    ] = None,
    retry: Annotated[
        bool | None,
        typer.Option(
            "--retry/--no-retry",
            help="Override JMP_SCHEDULE_RETRY_FAILED.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Classify only; do not run jobs."),
    ] = False,
    no_video: Annotated[
        bool,
        typer.Option("--no-video", help="Skip visual matching during align."),
    ] = False,
    skip_fine_warp: Annotated[
        bool,
        typer.Option("--skip-fine-warp", help="Mux without fine-warp."),
    ] = False,
    cleanup: Annotated[
        bool,
        typer.Option("--cleanup", help="Cleanup bulky workspace media on success."),
    ] = False,
) -> None:
    """Optional in-process interval loop (prefer OS schedulers when possible)."""
    settings = get_settings()
    try:
        summaries = run_scheduled_loop(
            interval_seconds=interval,
            max_ticks=max_ticks,
            input_dir=input_dir,
            output_dir=output_dir,
            retry=retry,
            dry_run=dry_run,
            use_video=not no_video,
            fine_warp=not skip_fine_warp,
            cleanup_workspace_on_success=cleanup or None,
            settings=settings,
        )
    except KeyboardInterrupt as exc:
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(code=130) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    failed_ticks = sum(1 for s in summaries if s.failed > 0)
    typer.echo(
        json.dumps(
            {
                "ticks": len(summaries),
                "failed_ticks": failed_ticks,
                "batch_ids": [s.id for s in summaries],
            },
            indent=2,
        )
    )
    if failed_ticks > 0:
        raise typer.Exit(code=1)
