"""Batch enqueue / skip / retry orchestration over the job store."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from jarvis_media_processor.batch.models import (
    BatchItemOutcome,
    BatchItemResult,
    BatchSummary,
    MediaPair,
    PairDisposition,
)
from jarvis_media_processor.batch.pairing import discover_pairs
from jarvis_media_processor.batch.store import BatchStore
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.jobs.models import JobRecord, JobState
from jarvis_media_processor.jobs.pipeline import default_idempotency_key
from jarvis_media_processor.jobs.service import submit_and_run
from jarvis_media_processor.jobs.store import JobStore
from jarvis_media_processor.output.mux import validate_output_path

logger = logging.getLogger(__name__)

JobRunner = Callable[..., JobRecord]


def classify_pair(
    pair: MediaPair,
    store: JobStore,
    *,
    retry: bool = False,
    use_video: bool = True,
    fine_warp: bool = True,
) -> tuple[PairDisposition, str, JobRecord | None]:
    """Decide run / skip / retry for one pair against the job store.

    Completed work is skipped by default. Failed/cancelled jobs are only
    re-queued when ``retry`` is True, so retries stay explicit.
    """
    key = default_idempotency_key(
        pair.base_path,
        pair.donor_path,
        pair.output_path,
        use_video=use_video,
        fine_warp=fine_warp,
    )
    existing = store.find_by_idempotency_key(key)
    if existing is None:
        return PairDisposition.RUN, "New pair; no prior job.", None

    if existing.state == JobState.SUCCEEDED:
        return (
            PairDisposition.SKIP_COMPLETED,
            f"Already succeeded as job {existing.id}.",
            existing,
        )
    if existing.state in {JobState.PENDING, JobState.RUNNING}:
        return (
            PairDisposition.SKIP_IN_FLIGHT,
            f"Job {existing.id} is {existing.state.value}.",
            existing,
        )
    if existing.state in {JobState.FAILED, JobState.CANCELLED}:
        if retry:
            return (
                PairDisposition.RETRY,
                f"Retrying after {existing.state.value} job {existing.id}.",
                existing,
            )
        return (
            PairDisposition.SKIP_FAILED,
            f"Prior job {existing.id} {existing.state.value}; pass --retry to re-run.",
            existing,
        )
    return PairDisposition.RUN, "No matching skip rule.", existing


def resolve_batch_dirs(
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    settings: Settings | None = None,
) -> tuple[Path, Path]:
    """Resolve input/output roots from args or ``JMP_BATCH_*`` settings."""
    cfg = settings or get_settings()
    resolved_input = Path(input_dir) if input_dir is not None else cfg.batch_input_dir
    resolved_output = Path(output_dir) if output_dir is not None else cfg.batch_output_dir
    if resolved_input is None:
        raise ValueError("Input directory required (pass a path or set JMP_BATCH_INPUT_DIR).")
    if resolved_output is None:
        raise ValueError(
            "Output directory required (pass --output-dir / -o or set JMP_BATCH_OUTPUT_DIR)."
        )
    return resolved_input, resolved_output


def run_batch(
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    retry: bool = False,
    dry_run: bool = False,
    use_video: bool = True,
    fine_warp: bool = True,
    donor_language: str = "spa",
    donor_title: str = "Spanish (reconstructed)",
    cleanup_workspace_on_success: bool | None = None,
    settings: Settings | None = None,
    job_store: JobStore | None = None,
    batch_store: BatchStore | None = None,
    job_runner: JobRunner | None = None,
) -> BatchSummary:
    """Discover pairs, classify skip/retry, optionally run jobs, persist summary.

    Sources are never overwritten. Output paths are deterministic under the
    output root. Failed jobs are not reprocessed unless ``retry`` is True.
    """
    cfg = settings or get_settings()
    in_dir, out_dir = resolve_batch_dirs(input_dir=input_dir, output_dir=output_dir, settings=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional batch temp root: jobs + batch summaries under that tree instead of
    # the global JMP_WORK_DIR (keeps overnight batch scratch separable).
    if job_store is not None:
        jobs = job_store
    elif cfg.batch_temp_dir is not None:
        jobs = JobStore(Path(cfg.batch_temp_dir) / "jobs", settings=cfg)
    else:
        jobs = JobStore(settings=cfg)

    if batch_store is not None:
        batches = batch_store
    elif cfg.batch_temp_dir is not None:
        batches = BatchStore(Path(cfg.batch_temp_dir) / "batches", settings=cfg)
    else:
        batches = BatchStore(settings=cfg)

    runner = job_runner or submit_and_run

    pairs = discover_pairs(in_dir, out_dir)
    summary = BatchSummary(
        id=batches.new_id(),
        input_dir=in_dir,
        output_dir=out_dir,
        dry_run=dry_run,
        retry=retry,
    )

    for pair in pairs:
        _ensure_safe_output(pair)
        disposition, reason, prior = classify_pair(
            pair,
            jobs,
            retry=retry,
            use_video=use_video,
            fine_warp=fine_warp,
        )

        if disposition in {
            PairDisposition.SKIP_COMPLETED,
            PairDisposition.SKIP_IN_FLIGHT,
            PairDisposition.SKIP_FAILED,
        }:
            summary.items.append(
                BatchItemResult(
                    pair_id=pair.pair_id,
                    disposition=disposition,
                    outcome=BatchItemOutcome.SKIPPED,
                    reason=reason,
                    job_id=prior.id if prior else None,
                    base_path=pair.base_path,
                    donor_path=pair.donor_path,
                    output_path=pair.output_path,
                    layout=pair.layout,
                )
            )
            continue

        if dry_run:
            summary.items.append(
                BatchItemResult(
                    pair_id=pair.pair_id,
                    disposition=disposition,
                    outcome=BatchItemOutcome.PLANNED,
                    reason=reason,
                    job_id=prior.id if prior else None,
                    base_path=pair.base_path,
                    donor_path=pair.donor_path,
                    output_path=pair.output_path,
                    layout=pair.layout,
                )
            )
            continue

        logger.info(
            "Batch %s: %s pair %s (%s)",
            summary.id,
            disposition.value,
            pair.pair_id,
            reason,
        )
        try:
            record = runner(
                pair.base_path,
                pair.donor_path,
                pair.output_path,
                use_video=use_video,
                fine_warp=fine_warp,
                donor_language=donor_language,
                donor_title=donor_title,
                auto_idempotency=True,
                cleanup_workspace_on_success=cleanup_workspace_on_success,
                settings=cfg,
                store=jobs,
            )
        except Exception as exc:
            logger.exception("Batch pair %s failed to start", pair.pair_id)
            summary.items.append(
                BatchItemResult(
                    pair_id=pair.pair_id,
                    disposition=disposition,
                    outcome=BatchItemOutcome.FAILED,
                    reason=str(exc),
                    job_id=None,
                    base_path=pair.base_path,
                    donor_path=pair.donor_path,
                    output_path=pair.output_path,
                    layout=pair.layout,
                )
            )
            continue

        if record.state == JobState.SUCCEEDED:
            outcome = BatchItemOutcome.SUCCEEDED
            item_reason = reason
        elif record.state == JobState.CANCELLED:
            outcome = BatchItemOutcome.FAILED
            item_reason = "Job cancelled."
        else:
            outcome = BatchItemOutcome.FAILED
            item_reason = record.error.message if record.error is not None else reason

        summary.items.append(
            BatchItemResult(
                pair_id=pair.pair_id,
                disposition=disposition,
                outcome=outcome,
                reason=item_reason,
                job_id=record.id,
                base_path=pair.base_path,
                donor_path=pair.donor_path,
                output_path=pair.output_path,
                layout=pair.layout,
            )
        )

    batches.save(summary)
    return summary


def _ensure_safe_output(pair: MediaPair) -> None:
    """Refuse outputs that would overwrite sources."""
    validate_output_path(pair.output_path, pair.base_path, pair.donor_path)
