"""Scheduled batch ticks: plan (dedupe) + run + optional interval loop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from jarvis_media_processor.batch.models import BatchSummary, PairDisposition
from jarvis_media_processor.batch.pairing import discover_pairs
from jarvis_media_processor.batch.service import classify_pair, resolve_batch_dirs, run_batch
from jarvis_media_processor.batch.store import BatchStore
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.jobs.store import JobStore
from jarvis_media_processor.schedule.models import PlannedPair, TickAction, TickPlan

logger = logging.getLogger(__name__)

Sleeper = Callable[[float], None]
BatchRunner = Callable[..., BatchSummary]


_PROCESS_DISPOSITIONS = frozenset({PairDisposition.RUN, PairDisposition.RETRY})


def plan_scheduled_tick(
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    retry: bool = False,
    use_video: bool = True,
    fine_warp: bool = True,
    settings: Settings | None = None,
    job_store: JobStore | None = None,
) -> TickPlan:
    """Discover pairs and classify skip/process without creating jobs.

    Deduplication uses the job store idempotency index: succeeded and in-flight
    pairs are skipped; failed/cancelled pairs process only when ``retry`` is True.
    """
    cfg = settings or get_settings()
    in_dir, out_dir = resolve_batch_dirs(input_dir=input_dir, output_dir=output_dir, settings=cfg)
    jobs = job_store or _default_job_store(cfg)

    planned: list[PlannedPair] = []
    for pair in discover_pairs(in_dir, out_dir):
        disposition, reason, prior = classify_pair(
            pair,
            jobs,
            retry=retry,
            use_video=use_video,
            fine_warp=fine_warp,
        )
        action = TickAction.PROCESS if disposition in _PROCESS_DISPOSITIONS else TickAction.SKIP
        planned.append(
            PlannedPair(
                pair=pair,
                disposition=disposition,
                action=action,
                reason=reason,
                prior_job_id=prior.id if prior else None,
            )
        )
    return TickPlan(pairs=planned)


def run_scheduled_tick(
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    retry: bool | None = None,
    dry_run: bool = False,
    use_video: bool = True,
    fine_warp: bool = True,
    donor_language: str = "spa",
    donor_title: str = "Spanish (reconstructed)",
    cleanup_workspace_on_success: bool | None = None,
    settings: Settings | None = None,
    job_store: JobStore | None = None,
    batch_store: BatchStore | None = None,
    batch_runner: BatchRunner | None = None,
) -> BatchSummary:
    """Run one unattended batch tick with start/end summary logging.

    Intended entry point for OS schedulers (cron / Task Scheduler / systemd).
    Completed work is not reprocessed; failures remain in the job store for
    later ``--retry`` / ``JMP_SCHEDULE_RETRY_FAILED``.
    """
    cfg = settings or get_settings()
    do_retry = cfg.schedule_retry_failed if retry is None else retry
    in_dir, out_dir = resolve_batch_dirs(input_dir=input_dir, output_dir=output_dir, settings=cfg)

    plan = plan_scheduled_tick(
        input_dir=in_dir,
        output_dir=out_dir,
        retry=do_retry,
        use_video=use_video,
        fine_warp=fine_warp,
        settings=cfg,
        job_store=job_store,
    )
    counts = plan.counts()
    logger.info(
        "schedule.tick.start input=%s output=%s retry=%s dry_run=%s "
        "pairs_total=%s process=%s skip=%s",
        in_dir,
        out_dir,
        do_retry,
        dry_run,
        counts["total"],
        counts["process"],
        counts["skip"],
    )
    for planned in plan.to_skip:
        logger.info(
            "schedule.tick.skip pair_id=%s disposition=%s reason=%s prior_job=%s",
            planned.pair.pair_id,
            planned.disposition.value,
            planned.reason,
            planned.prior_job_id,
        )
    for planned in plan.to_process:
        logger.info(
            "schedule.tick.queue pair_id=%s disposition=%s reason=%s",
            planned.pair.pair_id,
            planned.disposition.value,
            planned.reason,
        )

    runner = batch_runner or run_batch
    summary = runner(
        input_dir=in_dir,
        output_dir=out_dir,
        retry=do_retry,
        dry_run=dry_run,
        use_video=use_video,
        fine_warp=fine_warp,
        donor_language=donor_language,
        donor_title=donor_title,
        cleanup_workspace_on_success=cleanup_workspace_on_success,
        settings=cfg,
        job_store=job_store,
        batch_store=batch_store,
    )

    result_counts = summary.counts()
    logger.info(
        "schedule.tick.end batch_id=%s succeeded=%s failed=%s skipped=%s planned=%s",
        summary.id,
        result_counts["succeeded"],
        result_counts["failed"],
        result_counts["skipped"],
        result_counts["planned"],
    )
    for result in summary.items:
        if result.outcome.value == "failed":
            logger.error(
                "schedule.tick.failure pair_id=%s job_id=%s reason=%s",
                result.pair_id,
                result.job_id,
                result.reason,
            )
    return summary


def run_scheduled_loop(
    *,
    interval_seconds: float | None = None,
    max_ticks: int | None = None,
    sleeper: Sleeper | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    retry: bool | None = None,
    dry_run: bool = False,
    use_video: bool = True,
    fine_warp: bool = True,
    donor_language: str = "spa",
    donor_title: str = "Spanish (reconstructed)",
    cleanup_workspace_on_success: bool | None = None,
    settings: Settings | None = None,
    job_store: JobStore | None = None,
    batch_store: BatchStore | None = None,
    batch_runner: BatchRunner | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[BatchSummary]:
    """Optional in-process loop: run ticks forever (or ``max_ticks``) with sleep.

    Prefer OS schedulers for production overnight runs; use this loop only when
    installing cron/Task Scheduler is inconvenient. ``sleeper`` and ``should_stop``
    are injectable for unit tests.
    """
    cfg = settings or get_settings()
    interval = (
        float(cfg.schedule_interval_seconds)
        if interval_seconds is None
        else float(interval_seconds)
    )
    if interval <= 0:
        raise ValueError("interval_seconds must be > 0")

    sleep_fn = sleeper or time.sleep
    stop = should_stop or (lambda: False)
    summaries: list[BatchSummary] = []
    tick = 0

    logger.info(
        "schedule.loop.start interval_seconds=%s max_ticks=%s",
        interval,
        max_ticks,
    )
    while not stop():
        tick += 1
        logger.info("schedule.loop.tick n=%s", tick)
        summary = run_scheduled_tick(
            input_dir=input_dir,
            output_dir=output_dir,
            retry=retry,
            dry_run=dry_run,
            use_video=use_video,
            fine_warp=fine_warp,
            donor_language=donor_language,
            donor_title=donor_title,
            cleanup_workspace_on_success=cleanup_workspace_on_success,
            settings=cfg,
            job_store=job_store,
            batch_store=batch_store,
            batch_runner=batch_runner,
        )
        summaries.append(summary)
        if max_ticks is not None and tick >= max_ticks:
            break
        if stop():
            break
        logger.info("schedule.loop.sleep seconds=%s", interval)
        sleep_fn(interval)

    logger.info("schedule.loop.end ticks=%s", len(summaries))
    return summaries


def _default_job_store(cfg: Settings) -> JobStore:
    if cfg.batch_temp_dir is not None:
        return JobStore(Path(cfg.batch_temp_dir) / "jobs", settings=cfg)
    return JobStore(settings=cfg)
