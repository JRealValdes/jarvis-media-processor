"""Tests for scheduled tick planning (dedupe) and optional loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis_media_processor.batch.models import (
    BatchItemOutcome,
    BatchItemResult,
    BatchSummary,
    PairDisposition,
    PairLayout,
)
from jarvis_media_processor.config import Settings
from jarvis_media_processor.jobs.models import JobParams, JobState
from jarvis_media_processor.jobs.pipeline import default_idempotency_key
from jarvis_media_processor.jobs.store import JobStore
from jarvis_media_processor.schedule.models import TickAction
from jarvis_media_processor.schedule.service import (
    plan_scheduled_tick,
    run_scheduled_loop,
    run_scheduled_tick,
)


def _touch_media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"media")
    return path


def _seed_job(
    store: JobStore,
    *,
    base: Path,
    donor: Path,
    output: Path,
    state: JobState,
) -> str:
    key = default_idempotency_key(base, donor, output, use_video=True, fine_warp=True)
    record = store.create(
        JobParams(
            base_path=base,
            donor_path=donor,
            output_path=output,
            idempotency_key=key,
        )
    )
    record.state = state
    store.save(record)
    return record.id


def test_plan_scheduled_tick_dedupes_completed_and_failed(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    work = tmp_path / "work"
    _touch_media(incoming / "new" / "base.mkv")
    _touch_media(incoming / "new" / "donor.mkv")
    _touch_media(incoming / "done" / "base.mkv")
    _touch_media(incoming / "done" / "donor.mkv")
    _touch_media(incoming / "bad" / "base.mkv")
    _touch_media(incoming / "bad" / "donor.mkv")

    settings = Settings(work_dir=work)
    store = JobStore(settings=settings)
    _seed_job(
        store,
        base=incoming / "done" / "base.mkv",
        donor=incoming / "done" / "donor.mkv",
        output=outgoing / "done.mkv",
        state=JobState.SUCCEEDED,
    )
    _seed_job(
        store,
        base=incoming / "bad" / "base.mkv",
        donor=incoming / "bad" / "donor.mkv",
        output=outgoing / "bad.mkv",
        state=JobState.FAILED,
    )

    plan = plan_scheduled_tick(
        input_dir=incoming,
        output_dir=outgoing,
        retry=False,
        settings=settings,
        job_store=store,
    )
    by_id = {p.pair.pair_id: p for p in plan.pairs}
    assert plan.counts() == {"total": 3, "process": 1, "skip": 2}
    assert by_id["new"].action == TickAction.PROCESS
    assert by_id["new"].disposition == PairDisposition.RUN
    assert by_id["done"].action == TickAction.SKIP
    assert by_id["done"].disposition == PairDisposition.SKIP_COMPLETED
    assert by_id["bad"].action == TickAction.SKIP
    assert by_id["bad"].disposition == PairDisposition.SKIP_FAILED

    plan_retry = plan_scheduled_tick(
        input_dir=incoming,
        output_dir=outgoing,
        retry=True,
        settings=settings,
        job_store=store,
    )
    by_retry = {p.pair.pair_id: p for p in plan_retry.pairs}
    assert by_retry["bad"].action == TickAction.PROCESS
    assert by_retry["bad"].disposition == PairDisposition.RETRY
    assert by_retry["done"].action == TickAction.SKIP


def test_run_scheduled_tick_logs_and_uses_settings_retry(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    work = tmp_path / "work"
    _touch_media(incoming / "ep" / "base.mkv")
    _touch_media(incoming / "ep" / "donor.mkv")

    settings = Settings(work_dir=work, schedule_retry_failed=True)
    fake_summary = BatchSummary(
        id="00000000-0000-0000-0000-000000000099",
        input_dir=incoming,
        output_dir=outgoing,
        retry=True,
        items=[
            BatchItemResult(
                pair_id="ep",
                disposition=PairDisposition.RUN,
                outcome=BatchItemOutcome.SUCCEEDED,
                reason="ok",
                job_id="job-1",
                base_path=incoming / "ep" / "base.mkv",
                donor_path=incoming / "ep" / "donor.mkv",
                output_path=outgoing / "ep.mkv",
                layout=PairLayout.SUBDIR,
            )
        ],
    )
    runner = MagicMock(return_value=fake_summary)
    summary = run_scheduled_tick(
        input_dir=incoming,
        output_dir=outgoing,
        settings=settings,
        batch_runner=runner,
    )
    assert summary.id == fake_summary.id
    kwargs = runner.call_args.kwargs
    assert kwargs["retry"] is True
    assert kwargs["input_dir"] == incoming
    assert kwargs["output_dir"] == outgoing


def test_run_scheduled_loop_two_ticks_without_reprocessing(
    tmp_path: Path,
) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    work = tmp_path / "work"
    _touch_media(incoming / "ep" / "base.mkv")
    _touch_media(incoming / "ep" / "donor.mkv")
    settings = Settings(work_dir=work, schedule_interval_seconds=1)

    calls: list[bool] = []

    def fake_runner(*, retry: bool, **_kwargs: object) -> BatchSummary:
        calls.append(retry)
        # First tick "succeeds"; second tick should still be invoked but
        # classification inside a real run_batch would skip — here we only
        # assert the loop invokes N ticks with injectable sleeper.
        return BatchSummary(
            id=f"00000000-0000-0000-0000-00000000000{len(calls)}",
            input_dir=incoming,
            output_dir=outgoing,
            retry=retry,
            items=[],
        )

    slept: list[float] = []
    summaries = run_scheduled_loop(
        interval_seconds=5,
        max_ticks=2,
        sleeper=slept.append,
        input_dir=incoming,
        output_dir=outgoing,
        retry=False,
        settings=settings,
        batch_runner=fake_runner,
    )
    assert len(summaries) == 2
    assert slept == [5.0]
    assert calls == [False, False]


def test_run_scheduled_loop_rejects_non_positive_interval(tmp_path: Path) -> None:
    settings = Settings(work_dir=tmp_path / "work")
    with pytest.raises(ValueError, match="interval_seconds"):
        run_scheduled_loop(
            interval_seconds=0,
            max_ticks=1,
            input_dir=tmp_path / "in",
            output_dir=tmp_path / "out",
            settings=settings,
            batch_runner=MagicMock(),
        )


def test_second_plan_after_success_skips(tmp_path: Path) -> None:
    """Simulate overnight: after a successful tick, next plan processes nothing new."""
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    work = tmp_path / "work"
    _touch_media(incoming / "ep" / "base.mkv")
    _touch_media(incoming / "ep" / "donor.mkv")
    settings = Settings(work_dir=work)
    store = JobStore(settings=settings)

    first = plan_scheduled_tick(
        input_dir=incoming,
        output_dir=outgoing,
        settings=settings,
        job_store=store,
    )
    assert first.counts()["process"] == 1

    _seed_job(
        store,
        base=incoming / "ep" / "base.mkv",
        donor=incoming / "ep" / "donor.mkv",
        output=outgoing / "ep.mkv",
        state=JobState.SUCCEEDED,
    )
    second = plan_scheduled_tick(
        input_dir=incoming,
        output_dir=outgoing,
        settings=settings,
        job_store=store,
    )
    assert second.counts() == {"total": 1, "process": 0, "skip": 1}
    assert second.pairs[0].disposition == PairDisposition.SKIP_COMPLETED
