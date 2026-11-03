"""Tests for batch pairing rules and skip/retry classification."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis_media_processor.batch.models import (
    BatchItemOutcome,
    PairDisposition,
    PairLayout,
)
from jarvis_media_processor.batch.pairing import (
    discover_pairs,
    output_path_for_pair,
)
from jarvis_media_processor.batch.service import classify_pair, run_batch
from jarvis_media_processor.batch.store import BatchStore
from jarvis_media_processor.jobs.models import JobParams, JobState
from jarvis_media_processor.jobs.pipeline import default_idempotency_key
from jarvis_media_processor.jobs.store import JobStore


def _touch_media(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"media")
    return path


def test_output_path_for_pair_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid pair id"):
        output_path_for_pair(tmp_path, "../escape")


def test_discover_subdir_pairs(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    _touch_media(incoming / "ep01" / "base.mkv")
    _touch_media(incoming / "ep01" / "donor.mp4")
    _touch_media(incoming / "ep02" / "Base.MKV")
    _touch_media(incoming / "ep02" / "Donor.mkv")
    # Incomplete pair ignored
    _touch_media(incoming / "ep03" / "base.mkv")

    pairs = discover_pairs(incoming, outgoing)
    assert [p.pair_id for p in pairs] == ["ep01", "ep02"]
    assert pairs[0].layout == PairLayout.SUBDIR
    assert pairs[0].output_path == outgoing / "ep01.mkv"
    assert pairs[0].donor_path.suffix == ".mp4"


def test_discover_flat_pairs(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    _touch_media(incoming / "show.s01e01.base.mkv")
    _touch_media(incoming / "show.s01e01.donor.mkv")
    _touch_media(incoming / "orphan.base.mkv")

    pairs = discover_pairs(incoming, outgoing)
    assert len(pairs) == 1
    assert pairs[0].pair_id == "show.s01e01"
    assert pairs[0].layout == PairLayout.FLAT
    assert pairs[0].output_path == outgoing / "show.s01e01.mkv"


def test_discover_split_pairs_nested(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    _touch_media(incoming / "base" / "show" / "s01e01.mkv")
    _touch_media(incoming / "donor" / "show" / "s01e01.mkv")
    _touch_media(incoming / "base" / "only_base.mkv")

    pairs = discover_pairs(incoming, outgoing)
    assert len(pairs) == 1
    assert pairs[0].pair_id == "show/s01e01"
    assert pairs[0].layout == PairLayout.SPLIT
    assert pairs[0].output_path == outgoing / "show" / "s01e01.mkv"


def test_subdir_wins_over_flat_same_id(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    _touch_media(incoming / "ep01" / "base.mkv")
    _touch_media(incoming / "ep01" / "donor.mkv")
    _touch_media(incoming / "ep01.base.mkv")
    _touch_media(incoming / "ep01.donor.mkv")

    pairs = discover_pairs(incoming, outgoing)
    assert len(pairs) == 1
    assert pairs[0].layout == PairLayout.SUBDIR
    assert pairs[0].base_path.parent.name == "ep01"


def test_classify_skip_completed_and_failed(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    base = _touch_media(tmp_path / "base.mkv")
    donor = _touch_media(tmp_path / "donor.mkv")
    out = tmp_path / "out" / "ep.mkv"
    from jarvis_media_processor.batch.models import MediaPair

    pair = MediaPair(
        pair_id="ep",
        base_path=base,
        donor_path=donor,
        output_path=out,
        layout=PairLayout.FLAT,
    )
    key = default_idempotency_key(base, donor, out, use_video=True, fine_warp=True)
    record = store.create(
        JobParams(
            base_path=base,
            donor_path=donor,
            output_path=out,
            idempotency_key=key,
        )
    )
    record.state = JobState.SUCCEEDED
    store.save(record)

    disposition, reason, prior = classify_pair(pair, store, retry=False)
    assert disposition == PairDisposition.SKIP_COMPLETED
    assert prior is not None and prior.id == record.id
    assert "succeeded" in reason.lower()

    record.state = JobState.FAILED
    store.save(record)
    # Refresh idempotency pointer still points at failed job
    disposition, _, _ = classify_pair(pair, store, retry=False)
    assert disposition == PairDisposition.SKIP_FAILED

    disposition, _, _ = classify_pair(pair, store, retry=True)
    assert disposition == PairDisposition.RETRY


def test_classify_in_flight(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    base = _touch_media(tmp_path / "base.mkv")
    donor = _touch_media(tmp_path / "donor.mkv")
    out = tmp_path / "out.mkv"
    from jarvis_media_processor.batch.models import MediaPair

    pair = MediaPair(
        pair_id="ep",
        base_path=base,
        donor_path=donor,
        output_path=out,
        layout=PairLayout.FLAT,
    )
    key = default_idempotency_key(base, donor, out, use_video=True, fine_warp=True)
    record = store.create(
        JobParams(
            base_path=base,
            donor_path=donor,
            output_path=out,
            idempotency_key=key,
        )
    )
    record.state = JobState.RUNNING
    store.save(record)

    disposition, _, _ = classify_pair(pair, store, retry=True)
    assert disposition == PairDisposition.SKIP_IN_FLIGHT


def test_run_batch_dry_run_and_skip_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JMP_BATCH_INPUT_DIR", raising=False)
    monkeypatch.delenv("JMP_BATCH_OUTPUT_DIR", raising=False)

    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    work = tmp_path / "work"
    _touch_media(incoming / "new" / "base.mkv")
    _touch_media(incoming / "new" / "donor.mkv")
    _touch_media(incoming / "done" / "base.mkv")
    _touch_media(incoming / "done" / "donor.mkv")
    _touch_media(incoming / "bad" / "base.mkv")
    _touch_media(incoming / "bad" / "donor.mkv")

    from jarvis_media_processor.config import Settings

    settings = Settings(work_dir=work)
    job_store = JobStore(settings=settings)
    batch_store = BatchStore(settings=settings)

    done_out = outgoing / "done.mkv"
    bad_out = outgoing / "bad.mkv"
    done_key = default_idempotency_key(
        incoming / "done" / "base.mkv",
        incoming / "done" / "donor.mkv",
        done_out,
        use_video=True,
        fine_warp=True,
    )
    bad_key = default_idempotency_key(
        incoming / "bad" / "base.mkv",
        incoming / "bad" / "donor.mkv",
        bad_out,
        use_video=True,
        fine_warp=True,
    )
    done = job_store.create(
        JobParams(
            base_path=incoming / "done" / "base.mkv",
            donor_path=incoming / "done" / "donor.mkv",
            output_path=done_out,
            idempotency_key=done_key,
        )
    )
    done.state = JobState.SUCCEEDED
    job_store.save(done)
    bad = job_store.create(
        JobParams(
            base_path=incoming / "bad" / "base.mkv",
            donor_path=incoming / "bad" / "donor.mkv",
            output_path=bad_out,
            idempotency_key=bad_key,
        )
    )
    bad.state = JobState.FAILED
    job_store.save(bad)

    runner = MagicMock()
    summary = run_batch(
        input_dir=incoming,
        output_dir=outgoing,
        dry_run=True,
        retry=False,
        settings=settings,
        job_store=job_store,
        batch_store=batch_store,
        job_runner=runner,
    )
    runner.assert_not_called()
    assert summary.counts()["planned"] == 1
    assert summary.counts()["skipped"] == 2
    by_id = {i.pair_id: i for i in summary.items}
    assert by_id["new"].disposition == PairDisposition.RUN
    assert by_id["new"].outcome == BatchItemOutcome.PLANNED
    assert by_id["done"].outcome == BatchItemOutcome.SKIPPED
    assert by_id["bad"].disposition == PairDisposition.SKIP_FAILED

    # With retry: failed becomes planned retry; completed stays skipped
    summary2 = run_batch(
        input_dir=incoming,
        output_dir=outgoing,
        dry_run=True,
        retry=True,
        settings=settings,
        job_store=job_store,
        batch_store=batch_store,
        job_runner=runner,
    )
    by_id2 = {i.pair_id: i for i in summary2.items}
    assert by_id2["bad"].disposition == PairDisposition.RETRY
    assert by_id2["done"].disposition == PairDisposition.SKIP_COMPLETED


def test_run_batch_invokes_runner_and_persists(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    work = tmp_path / "work"
    _touch_media(incoming / "ep" / "base.mkv")
    _touch_media(incoming / "ep" / "donor.mkv")

    from jarvis_media_processor.config import Settings
    from jarvis_media_processor.jobs.models import JobProgress, JobRecord

    settings = Settings(work_dir=work)
    job_store = JobStore(settings=settings)
    batch_store = BatchStore(settings=settings)

    def fake_runner(
        base: Path,
        donor: Path,
        output: Path,
        **_kwargs: object,
    ) -> JobRecord:
        key = default_idempotency_key(base, donor, output, use_video=True, fine_warp=True)
        record = job_store.create(
            JobParams(
                base_path=base,
                donor_path=donor,
                output_path=output,
                idempotency_key=key,
            )
        )
        record.state = JobState.SUCCEEDED
        record.progress = JobProgress(percent=100.0, stage="mux", message="ok")
        job_store.save(record)
        return record

    summary = run_batch(
        input_dir=incoming,
        output_dir=outgoing,
        settings=settings,
        job_store=job_store,
        batch_store=batch_store,
        job_runner=fake_runner,
    )
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.skipped == 0
    assert summary.items[0].job_id is not None

    loaded = batch_store.load(summary.id)
    assert loaded.counts() == summary.counts()

    # Second run skips completed
    summary2 = run_batch(
        input_dir=incoming,
        output_dir=outgoing,
        settings=settings,
        job_store=job_store,
        batch_store=batch_store,
        job_runner=fake_runner,
    )
    assert summary2.skipped == 1
    assert summary2.succeeded == 0


def test_run_batch_uses_batch_temp_dir(tmp_path: Path) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    temp = tmp_path / "batch-temp"
    _touch_media(incoming / "ep" / "base.mkv")
    _touch_media(incoming / "ep" / "donor.mkv")

    from jarvis_media_processor.config import Settings
    from jarvis_media_processor.jobs.models import JobProgress, JobRecord

    settings = Settings(work_dir=tmp_path / "work", batch_temp_dir=temp)

    def fake_runner(
        base: Path,
        donor: Path,
        output: Path,
        *,
        store: JobStore,
        **_kwargs: object,
    ) -> JobRecord:
        key = default_idempotency_key(base, donor, output, use_video=True, fine_warp=True)
        record = store.create(
            JobParams(
                base_path=base,
                donor_path=donor,
                output_path=output,
                idempotency_key=key,
            )
        )
        record.state = JobState.SUCCEEDED
        record.progress = JobProgress(percent=100.0, stage="mux", message="ok")
        store.save(record)
        return record

    summary = run_batch(
        input_dir=incoming,
        output_dir=outgoing,
        settings=settings,
        job_runner=fake_runner,
    )
    assert (temp / "batches" / summary.id / "batch.json").is_file()
    assert summary.items[0].job_id is not None
    assert (temp / "jobs" / summary.items[0].job_id / "job.json").is_file()
    assert not (tmp_path / "work" / "batches").exists()


def test_run_batch_retry_does_not_corrupt_completed(
    tmp_path: Path,
) -> None:
    incoming = tmp_path / "in"
    outgoing = tmp_path / "out"
    work = tmp_path / "work"
    _touch_media(incoming / "ep" / "base.mkv")
    _touch_media(incoming / "ep" / "donor.mkv")

    from jarvis_media_processor.config import Settings

    settings = Settings(work_dir=work)
    job_store = JobStore(settings=settings)
    batch_store = BatchStore(settings=settings)
    out = outgoing / "ep.mkv"
    key = default_idempotency_key(
        incoming / "ep" / "base.mkv",
        incoming / "ep" / "donor.mkv",
        out,
        use_video=True,
        fine_warp=True,
    )
    record = job_store.create(
        JobParams(
            base_path=incoming / "ep" / "base.mkv",
            donor_path=incoming / "ep" / "donor.mkv",
            output_path=out,
            idempotency_key=key,
        )
    )
    record.state = JobState.SUCCEEDED
    job_store.save(record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"good-output")

    runner = MagicMock()
    summary = run_batch(
        input_dir=incoming,
        output_dir=outgoing,
        retry=True,
        settings=settings,
        job_store=job_store,
        batch_store=batch_store,
        job_runner=runner,
    )
    runner.assert_not_called()
    assert summary.items[0].disposition == PairDisposition.SKIP_COMPLETED
    assert out.read_bytes() == b"good-output"
