"""Tests for standalone job store, cancel, and pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis_media_processor.jobs.models import JobParams, JobState
from jarvis_media_processor.jobs.pipeline import (
    JobCancelled,
    default_idempotency_key,
    run_graft_pipeline,
)
from jarvis_media_processor.jobs.store import JobStore, JobStoreError


def _params(tmp: Path, *, key: str | None = "k1") -> JobParams:
    return JobParams(
        base_path=tmp / "base.mkv",
        donor_path=tmp / "donor.mkv",
        output_path=tmp / "out.mkv",
        idempotency_key=key,
    )


def test_store_create_save_load_roundtrip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "base.mkv").write_bytes(b"x")
    (tmp_path / "donor.mkv").write_bytes(b"y")
    record = store.create(_params(tmp_path))
    assert record.state == JobState.PENDING
    assert record.workspace is not None
    assert record.workspace.is_dir()

    loaded = store.load(record.id)
    assert loaded.id == record.id
    assert loaded.params.output_path == tmp_path / "out.mkv"


def test_idempotency_returns_same_nonterminal_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "base.mkv").write_bytes(b"x")
    (tmp_path / "donor.mkv").write_bytes(b"y")
    first = store.create(_params(tmp_path, key="same"))
    second = store.create(_params(tmp_path, key="same"))
    assert first.id == second.id


def test_idempotency_allows_retry_after_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "base.mkv").write_bytes(b"x")
    (tmp_path / "donor.mkv").write_bytes(b"y")
    first = store.create(_params(tmp_path, key="retry"))
    first.state = JobState.FAILED
    store.save(first)
    second = store.create(_params(tmp_path, key="retry"))
    assert second.id != first.id
    assert second.state == JobState.PENDING


def test_cancel_pending_is_immediate(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "base.mkv").write_bytes(b"x")
    (tmp_path / "donor.mkv").write_bytes(b"y")
    record = store.create(_params(tmp_path))
    cancelled = store.request_cancel(record.id)
    assert cancelled.state == JobState.CANCELLED
    assert cancelled.is_terminal()


def test_cancel_running_sets_flag(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "base.mkv").write_bytes(b"x")
    (tmp_path / "donor.mkv").write_bytes(b"y")
    record = store.create(_params(tmp_path))
    record.state = JobState.RUNNING
    store.save(record)
    store.request_cancel(record.id)
    assert store.is_cancel_requested(record.id)
    loaded = store.load(record.id)
    assert loaded.state == JobState.RUNNING


def test_load_missing_raises(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    with pytest.raises(JobStoreError, match="not found"):
        store.load("00000000-0000-0000-0000-000000000001")


def test_default_idempotency_key_stable(tmp_path: Path) -> None:
    a = tmp_path / "a.mkv"
    b = tmp_path / "b.mkv"
    o = tmp_path / "o.mkv"
    k1 = default_idempotency_key(a, b, o, use_video=True, fine_warp=True)
    k2 = default_idempotency_key(a, b, o, use_video=True, fine_warp=True)
    k3 = default_idempotency_key(a, b, o, use_video=False, fine_warp=True)
    k4 = default_idempotency_key(
        a, b, o, use_video=True, fine_warp=True, optimize_after_mux=False
    )
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_pipeline_cancel_between_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JobStore(tmp_path / "jobs")
    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.mkv"
    out = tmp_path / "out.mkv"
    base.write_bytes(b"base")
    donor.write_bytes(b"donor")

    record = store.create(
        JobParams(
            base_path=base,
            donor_path=donor,
            output_path=out,
            fine_warp=False,
            idempotency_key=None,
        )
    )

    inspect = MagicMock()
    inspect.duration_seconds = 10.0
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.inspect_media",
        lambda *a, **k: inspect,
    )

    def _align_then_cancel(*_a: object, **_k: object) -> object:
        store.cancel_flag_path(record.id).write_text("x", encoding="utf-8")
        raise AssertionError("align should not finish after cancel flag in this test")

    # Cancel after inspect, before align check
    calls = {"n": 0}

    def fake_check(store_: JobStore, job_id: str) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            raise JobCancelled("Cancel requested")
        if store_.is_cancel_requested(job_id):
            raise JobCancelled("Cancel requested")

    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline._check_cancel",
        fake_check,
    )
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.detect_alignment",
        _align_then_cancel,
    )

    result = run_graft_pipeline(record, store, cleanup_workspace_on_success=False)
    assert result.state == JobState.CANCELLED
    assert result.finished_at is not None


def test_pipeline_success_with_mocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JobStore(tmp_path / "jobs")
    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.mkv"
    out = tmp_path / "out.mkv"
    base.write_bytes(b"base")
    donor.write_bytes(b"donor")

    record = store.create(
        JobParams(
            base_path=base,
            donor_path=donor,
            output_path=out,
            fine_warp=True,
            idempotency_key=None,
        )
    )

    inspect = MagicMock()
    inspect.duration_seconds = 12.5
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.inspect_media",
        lambda *a, **k: inspect,
    )

    alignment = MagicMock()
    alignment.model_dump_json.return_value = "{}"
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.detect_alignment",
        lambda *a, **k: alignment,
    )

    recon_wav = tmp_path / "jobs" / record.id / "workspace" / "reconstruct" / "donor_on_base.wav"
    recon_wav.parent.mkdir(parents=True, exist_ok=True)
    recon_wav.write_bytes(b"wav")
    recon = MagicMock()
    recon.output_wav = recon_wav
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.reconstruct_donor_audio",
        lambda *a, **k: recon,
    )

    fine_wav = tmp_path / "jobs" / record.id / "workspace" / "fine-warp" / "donor_on_base_fine.wav"
    fine_wav.parent.mkdir(parents=True, exist_ok=True)
    fine_wav.write_bytes(b"fine")
    fine = MagicMock()
    fine.output_wav = fine_wav
    fine.plan_json = fine_wav.with_name("fine_warp_plan.json")
    fine.plan_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.fine_warp_reconstructed_audio",
        lambda *a, **k: fine,
    )

    def fake_mux(*_a: object, **_k: object) -> Path:
        out.write_bytes(b"mkv")
        return out

    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.mux_mkv",
        fake_mux,
    )

    result = run_graft_pipeline(record, store)
    assert result.state == JobState.SUCCEEDED
    assert result.artifacts["output_mkv"] == out
    assert result.artifacts["fine_warped_wav"] == fine_wav
    assert out.is_file()
    assert any("optimize_decision:" in n for n in result.notes)
    reloaded = store.load(record.id)
    assert reloaded.state == JobState.SUCCEEDED


def test_pipeline_optimize_after_mux_when_policy_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis_media_processor.config import Settings
    from jarvis_media_processor.output.optimize import OptimizeDecision

    store = JobStore(tmp_path / "jobs")
    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.mkv"
    out = tmp_path / "out.mkv"
    base.write_bytes(b"base")
    donor.write_bytes(b"donor")

    record = store.create(
        JobParams(
            base_path=base,
            donor_path=donor,
            output_path=out,
            fine_warp=False,
            optimize_after_mux=True,
            idempotency_key=None,
        )
    )

    inspect = MagicMock()
    inspect.duration_seconds = 12.5
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.inspect_media",
        lambda *a, **k: inspect,
    )
    alignment = MagicMock()
    alignment.model_dump_json.return_value = "{}"
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.detect_alignment",
        lambda *a, **k: alignment,
    )
    recon_wav = tmp_path / "jobs" / record.id / "workspace" / "reconstruct" / "d.wav"
    recon_wav.parent.mkdir(parents=True, exist_ok=True)
    recon_wav.write_bytes(b"wav")
    recon = MagicMock()
    recon.output_wav = recon_wav
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.reconstruct_donor_audio",
        lambda *a, **k: recon,
    )

    def fake_mux(*_a: object, **_k: object) -> Path:
        out.write_bytes(b"muxed-large")
        return out

    monkeypatch.setattr("jarvis_media_processor.jobs.pipeline.mux_mkv", fake_mux)
    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.should_optimize_output",
        lambda *_a, **_k: OptimizeDecision(True, "test force optimize"),
    )

    def fake_optimize(source: Path, dest: Path, **_k: object) -> Path:
        Path(dest).write_bytes(b"optimized")
        return Path(dest)

    monkeypatch.setattr(
        "jarvis_media_processor.jobs.pipeline.optimize_mkv",
        fake_optimize,
    )

    settings = Settings(
        _env_file=None,
        jobs_optimize_after_mux=True,
        jobs_optimize_profile="archive",
        optimize_size_threshold_bytes=1,
    )
    result = run_graft_pipeline(record, store, settings=settings)
    assert result.state == JobState.SUCCEEDED
    assert out.read_bytes() == b"optimized"
    assert any("optimize applied" in n for n in result.notes)
