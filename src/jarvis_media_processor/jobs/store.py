"""Filesystem persistence for job records (restart-safe)."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.jobs.models import JobParams, JobRecord, JobState, utc_now

_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class JobStoreError(RuntimeError):
    """Job store / lookup failure."""


class JobStore:
    """Jobs under ``{work_dir}/jobs/<id>/job.json`` plus optional idempotency index."""

    def __init__(self, root: Path | None = None, *, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.root = Path(root) if root is not None else Path(cfg.work_dir) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._keys_dir = self.root / "_idempotency"
        self._keys_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Directory for one job's metadata and workspace."""
        return self.root / job_id

    def job_json_path(self, job_id: str) -> Path:
        """Path to the persisted ``job.json``."""
        return self.job_dir(job_id) / "job.json"

    def cancel_flag_path(self, job_id: str) -> Path:
        """Path to cooperative cancel marker."""
        return self.job_dir(job_id) / "cancel.requested"

    def create(self, params: JobParams, *, job_id: str | None = None) -> JobRecord:
        """Create a pending job (or return an existing one for the same idempotency key)."""
        if params.idempotency_key:
            existing = self.find_by_idempotency_key(params.idempotency_key)
            if existing is not None and existing.state not in {
                JobState.FAILED,
                JobState.CANCELLED,
            }:
                return existing

        jid = job_id or str(uuid.uuid4())
        if not _JOB_ID_RE.match(jid):
            raise JobStoreError(f"Invalid job id: {jid}")
        if self.job_json_path(jid).is_file():
            raise JobStoreError(f"Job already exists: {jid}")

        work = self.job_dir(jid) / "workspace"
        work.mkdir(parents=True, exist_ok=True)
        record = JobRecord(
            id=jid,
            params=params,
            state=JobState.PENDING,
            workspace=work,
        )
        self.save(record)
        if params.idempotency_key:
            self._write_idempotency(params.idempotency_key, jid)
        return record

    def save(self, record: JobRecord) -> None:
        """Atomically write ``job.json``."""
        record.touch()
        path = self.job_json_path(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            record.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def load(self, job_id: str) -> JobRecord:
        """Load a job by id."""
        path = self.job_json_path(job_id)
        if not path.is_file():
            raise JobStoreError(f"Job not found: {job_id}")
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_jobs(self) -> list[JobRecord]:
        """Return all jobs (newest updated first)."""
        jobs: list[JobRecord] = []
        for child in self.root.iterdir():
            if not child.is_dir() or child.name.startswith("_"):
                continue
            path = child / "job.json"
            if path.is_file():
                jobs.append(
                    JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
                )
        jobs.sort(key=lambda j: j.updated_at, reverse=True)
        return jobs

    def find_by_idempotency_key(self, key: str) -> JobRecord | None:
        """Resolve idempotency key → job, if present."""
        pointer = self._keys_dir / _safe_key_filename(key)
        if not pointer.is_file():
            return None
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            job_id = str(data["job_id"])
            return self.load(job_id)
        except (OSError, KeyError, json.JSONDecodeError, JobStoreError):
            return None

    def request_cancel(self, job_id: str) -> JobRecord:
        """Mark cancel requested; terminal jobs are unchanged."""
        record = self.load(job_id)
        if record.is_terminal():
            return record
        self.cancel_flag_path(job_id).write_text(
            utc_now().isoformat(),
            encoding="utf-8",
        )
        if record.state == JobState.PENDING:
            record.state = JobState.CANCELLED
            record.finished_at = utc_now()
            record.notes.append("Cancelled before start.")
            self.save(record)
        return record

    def is_cancel_requested(self, job_id: str) -> bool:
        """True when a cooperative cancel flag exists."""
        return self.cancel_flag_path(job_id).is_file()

    def _write_idempotency(self, key: str, job_id: str) -> None:
        pointer = self._keys_dir / _safe_key_filename(key)
        pointer.write_text(
            json.dumps({"job_id": job_id, "key": key}, indent=2) + "\n",
            encoding="utf-8",
        )


def _safe_key_filename(key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._") or "key"
    return cleaned[:180] + ".json"
