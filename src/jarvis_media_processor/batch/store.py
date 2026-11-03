"""Filesystem persistence for batch summaries."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from jarvis_media_processor.batch.models import BatchSummary
from jarvis_media_processor.config import Settings, get_settings

_BATCH_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class BatchStoreError(RuntimeError):
    """Batch store / lookup failure."""


class BatchStore:
    """Batches under ``{work_dir}/batches/<id>/batch.json``."""

    def __init__(self, root: Path | None = None, *, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.root = Path(root) if root is not None else Path(cfg.work_dir) / "batches"
        self.root.mkdir(parents=True, exist_ok=True)

    def batch_dir(self, batch_id: str) -> Path:
        return self.root / batch_id

    def batch_json_path(self, batch_id: str) -> Path:
        return self.batch_dir(batch_id) / "batch.json"

    def new_id(self) -> str:
        return str(uuid.uuid4())

    def save(self, summary: BatchSummary) -> None:
        """Atomically write ``batch.json``."""
        if not _BATCH_ID_RE.match(summary.id):
            raise BatchStoreError(f"Invalid batch id: {summary.id}")
        path = self.batch_json_path(summary.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    def load(self, batch_id: str) -> BatchSummary:
        path = self.batch_json_path(batch_id)
        if not path.is_file():
            raise BatchStoreError(f"Batch not found: {batch_id}")
        return BatchSummary.model_validate_json(path.read_text(encoding="utf-8"))

    def list_batches(self) -> list[BatchSummary]:
        batches: list[BatchSummary] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            path = child / "batch.json"
            if path.is_file():
                batches.append(BatchSummary.model_validate_json(path.read_text(encoding="utf-8")))
        batches.sort(key=lambda b: b.created_at, reverse=True)
        return batches
