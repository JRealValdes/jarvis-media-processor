"""Schedule tick planning models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from jarvis_media_processor.batch.models import MediaPair, PairDisposition


class TickAction(StrEnum):
    """High-level action for one pair inside a scheduled tick."""

    PROCESS = "process"
    SKIP = "skip"


class PlannedPair(BaseModel):
    """Dedupe decision for one discovered pair before any job runs."""

    pair: MediaPair
    disposition: PairDisposition
    action: TickAction
    reason: str
    prior_job_id: str | None = None


class TickPlan(BaseModel):
    """Dry plan for a scheduled tick (testable without running media jobs)."""

    pairs: list[PlannedPair] = Field(default_factory=list)

    @property
    def to_process(self) -> list[PlannedPair]:
        return [p for p in self.pairs if p.action == TickAction.PROCESS]

    @property
    def to_skip(self) -> list[PlannedPair]:
        return [p for p in self.pairs if p.action == TickAction.SKIP]

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.pairs),
            "process": len(self.to_process),
            "skip": len(self.to_skip),
        }
