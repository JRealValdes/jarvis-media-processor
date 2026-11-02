"""Catalog of synthetic base/donor alignment scenarios."""

from __future__ import annotations

from jarvis_media_processor.fixtures.models import (
    GroundTruth,
    Landmark,
    MediaRef,
    SegmentMapping,
    SyncStrategyLevel,
    TimeInterval,
)

# Shared building blocks: three “scenes” of tone+color, 2s each on the base.
_SCENE = 2.0
_BASE_DURATION = 6.0


def _base_ref(filename: str = "base.mkv") -> MediaRef:
    return MediaRef(filename=filename, duration_seconds=_BASE_DURATION)


def scenario_already_synced() -> GroundTruth:
    return GroundTruth(
        scenario_id="already_synced",
        description="Donor audio matches the base timeline with no offset or edits.",
        expected_strategy=SyncStrategyLevel.ALREADY_SYNCED,
        base=_base_ref(),
        donor=MediaRef(filename="donor.mkv", duration_seconds=_BASE_DURATION),
        segments=[
            SegmentMapping(
                base_start=0.0,
                base_end=_BASE_DURATION,
                donor_start=0.0,
                donor_end=_BASE_DURATION,
            )
        ],
        landmarks=[
            Landmark(kind="beep", base_time=0.5, donor_time=0.5),
            Landmark(kind="beep", base_time=2.5, donor_time=2.5),
            Landmark(kind="beep", base_time=4.5, donor_time=4.5),
        ],
        global_offset_seconds=0.0,
        global_speed=1.0,
    )


def scenario_constant_offset() -> GroundTruth:
    offset = 1.0
    donor_duration = _BASE_DURATION + offset
    return GroundTruth(
        scenario_id="constant_offset",
        description="Donor starts 1s later (leading silence); a single offset aligns all content.",
        expected_strategy=SyncStrategyLevel.CONSTANT_OFFSET,
        base=_base_ref(),
        donor=MediaRef(filename="donor.mkv", duration_seconds=donor_duration),
        segments=[
            SegmentMapping(
                base_start=0.0,
                base_end=_BASE_DURATION,
                donor_start=offset,
                donor_end=offset + _BASE_DURATION,
            )
        ],
        landmarks=[
            Landmark(kind="beep", base_time=0.5, donor_time=0.5 + offset),
            Landmark(kind="beep", base_time=2.5, donor_time=2.5 + offset),
            Landmark(kind="beep", base_time=4.5, donor_time=4.5 + offset),
        ],
        global_offset_seconds=offset,
        global_speed=1.0,
    )


def scenario_unrelated_block() -> GroundTruth:
    """Donor inserts a 2s unrelated block between scenes (ad-like), no base change."""
    block = 2.0
    donor_duration = _BASE_DURATION + block
    return GroundTruth(
        scenario_id="unrelated_block",
        description="Donor inserts an unrelated middle block; piecewise exclusion required.",
        expected_strategy=SyncStrategyLevel.PIECEWISE,
        base=_base_ref(),
        donor=MediaRef(filename="donor.mkv", duration_seconds=donor_duration),
        segments=[
            SegmentMapping(
                base_start=0.0,
                base_end=2.0,
                donor_start=0.0,
                donor_end=2.0,
            ),
            SegmentMapping(
                base_start=2.0,
                base_end=_BASE_DURATION,
                donor_start=2.0 + block,
                donor_end=2.0 + block + 4.0,
            ),
        ],
        unrelated_donor_intervals=[TimeInterval(start=2.0, end=2.0 + block)],
        landmarks=[
            Landmark(kind="beep", base_time=0.5, donor_time=0.5),
            Landmark(kind="beep", base_time=2.5, donor_time=2.5 + block),
            Landmark(kind="beep", base_time=4.5, donor_time=4.5 + block),
        ],
    )


def scenario_constant_speed() -> GroundTruth:
    speed = 1.1
    donor_duration = _BASE_DURATION / speed
    return GroundTruth(
        scenario_id="constant_speed",
        description="Donor plays ~1.1× faster globally; constant speed should suffice.",
        expected_strategy=SyncStrategyLevel.CONSTANT_SPEED,
        base=_base_ref(),
        donor=MediaRef(filename="donor.mkv", duration_seconds=round(donor_duration, 6)),
        segments=[
            SegmentMapping(
                base_start=0.0,
                base_end=_BASE_DURATION,
                donor_start=0.0,
                donor_end=round(donor_duration, 6),
                speed=speed,
            )
        ],
        landmarks=[
            Landmark(
                kind="beep",
                base_time=0.5,
                donor_time=round(0.5 / speed, 6),
            ),
            Landmark(
                kind="beep",
                base_time=2.5,
                donor_time=round(2.5 / speed, 6),
            ),
            Landmark(
                kind="beep",
                base_time=4.5,
                donor_time=round(4.5 / speed, 6),
            ),
        ],
        global_offset_seconds=0.0,
        global_speed=speed,
    )


def scenario_local_length_drift() -> GroundTruth:
    """Middle scene longer on donor — progressive local desync (hard edit case)."""
    # Base scenes: 2+2+2. Donor: 2 + 3 + 2 (middle elongated by 1s).
    extra = 1.0
    donor_duration = _BASE_DURATION + extra
    return GroundTruth(
        scenario_id="local_length_drift",
        description=(
            "Donor middle scene is 1s longer than base; offsets change after the middle. "
            "Models hard rip/edit cases without commercials."
        ),
        expected_strategy=SyncStrategyLevel.PIECEWISE,
        base=_base_ref(),
        donor=MediaRef(filename="donor.mkv", duration_seconds=donor_duration),
        segments=[
            SegmentMapping(base_start=0.0, base_end=2.0, donor_start=0.0, donor_end=2.0),
            SegmentMapping(base_start=2.0, base_end=4.0, donor_start=2.0, donor_end=5.0),
            SegmentMapping(base_start=4.0, base_end=_BASE_DURATION, donor_start=5.0, donor_end=7.0),
        ],
        landmarks=[
            Landmark(kind="beep", base_time=0.5, donor_time=0.5, note="scene1"),
            Landmark(kind="beep", base_time=2.5, donor_time=2.5, note="scene2-start-aligned"),
            Landmark(
                kind="beep",
                base_time=4.5,
                donor_time=5.5,
                note="scene3-shifted-by-extra",
            ),
        ],
    )


SCENARIOS: dict[str, GroundTruth] = {
    s.scenario_id: s
    for s in (
        scenario_already_synced(),
        scenario_constant_offset(),
        scenario_unrelated_block(),
        scenario_constant_speed(),
        scenario_local_length_drift(),
    )
}


def list_scenario_ids() -> list[str]:
    """Return known scenario identifiers in catalog order."""
    return list(SCENARIOS.keys())


def get_scenario(scenario_id: str) -> GroundTruth:
    """Return ground truth for ``scenario_id``.

    Raises:
        KeyError: If the scenario is unknown.
    """
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        known = ", ".join(list_scenario_ids())
        raise KeyError(f"Unknown scenario {scenario_id!r}. Known: {known}") from exc
