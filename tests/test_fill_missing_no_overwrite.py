"""EN missing-base fill must not stomp existing Spanish placements."""

from __future__ import annotations

import numpy as np
import pytest

from jarvis_media_processor.alignment.models import TimelineInterval
from jarvis_media_processor.reconstruct.render import fill_missing_base_audio


def test_fill_missing_base_preserves_occupied_spanish() -> None:
    sample_rate = 100
    # 2 seconds canvas; Spanish already in 0.5–1.5s
    canvas = np.zeros((1, 200), dtype=np.float64)
    canvas[0, 50:150] = 0.5
    base = np.ones((1, 200), dtype=np.float64)  # EN would be loud 1.0
    intervals = [
        TimelineInterval(start=0.0, end=2.0, timeline="base", reason="missing")
    ]
    out = fill_missing_base_audio(
        canvas,
        base,
        sample_rate,
        intervals,
        min_seconds=0.5,
        crossfade_seconds=0.0,
    )
    # Occupied Spanish region unchanged.
    assert np.allclose(out[0, 50:150], 0.5)
    # Silent sides may receive EN.
    assert float(out[0, 10]) == pytest.approx(1.0)
    assert float(out[0, 190]) == pytest.approx(1.0)
