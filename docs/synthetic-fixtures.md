# Synthetic fixtures

Automated alignment tests must not depend on copyrighted commercial movies. This project ships a **catalog of synthetic base/donor scenarios** with machine-readable ground truth, plus an FFmpeg-based generator.

Product terminology and sync strategy: GitHub issue [#16](https://github.com/JRealValdes/jarvis-media-processor/issues/16).

## Scenarios

| Id | Expected strategy | What it models |
|----|-------------------|----------------|
| `already_synced` | `already_synced` | Identical timelines |
| `constant_offset` | `constant_offset` | Donor starts later (leading pad) |
| `unrelated_block` | `piecewise` | Unrelated middle block (ad-like) |
| `constant_speed` | `constant_speed` | Global ~1.1× speed |
| `local_length_drift` | `piecewise` | Middle scene longer on donor (progressive desync; hard rip/edit case) |

Each generated scenario directory contains:

- `base.mkv` — timeline authority
- `donor.mkv` — audio (and video) to align
- `ground_truth.json` — expected segments, landmarks, strategy level

Shared **beep landmarks** are embedded so later acoustic matching tests have known anchors.

## Generate

Requires FFmpeg on `PATH` (or `JMP_FFMPEG_PATH`).

```bash
uv run jmp fixtures list
uv run jmp fixtures generate -o fixtures/synthetic
uv run jmp fixtures generate -o fixtures/synthetic -s local_length_drift
```

Generated media under `fixtures/synthetic/` is gitignored by default; regenerate locally as needed.

## Use in tests

```python
from jarvis_media_processor.fixtures import get_scenario, list_scenario_ids, load_scenario_truth
from jarvis_media_processor.testing import requires_ffmpeg
```

- Catalog ground truth (no FFmpeg): `get_scenario("constant_offset")`
- After generation: `load_scenario_truth(path)`
- Mark FFmpeg-backed tests with `@requires_ffmpeg`
