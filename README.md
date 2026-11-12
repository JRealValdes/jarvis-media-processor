# Jarvis Media Processor

Independent multimedia processor that grafts **donor** audio onto a **base** media timeline as an extra track. Designed to run and be tested without Jarvis; Jarvis may later orchestrate jobs via a stable worker interface.

Product model and sync strategy: see GitHub issue [#16](https://github.com/JRealValdes/jarvis-media-processor/issues/16).

## Terminology

| Term | Meaning |
|------|---------|
| **base** | Media that keeps its video and already-synced audio. Timeline authority. |
| **donor** | Media that provides the audio track/language to add. |

Original inputs are never modified or overwritten.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- **FFmpeg** and **FFprobe** available on `PATH` (required for media work in later issues; not needed for the current CLI smoke path)

## Installation

```bash
uv sync
```

This creates `.venv` and installs runtime plus development dependencies (pytest, ruff, mypy).

Optional local config:

```bash
cp .env.example .env
```

## CLI

```bash
uv run jmp --help
uv run jmp version
uv run jmp --version
```

Inspect one or more media files (e.g. base and donor) and print structured JSON:

```bash
uv run jmp inspect path/to/base.mkv path/to/donor.mkv
```

You can also run:

```bash
uv run python -m jarvis_media_processor --help
```

Requires **FFprobe** on `PATH` (or `JMP_FFPROBE_PATH`) for `inspect`.

### Synthetic fixtures

Generate reproducible base/donor pairs with ground-truth JSON (requires **FFmpeg**):

```bash
uv run jmp fixtures list
uv run jmp fixtures generate -o fixtures/synthetic
```

Details: [docs/synthetic-fixtures.md](docs/synthetic-fixtures.md).

### Audio landmarks

Extract analysis WAV and detect sparse energy-peak landmarks:

```bash
uv run jmp analyze-audio path/to/base.mkv --role base
uv run jmp analyze-audio path/to/donor.mkv --role donor -w .jmp/audio-analysis
```

Requires **FFmpeg** on `PATH` (or `JMP_FFMPEG_PATH`).

### Visual matching

Sample frames and match base↔donor via perceptual hashes:

```bash
uv run jmp match-visual path/to/base.mkv path/to/donor.mkv -w .jmp/visual-match
```

Notes: [docs/visual-matching.md](docs/visual-matching.md).

### Alignment

Escalate sync hypotheses (already synced → offset → speed → piecewise) for a base/donor pair:

```bash
uv run jmp align path/to/base.mkv path/to/donor.mkv -w .jmp/alignment
uv run jmp align path/to/base.mkv path/to/donor.mkv --no-video
```

### Reconstruct donor audio

Place donor audio onto the base timeline (silence in gaps; no shifting):

```bash
uv run jmp reconstruct path/to/donor.mkv .jmp/alignment/alignment.json --base-duration 7200 -w .jmp/reconstruct
```

### End-to-end graft pipeline

Typical path from a base/donor pair to a playable MKV:

```text
align → reconstruct → fine-warp → mux
```

```bash
uv run jmp align path/to/base.mkv path/to/donor.mkv -w .jmp/job
# use base duration from inspect/ffprobe
uv run jmp reconstruct path/to/donor.mkv .jmp/job/alignment.json --base-duration 7200 -w .jmp/job/reconstruct
uv run jmp fine-warp path/to/base.mkv .jmp/job/reconstruct/donor_on_base.wav -w .jmp/job/fine-warp
uv run jmp mux path/to/base.mkv .jmp/job/fine-warp/donor_on_base_fine.wav -o out.mkv
```

`fine-warp` is optional but recommended when lip-sync is close but not exact after reconstruct. It compares the reconstructed WAV to **base audio track 0**, builds a dense residual lag field from band-limited **onset** envelopes (not dialogue landmark snapping), keeps only high-confidence “pillars”, and interpolates elsewhere. Design notes and failure playbook: [docs/fine-warp.md](docs/fine-warp.md).

### Standalone jobs

Submit a full graft as a persisted job (status under `JMP_WORK_DIR/jobs/`):

```bash
uv run jmp job submit path/to/base.mkv path/to/donor.mkv -o out.mkv
uv run jmp job status <job-id>
uv run jmp job list
uv run jmp job cancel <job-id>
```

### Batch / folder workflows

Discover and enqueue many base/donor pairs from a folder layout (completed work skipped by default; failed pairs need `--retry`):

```bash
uv run jmp batch discover path/to/incoming -o path/to/outgoing
uv run jmp batch run path/to/incoming -o path/to/outgoing
uv run jmp batch run path/to/incoming -o path/to/outgoing --dry-run
uv run jmp batch run path/to/incoming -o path/to/outgoing --retry
uv run jmp batch status <batch-id>
uv run jmp batch list
```

Pairing layouts, roots, and skip/retry rules: [docs/batch-workflows.md](docs/batch-workflows.md).

### Scheduled execution

Prefer an OS scheduler invoking one tick (dedupe against completed jobs). Optional in-process loop for local polling:

```bash
uv run jmp schedule plan path/to/incoming -o path/to/outgoing
uv run jmp schedule tick path/to/incoming -o path/to/outgoing
uv run jmp schedule loop path/to/incoming -o path/to/outgoing --interval 3600
uv run jmp schedule show
```

Rationale and cron / systemd / Task Scheduler examples: [docs/scheduled-execution.md](docs/scheduled-execution.md).

### Mux

Stream-copy base A/V and attach the reconstructed (or fine-warped) donor audio as an extra track:

```bash
uv run jmp mux path/to/base.mkv path/to/donor_on_base.wav -o out.mkv
uv run jmp mux path/to/base.mkv path/to/donor_on_base_fine.wav -o out.mkv \
  --donor-language spa --donor-title "Spanish (reconstructed)"
```

### Optimize / compress

Shrink an already-complete multi-track MKV (e.g. a remux that has the languages you want but is too large). Soft subtitles stay selectable — never burned in.

Profiles:

| Profile | Video | Audio |
|---------|-------|-------|
| `archive` | HEVC (`libx265`) CRF ~18 (default) | re-encode (`JMP_OPTIMIZE_AUDIO_*`) |
| `balanced` | HEVC CRF ~21 | re-encode |
| `audio-slim` | stream copy | re-encode only |
| `remux` | stream copy | stream copy (drop unwanted tracks) |

```bash
uv run jmp optimize path/to/big.mkv -o path/to/smaller.mkv
uv run jmp optimize path/to/big.mkv -o out.mkv --profile archive
uv run jmp optimize path/to/big.mkv -o out.mkv --profile remux \
  --keep-audio spa,eng --keep-subs spa,eng
uv run jmp optimize path/to/big.mkv -o out.mkv --dry-run
```

Omit `--keep-audio` / `--keep-subs` to keep every track of that kind. CRF and audio encoder defaults are under `JMP_OPTIMIZE_*` (see `.env.example`).

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
```

Auto-format:

```bash
uv run ruff format src tests
```

## Configuration

Settings live in `jarvis_media_processor.config.settings.Settings` and are loaded from environment variables with the `JMP_` prefix (see `.env.example`).

| Variable | Default | Purpose |
|----------|---------|---------|
| `JMP_LOG_LEVEL` | `INFO` | Logging level |
| `JMP_WORK_DIR` | `.jmp` | Temp / job workspace root |
| `JMP_BATCH_INPUT_DIR` | (unset) | Default batch input root |
| `JMP_BATCH_OUTPUT_DIR` | (unset) | Default batch output root |
| `JMP_SCHEDULE_INTERVAL_SECONDS` | `3600` | Default `jmp schedule loop` interval |
| `JMP_SCHEDULE_RETRY_FAILED` | `false` | Re-queue failed pairs on schedule ticks |
| `JMP_FFMPEG_PATH` | `ffmpeg` | FFmpeg executable |
| `JMP_FFPROBE_PATH` | `ffprobe` | FFprobe executable |
| `JMP_OPTIMIZE_ARCHIVE_CRF` | `18` | libx265 CRF for `jmp optimize --profile archive` |
| `JMP_OPTIMIZE_BALANCED_CRF` | `21` | libx265 CRF for `balanced` |
| `JMP_OPTIMIZE_X265_PRESET` | `medium` | libx265 preset |
| `JMP_OPTIMIZE_AUDIO_CODEC` | `libopus` | Audio encoder for non-`remux` profiles |
| `JMP_OPTIMIZE_AUDIO_BITRATE` | `192k` | Audio bitrate for those profiles |
| `JMP_FINE_WARP_*` | (see Settings) | Dense residual lag field after reconstruct — [docs/fine-warp.md](docs/fine-warp.md) |

## Package layout

```
src/jarvis_media_processor/
  cli/           # Typer CLI (jmp)
  config/        # Typed settings
  media/         # Inspection
  audio/         # Extraction, landmarks, WAV I/O
  video/         # Frame matching / activity
  alignment/     # Sync model + curve walk
  reconstruct/   # Donor on base timeline + fine-warp
  output/        # MKV mux + optimize/compress
  report/        # Processing reports / validation
  jobs/          # Standalone job runner + store
  batch/         # Folder pairing + batch enqueue
  schedule/      # Unattended ticks (OS scheduler preferred)
  api/           # HTTP API (planned)
```

## License

MIT
