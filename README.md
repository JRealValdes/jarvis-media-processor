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

You can also run:

```bash
uv run python -m jarvis_media_processor --help
```

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
| `JMP_FFMPEG_PATH` | `ffmpeg` | FFmpeg executable |
| `JMP_FFPROBE_PATH` | `ffprobe` | FFprobe executable |

## Package layout

```
src/jarvis_media_processor/
  cli/           # Typer CLI (jmp)
  config/        # Typed settings
  media/         # Inspection (planned)
  audio/         # Extraction + landmarks (planned)
  video/         # Frame matching (planned)
  alignment/     # Sync model (planned)
  reconstruct/   # Donor audio on base timeline (planned)
  output/        # MKV mux (planned)
  validate/      # Reports (planned)
  jobs/          # Job runner (planned)
  api/           # HTTP API (planned)
```

## License

MIT
