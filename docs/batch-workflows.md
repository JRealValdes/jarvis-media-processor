# Batch and folder-based workflows

Operators process many **base** / **donor** pairs. Batch mode discovers pairs from a folder layout, maps each to a deterministic output path, and enqueues graft jobs via the existing job store.

Job workspaces and bulky intermediates live under the **temp** root (`…/jobs/<id>/workspace`). By default that is `JMP_WORK_DIR`; set `JMP_BATCH_TEMP_DIR` to keep batch job/batch JSON trees separate from other work.

## Directory roots

| Role | Setting / flag | Purpose |
|------|----------------|---------|
| **Input** | `JMP_BATCH_INPUT_DIR` or `jmp batch … <input>` | Read-only discovery root for pairs |
| **Output** | `JMP_BATCH_OUTPUT_DIR` or `--output-dir` / `-o` | Deterministic MKV outputs |
| **Temp** | `JMP_BATCH_TEMP_DIR` (else `JMP_WORK_DIR`) | Job JSON, workspaces, batch summaries |

Sources under the input root are **never** overwritten. Outputs always resolve under the output root as `<pair_id>.mkv` (nested ids keep relative folders).

## Pairing layouts

Discovery accepts three conventions. If the same `pair_id` appears in more than one layout, **subdir** wins, then **flat**, then **split**.

### 1. Per-pair subdirectory (preferred)

```text
<input>/<pair_id>/base.mkv
<input>/<pair_id>/donor.mkv
```

Any supported media extension works (`.mkv`, `.mp4`, `.mov`, …). Role names are case-insensitive (`base`, `Base`, …).

Output: `<output>/<pair_id>.mkv`

### 2. Flat stem suffixes

```text
<input>/<pair_id>.base.mkv
<input>/<pair_id>.donor.mkv
```

Output: `<output>/<pair_id>.mkv`

### 3. Split trees

```text
<input>/base/<relpath>.mkv
<input>/donor/<relpath>.mkv
```

`pair_id` is the relative path without extension (posix), e.g. `show/s01e01`.

Output: `<output>/<relpath>.mkv` (folders created as needed)

Folders named `base` / `donor` at the input root are reserved for this layout and are not treated as subdir pairs.

## Skip and retry

Classification uses the job store idempotency key derived from resolved base, donor, output, and options (`use_video`, `fine_warp`):

| Prior job state | Default | With `--retry` |
|-----------------|---------|----------------|
| none | run | run |
| succeeded | **skip** | **skip** (completed work is never reprocessed) |
| pending / running | **skip** | **skip** |
| failed / cancelled | **skip** | **retry** (new job; same deterministic output) |

Retries are explicit so a flaky overnight run does not loop forever. A retry creates a **new** job id (same as single-job idempotency after failure). Mux may replace a partial failed output at the deterministic path; successful outputs are not re-touched because completed pairs stay skipped.

## CLI

```bash
# List pairs only
uv run jmp batch discover path/to/incoming -o path/to/outgoing

# Classify + run (skips completed / failed-by-default)
uv run jmp batch run path/to/incoming -o path/to/outgoing

# Plan only
uv run jmp batch run path/to/incoming -o path/to/outgoing --dry-run

# Re-queue failed/cancelled pairs
uv run jmp batch run path/to/incoming -o path/to/outgoing --retry

uv run jmp batch status <batch-id>
uv run jmp batch list
```

Batch summaries persist under `JMP_WORK_DIR/batches/<id>/batch.json` and include per-pair disposition plus counts: `succeeded` / `failed` / `skipped` / `planned`.

## Safety checklist

- Never overwrite base or donor files (`validate_output_path`).
- Deterministic output naming from `pair_id` only.
- Completed jobs stay skipped even with `--retry`.
- Failed jobs require `--retry` to enqueue again.
