# Scheduled execution

Overnight / unattended processing runs the same folder batch pipeline as
[`jmp batch run`](batch-workflows.md), with logging and settings tuned for
recurring ticks.

## Approach choice (rationale)

| Mode | When to use |
|------|-------------|
| **OS scheduler → `jmp schedule tick`** (preferred) | Production overnight runs on Windows, Linux, or macOS. Survives reboots, uses OS credentials/logging, no long-lived Python process. |
| **Optional `jmp schedule loop`** | Quick local polling when installing a system task is inconvenient. Keep the process alive yourself; use `--max-ticks` for tests. |

We deliberately **do not** embed a full cron expression engine. Schedule *when*
with the OS; JMP only decides *what* to process on each tick (discover + dedupe
against the job store).

## Dedupe and failures

Each tick reuses batch classification:

- **Succeeded** pairs are never reprocessed (even with `--retry`).
- **Pending / running** pairs are skipped.
- **Failed / cancelled** pairs stay skipped unless `--retry` or
  `JMP_SCHEDULE_RETRY_FAILED=true`.
- Failures remain visible in `JMP_WORK_DIR/jobs/<id>/job.json` (or
  `JMP_BATCH_TEMP_DIR/jobs/…`) and in stderr logs tagged
  `schedule.tick.failure`.

Preview without running jobs:

```bash
uv run jmp schedule plan path/to/incoming -o path/to/outgoing
uv run jmp schedule tick path/to/incoming -o path/to/outgoing --dry-run
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `JMP_BATCH_INPUT_DIR` | unset | Default input root |
| `JMP_BATCH_OUTPUT_DIR` | unset | Default output root |
| `JMP_BATCH_TEMP_DIR` | unset | Optional jobs/batches root |
| `JMP_SCHEDULE_INTERVAL_SECONDS` | `3600` | Default `loop` sleep |
| `JMP_SCHEDULE_RETRY_FAILED` | `false` | Re-queue failed pairs on ticks |
| `JMP_LOG_LEVEL` | `INFO` | Unattended log verbosity |

```bash
uv run jmp schedule show
```

## CLI

```bash
# One tick (OS scheduler entry point)
uv run jmp schedule tick path/to/incoming -o path/to/outgoing

# Optional in-process loop
uv run jmp schedule loop path/to/incoming -o path/to/outgoing --interval 3600

# Limited loop for smoke tests
uv run jmp schedule loop … --max-ticks 2 --dry-run
```

## OS scheduler examples

Assume the repo is at `/opt/jmp` (Linux) or `C:\jmp` (Windows), with
`JMP_BATCH_INPUT_DIR` / `JMP_BATCH_OUTPUT_DIR` set in the environment or `.env`.

### cron

```cron
# Every night at 02:15
15 2 * * * cd /opt/jmp && /usr/bin/uv run jmp schedule tick >>/var/log/jmp-schedule.log 2>&1
```

### systemd timer

`/etc/systemd/system/jmp-schedule.service`:

```ini
[Unit]
Description=Jarvis Media Processor scheduled tick
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/opt/jmp
Environment=JMP_BATCH_INPUT_DIR=/data/incoming
Environment=JMP_BATCH_OUTPUT_DIR=/data/outgoing
Environment=JMP_LOG_LEVEL=INFO
ExecStart=/usr/bin/uv run jmp schedule tick
```

`/etc/systemd/system/jmp-schedule.timer`:

```ini
[Unit]
Description=Run JMP schedule tick overnight

[Timer]
OnCalendar=*-*-* 02:15:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now jmp-schedule.timer
```

### Windows Task Scheduler

1. Create a Basic Task → trigger **Daily** at the desired time.
2. Action: **Start a program**
   - Program: `uv` (full path if needed, e.g. `%USERPROFILE%\.local\bin\uv.exe`)
   - Arguments: `run jmp schedule tick`
   - Start in: `C:\jmp` (repo root with `.env` or env vars set on the task)
3. Optionally append `>> C:\jmp\logs\schedule.log 2>&1` via `cmd /c` if you want a file log.
4. Configure the task to run whether the user is logged on or not, with the
   account that can read input media and write the output root.

PowerShell one-liner to register a daily task (adjust paths):

```powershell
$action = New-ScheduledTaskAction -Execute "uv" -Argument "run jmp schedule tick" -WorkingDirectory "C:\jmp"
$trigger = New-ScheduledTaskTrigger -Daily -At 2:15am
Register-ScheduledTask -TaskName "JMP-Schedule-Tick" -Action $action -Trigger $trigger
```
