# Unattended training on the Windows laptop

These are the scripts that actually produced the measured models in this repo.
They live here for provenance and so the setup survives the machine: every
number in `README.md` for `fresh32` and `short24` came out of this automation
running for 13-16 hours unattended, and rebuilding it from scratch took a week of
finding out why jobs died.

**They are copies.** The running originals are in `C:\SIH26052_data\`, outside
this repo and outside OneDrive, because they reference absolute paths and the
project directory is synced. Edit them there and copy back, or the version that
runs and the version that is committed will drift.

**Nothing here contains a secret.** The ntfy topic and the team Drive folder id
are read at runtime from `C:\SIH26052_data\ntfy_topic.txt` and
`drive_folder_id.txt`, which are deliberately not in git. Verified before
committing rather than assumed.

## Start a run

```powershell
powershell -ExecutionPolicy Bypass -File ops\windows\start_short32.ps1
```

`start_*.ps1` is the only entry point you need. It launches four things and is
safe to run twice (anything already running is left alone):

| | what it does |
|---|---|
| `run_*_keep_awake.ps1` | runs the training loop and holds the machine awake while it runs |
| `train_*_loop.sh` | the self-healing loop: every attempt passes `--resume`, so a crash costs at most one epoch |
| `train_watchdog_v2.sh` | kills training that has HUNG so the loop can restart it |
| `auto_finish_any.ps1` | on a clean finish: export, all three test sets, benchmark, phone alert |

Stop a run by creating the stop file the loop checks, e.g.
`New-Item C:\SIH26052_data\STOP_SHORT32 -ItemType File`.

## Four things in here are load-bearing, and all four were learned the hard way

**1. Jobs must be launched through WMI, not `Start-Process`.** A process started
by a Claude Code session - even detached - belongs to that session's Windows job
object and is killed when the session closes. That silently stopped training
twice on 22 Sep 2026 (16:29 and 17:49). `Invoke-CimMethod Win32_Process Create`
parents the process to the WMI service instead, so it survives the editor, the
terminal and the session closing. `Launch()` in every `start_*.ps1` does this.

**2. Never pipe an evaluation through `Tee-Object`.** On 24 Sep the finish chain
stopped dead for 24 minutes with eleven worker processes frozen at ~25 s of CPU
each and nothing raised: the parent was blocked writing progress-bar output into
a PowerShell pipeline stage that had stopped draining, so it stopped feeding the
pool. `finish_any.ps1` redirects every step to its own file (`*> $log`) instead.
Diagnose a suspected hang by **CPU time per worker**, not by the log - frozen CPU
means a blocked parent, climbing CPU means genuinely slow.

**3. A hung job needs a watchdog, not just a retry loop.** The loop restarts a
process that EXITS; it cannot help one that hangs forever, which is what CUDA
did three times in WSL ("CUDA error: unknown error", then a wedged process).
`train_watchdog_v2.sh` kills on a logged CUDA error after the last restart, or
after 5 minutes of log silence confirmed over 4 consecutive checks - the repeat
check is what stops a lid-close from tripping it.

**4. `pgrep -fc` inside `bash -c` matches its own command line.** Every check
here uses the bracket trick (`[t]rain_short32_loop`), which matches the real
process but not the checking shell. Without it the check always reports
"already running" and the start script silently does nothing.

Two smaller ones: environment variables must be exported *inside* the loop
script, because a process created through WMI inherits the WMI service's
environment and not the caller's; and `$ErrorActionPreference = "Stop"` is
deliberately NOT set in the finish scripts, because PowerShell 5.1 turns a native
command's stderr into an error record and would abort on the ONNX exporter's
harmless "opset 18" notice. Exit codes are checked instead.

## Adapting this to another machine

Paths to change: the project directory and `C:\SIH26052_data` in every file, and
the WSL distro name (`Ubuntu-24.04`). `wsl_run.sh` is the single place that knows
where the project and the WSL venv are. On Linux none of this is needed - use
`scripts/run_training_loop.sh`, which is the same self-healing idea in one file.

## Not kept here

The per-run finish scripts (`finish_fresh32.ps1`, `finish_short24.ps1`,
`auto_finish_fresh32.ps1`) and `chain_short24.ps1` were superseded by
`finish_any.ps1` / `auto_finish_any.ps1`, which take `-Tag` and `-Nfft`/`-Hop`
and work for any checkpoint including one arriving from the second machine. The
`combat32` copies are gone the same way. They are not in git; if you need to know
exactly what produced a given number, the generic scripts plus the run's config
in `configs/` say it.
