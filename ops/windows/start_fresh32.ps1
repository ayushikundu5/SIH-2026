# Starts (or restarts) the fresh32 training run and its helpers, fully
# independent of Claude / VS Code.
#
#   Right-click > "Run with PowerShell"   (double-click only opens Notepad)
#
# Why Win32_Process.Create instead of Start-Process: processes a Claude Code
# session starts - even via Start-Process - live inside that session's Windows
# job object and are killed when the session closes. That stopped training
# twice on 22 Sep 2026 (16:29 and 17:49). Processes created through the WMI
# service are not in that job, so they survive Claude and VS Code restarts.
#
# Safe to run twice: anything already running is left alone, not duplicated.

$ErrorActionPreference = "Stop"

function Running($pattern) {
    # "[t]rain..." matches "train..." but not this bash -c command line itself,
    # which contains the literal "[t]rain...". Without it pgrep counts its own
    # parent shell and always reports "running".
    $p = "[" + $pattern.Substring(0, 1) + "]" + $pattern.Substring(1)
    $n = wsl.exe -d Ubuntu-24.04 -u root -- bash -c "pgrep -fc '$p' || true"
    return ([int]($n | Select-Object -Last 1)) -gt 0
}

function Launch($cmd) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
    if ($r.ReturnValue -ne 0) { throw "could not start: $cmd (code $($r.ReturnValue))" }
    "started pid $($r.ProcessId): $cmd"
}

if (Test-Path C:\SIH26052_data\STOP_FRESH32) {
    "STOP_FRESH32 is present - delete it first if you really want training to run."
    exit 1
}

if (Running "train_fresh32_loop") { "training loop already running - left alone" }
else { Launch 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\SIH26052_data\run_fresh32_keep_awake.ps1' }

if (Running "train_watchdog") { "watchdog already running - left alone" }
else { Launch ('wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/SIH26052_data/train_watchdog_v2.sh ' +
               '/mnt/c/SIH26052_data/train_fresh32.log "[s]rc.train --config configs/train_fresh32.yaml"') }

if (Running "phone_updates") { "phone notifier already running - left alone" }
else { Launch 'wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/SIH26052_data/phone_updates_fresh32.sh --quiet' }
