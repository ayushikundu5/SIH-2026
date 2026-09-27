# ONE COMMAND to start the short32 run and everything that looks after it.
#
#   Right-click this file > "Run with PowerShell"      (double-click opens Notepad)
#   or:  powershell -ExecutionPolicy Bypass -File C:\SIH26052_data\start_short32.ps1
#
# It starts four things, all through the WMI service so they do NOT belong to any
# editor or Claude session and survive them closing (processes a Claude Code
# session starts live in that session's Windows job object and die with it - that
# stopped training twice on 22 Sep):
#
#   1. the self-healing training loop, which also keeps Windows awake while it
#      runs (--resume every attempt, so a crash costs at most one epoch)
#   2. the hang watchdog, which kills and lets the loop restart training if CUDA
#      wedges or the log goes silent for 5 minutes
#   3. the phone notifier (ntfy)
#   4. the automatic finish: on a clean end it exports to streaming ONNX, scores
#      all three test sets, benchmarks latency/RTF and messages the phone
#
# Safe to run twice: anything already running is left alone, not duplicated.
#
# TO STOP EVERYTHING:  New-Item C:\SIH26052_data\STOP_SHORT32 -ItemType File
# then kill python if you want it to stop immediately rather than after the epoch.
#
# Expected: ~13 hours for 170 epochs at about 4.7 min each, or less if it
# early-stops the way short24 did (epoch 149).

$ErrorActionPreference = "Stop"

function Running($pattern) {
    # "[t]rain..." matches the real process but not this check's own command
    # line, which contains the literal "[t]rain...". Without the bracket trick
    # pgrep counts its own parent shell and always reports "running".
    $p = "[" + $pattern.Substring(0, 1) + "]" + $pattern.Substring(1)
    $n = wsl.exe -d Ubuntu-24.04 -u root -- bash -c "pgrep -fc '$p' || true"
    return ([int]($n | Select-Object -Last 1)) -gt 0
}

function Launch($cmd) {
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
    if ($r.ReturnValue -ne 0) { throw "could not start: $cmd (code $($r.ReturnValue))" }
    "started pid $($r.ProcessId): $cmd"
}

if (Test-Path C:\SIH26052_data\STOP_SHORT32) {
    "STOP_SHORT32 is present - delete it first if you really want training to run."
    exit 1
}
if (Test-Path C:\SIH26052_data\short32_finished.marker) {
    "short32_finished.marker is present - this run already completed."
    "Delete the marker to run it again (and rename the old checkpoints first)."
    exit 1
}

# Sanity checks before committing the machine to 13 hours, because each of these
# has actually gone wrong once: torch blocked by Smart App Control, the GPU
# missing, or the data root not mounted in WSL.
"checking the environment..."
$t = wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/SIH26052_data/wsl_run.sh -c `
     "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"  $t"
if ($t -notmatch "cuda True") { throw "CUDA not available in WSL - fix that before starting (see RESUME.md)" }
$d = wsl.exe -d Ubuntu-24.04 -u root -- bash -c "ls /root/sih/data > /dev/null 2>&1 && echo ok || echo MISSING"
if ($d -notmatch "ok") { throw "/root/sih/data missing in WSL - the mirrored dataset is what makes training fast" }
"  data root ok"

if (Running "train_short32_loop") { "training loop already running - left alone" }
else { Launch 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\SIH26052_data\run_short32_keep_awake.ps1' }

if (Running "train_watchdog") { "watchdog already running - left alone" }
else { Launch ('wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/SIH26052_data/train_watchdog_v2.sh ' +
               '/mnt/c/SIH26052_data/train_short32.log "[s]rc.train --config configs/train_short32.yaml"') }

if (Running "phone_updates") { "phone notifier already running - left alone" }
else { Launch 'wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/SIH26052_data/phone_updates_short32.sh --quiet' }

$af = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
      Where-Object { $_.CommandLine -match "auto_finish_any.*short32" }
if ($af) { "auto-finish already running - left alone" }
else { Launch 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\SIH26052_data\auto_finish_any.ps1 -Tag short32 -Nfft 320 -Hop 160' }

""
"short32 is starting. Watch it with:"
"  Get-Content C:\SIH26052_data\train_short32.log -Tail 5 -Wait"
"Results land in C:\SIH26052_data\short32_summary.txt and on the phone."
