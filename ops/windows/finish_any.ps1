# Export a checkpoint to streaming ONNX, score it on all three test sets,
# benchmark it, and send the headline to the phone. Works for ANY checkpoint -
# one of ours, or one that arrives from the second machine.
#
#   powershell -File C:\SIH26052_data\finish_any.ps1 -Tag short32 -Nfft 320 -Hop 160
#   powershell -File C:\SIH26052_data\finish_any.ps1 -Tag wide48
#
# It expects checkpoints/<Tag>_best.pt and writes artifacts/model_<Tag>.onnx,
# results/per_clip_*.csv rows for it, results/bench_edge_<Tag>.json and
# C:\SIH26052_data\<Tag>_summary.txt.
#
# This replaces finish_fresh32.ps1 and finish_short24.ps1, which were the same
# script copied twice.
#
# TWO THINGS IN HERE ARE LOAD-BEARING, both learned the hard way:
#
# 1. Every step's output is REDIRECTED TO A FILE (`*> $log`), never piped
#    through Tee-Object or any other PowerShell pipeline stage. On 24 Sep the
#    824-clip evaluation hung for 24 minutes because the parent process blocked
#    writing its progress bar into a Tee-Object that stopped draining, which
#    stalled all eleven workers. It does not raise, it does not time out, and it
#    looks exactly like a slow run. See CLAUDE.md measurement trap 21.
#
# 2. No $ErrorActionPreference = "Stop". In PowerShell 5.1 a native command
#    writing to stderr becomes an error record, and "Stop" then aborts on
#    harmless warnings - the ONNX exporter's "opset 18" notice killed an earlier
#    version of this. Exit codes are checked instead.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [int]$Nfft = 512,
    [int]$Hop = 0,                 # 0 means Nfft/2
    [int]$Workers = 11,
    [switch]$SkipBench             # for a machine that is busy; bench needs idle
)
$ErrorActionPreference = "Continue"
$PROJ = "C:\Users\Ayushi Kundu\OneDrive\Desktop\SIH_2026"
$PY = "C:\SIH26052_data\.venv\Scripts\python.exe"
$LOGDIR = "C:\SIH26052_data\finish_$Tag"
if ($Hop -eq 0) { $Hop = [int]($Nfft / 2) }
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
Set-Location $PROJ

# The framing must match what the checkpoint was TRAINED at. Getting it wrong
# cannot corrupt a result quietly: src.export_onnx reads the window size out of
# the weights and refuses a mismatch, and StreamingEnhancer refuses a graph whose
# bin count disagrees with its own STFT (CLAUDE.md invariant 12a).
$env:SIH_NFFT = "$Nfft"
$env:SIH_HOP = "$Hop"

function Notify($title, $msg, $prio = "default") {
    try {
        $topic = (Get-Content C:\SIH26052_data\ntfy_topic.txt -Raw).Trim()
        Invoke-RestMethod -Uri "https://ntfy.sh/$topic" -Method Post -Body $msg `
            -Headers @{ Title = $title; Priority = $prio } -TimeoutSec 20 | Out-Null
    } catch { "ntfy failed: $_" }
}

function Step($name, $file, $block) {
    "=== $name  $(Get-Date -Format HH:mm:ss)  -> $file"
    & $block
    if ($LASTEXITCODE -ne 0) {
        Notify "SIH $Tag : FAILED" "$name failed (exit $LASTEXITCODE). See $file" "high"
        throw "$name failed (exit $LASTEXITCODE) - see $file"
    }
}

$ckpt = "$PROJ\checkpoints\${Tag}_best.pt"
if (-not (Test-Path $ckpt)) { throw "no checkpoint at $ckpt" }
"finishing $Tag  (n_fft $Nfft, hop $Hop)  checkpoint $(Split-Path $ckpt -Leaf)"
Notify "SIH $Tag : evaluating" "Training done. Export + three test sets + benchmark, about 20 min."

# results/onnx_verify.json describes whatever was exported last; keep the
# shipped model's copy and save this run's beside it.
Copy-Item "$PROJ\results\onnx_verify.json" "$LOGDIR\onnx_verify.shipped.json" -Force -ErrorAction SilentlyContinue
$log = "$LOGDIR\1_export.log"
Step "1/5 export (WSL)" $log {
    wsl.exe -d Ubuntu-24.04 -u root -- env SIH_NFFT=$Nfft SIH_HOP=$Hop `
        bash /mnt/c/SIH26052_data/wsl_run.sh -m src.export_onnx `
        --ckpt "checkpoints/${Tag}_best.pt" --out "artifacts/model_$Tag.onnx" *> $log
}
Move-Item "$PROJ\results\onnx_verify.json" "$PROJ\results\onnx_verify_$Tag.json" -Force
Copy-Item "$LOGDIR\onnx_verify.shipped.json" "$PROJ\results\onnx_verify.json" -Force -ErrorAction SilentlyContinue

$M = "onnx:artifacts/model_${Tag}_simple.onnx"
$log = "$LOGDIR\2_defence.log"
Step "2/5 frozen defence set (720 clips)" $log {
    & $PY -m src.evaluate --workers $Workers --tag onnx --methods $M *> $log }
$log = "$LOGDIR\3_combat.log"
Step "3/5 real-combat set (150 clips)" $log {
    & $PY -m src.evaluate --workers $Workers --tag combat `
        --testset C:/SIH26052_data/testset_combat --methods $M *> $log }
$log = "$LOGDIR\4_vbd.log"
Step "4/5 VoiceBank-DEMAND (824 clips)" $log {
    & $PY -m src.evaluate --workers $Workers --tag vbd_onnx `
        --testset C:/SIH26052_data/voicebank_demand --methods $M *> $log }

if (-not $SkipBench) {
    # Benchmark wants an IDLE machine, and "idle" includes OneDrive: with it
    # syncing at ~1 core, RTF went 0.476 -> 0.520 and the worst frame 16.2 ->
    # 45.2 ms (CLAUDE.md invariant 13). Warn rather than silently mis-measure.
    $od = Get-Process OneDrive -ErrorAction SilentlyContinue
    $c1 = if ($od) { ($od | Measure-Object CPU -Sum).Sum } else { 0 }
    Start-Sleep -Seconds 6
    $od = Get-Process OneDrive -ErrorAction SilentlyContinue
    $c2 = if ($od) { ($od | Measure-Object CPU -Sum).Sum } else { 0 }
    $busy = ($c2 - $c1) -gt 1.5
    if ($busy) { "NOTE: OneDrive is using CPU - the speed numbers will be an upper bound" }
    $log = "$LOGDIR\5_bench.log"
    Step "5/5 real-time benchmark" $log {
        & $PY scripts\bench_edge.py --onnx "artifacts\model_${Tag}_simple.onnx" `
            --frames 800 --threads 1 --out "results\bench_edge_$Tag.json" *> $log }
    "onedrive_busy_during_bench=$busy" | Out-File -Encoding utf8 "$LOGDIR\bench_conditions.txt"
}

$line = & $PY scripts\summarize_results.py --oneline --highlight $M
& $PY scripts\summarize_results.py --highlight $M |
    Out-File -Encoding utf8 "C:\SIH26052_data\${Tag}_summary.txt"
$lat = ""
if (Test-Path "results\bench_edge_$Tag.json") {
    $b = Get-Content "results\bench_edge_$Tag.json" -Raw | ConvertFrom-Json
    $lat = " | latency $([math]::Round($b.latency_budget_ms.total_p95, 2)) ms, RTF $([math]::Round($b.onnx_runs[0].rtf_mean, 3))"
}
"$line$lat"
Notify "SIH $Tag : results ready" "$line$lat" "high"
"=== done $(Get-Date -Format HH:mm:ss). Table: C:\SIH26052_data\${Tag}_summary.txt  Logs: $LOGDIR"
