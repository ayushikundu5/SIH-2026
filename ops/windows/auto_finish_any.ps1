# Waits for a training run to print "done." and then finishes it by itself:
# export, all three test sets, benchmark, headline to the phone.
#
#   powershell -File C:\SIH26052_data\auto_finish_any.ps1 -Tag short32 -Nfft 320 -Hop 160
#
# Generic replacement for auto_finish_fresh32.ps1. Two differences that matter:
#
#  * it calls finish_any.ps1 WITHOUT a Tee-Object pipeline. Last night's hang
#    (CLAUDE.md trap 21) came from piping an evaluation through Tee; finish_any
#    redirects each step to its own file instead, and this script just writes its
#    own short log.
#  * it only acts on a CLEAN finish - "done. best val_pesq" in the log. A crash
#    does not trigger it, because the training loop restarts from the last
#    checkpoint and the watchdog handles hangs; finishing early on a crash would
#    score a half-trained model and send a misleading number.
#    NOTE: an EARLY STOP is a clean finish and does trigger it - short24 stopped
#    at epoch 149 of 170 because validation had not improved for 40 epochs, and
#    that model was correctly exported and scored.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [int]$Nfft = 512,
    [int]$Hop = 0
)
$ErrorActionPreference = "Continue"
$LOG = "C:\SIH26052_data\train_$Tag.log"
$OWN = "C:\SIH26052_data\auto_finish_$Tag.log"
$DONE_MARK = "C:\SIH26052_data\${Tag}_finished.marker"
$STOP = "C:\SIH26052_data\STOP_" + $Tag.ToUpper()

function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Out-File -Append -Encoding utf8 $OWN }

if (Test-Path $DONE_MARK) { Log "already finished earlier - nothing to do"; exit 0 }
Log "watching $LOG for a clean finish"

while ($true) {
    Start-Sleep -Seconds 120
    if (Test-Path $STOP) { Log "$STOP present - exiting without finishing"; exit 0 }
    if (-not (Test-Path $LOG)) { continue }
    $tail = Get-Content $LOG -Tail 40 -ErrorAction SilentlyContinue
    if ($tail -match "^done\. best val_pesq") {
        Start-Sleep -Seconds 30          # let the loop write its "=== exit 0"
        New-Item -ItemType File -Path $DONE_MARK -Force | Out-Null
        Log "clean finish detected - running finish_any.ps1"
        & C:\SIH26052_data\finish_any.ps1 -Tag $Tag -Nfft $Nfft -Hop $Hop |
            Out-File -Append -Encoding utf8 $OWN
        Log "finish_any.ps1 returned"
        break
    }
}
