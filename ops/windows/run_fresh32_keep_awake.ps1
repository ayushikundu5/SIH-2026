# Runs the self-healing WSL training loop and keeps Windows awake while it runs.
# Changes no power settings; the request ends when this process exits.
#
# Why re-assert every minute: on 21 Sep 2026 the laptop (Modern Standby) went
# into standby on "Idle Timeout" 6 minutes BEFORE this request was first made,
# and a request cannot pull a machine out of standby - it only prevents entry.
# Holding it continuously from while the machine is awake is what works.
# ES_DISPLAY_REQUIRED keeps the screen on: on Modern Standby, screen-off idle
# leads to standby, which pauses training.
# Lid-close / power button / Start > Sleep still suspend it (training PAUSES
# and continues on wake; if it crashes instead, the loop resumes it).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
$awake = $ES_CONTINUOUS -bor [uint32]1 -bor [uint32]2   # + SYSTEM_REQUIRED + DISPLAY_REQUIRED
$p = Start-Process wsl.exe -ArgumentList '-d','Ubuntu-24.04','-u','root','--','bash','/mnt/c/SIH26052_data/train_fresh32_loop.sh' -WindowStyle Hidden -PassThru
while (-not $p.HasExited) {
    [Win32.Power]::SetThreadExecutionState($awake) | Out-Null
    Start-Sleep -Seconds 60
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
