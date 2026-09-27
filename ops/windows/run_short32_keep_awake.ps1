# Runs the self-healing short32 training loop and keeps Windows awake while it
# runs. Changes no power settings; the request ends when this process exits, so
# the machine goes back to its normal 15-minute sleep as soon as training stops.
# See run_fresh32_keep_awake.ps1 for why the request is re-asserted every minute.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
$awake = $ES_CONTINUOUS -bor [uint32]1 -bor [uint32]2   # + SYSTEM_REQUIRED + DISPLAY_REQUIRED
$p = Start-Process wsl.exe -ArgumentList '-d','Ubuntu-24.04','-u','root','--','bash','/mnt/c/SIH26052_data/train_short32_loop.sh' -WindowStyle Hidden -PassThru
while (-not $p.HasExited) {
    [Win32.Power]::SetThreadExecutionState($awake) | Out-Null
    Start-Sleep -Seconds 60
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
