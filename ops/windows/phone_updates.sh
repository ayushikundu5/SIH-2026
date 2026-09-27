#!/bin/bash
# Phone notifications for the wide32 run via ntfy.sh, independent of Claude.
# Runs in WSL beside the training. Sends:
#   - immediately: crash / NaN / loop exit / finished / log stalled / new PESQ milestone
#   - every 2 h: a status line (skipped 00:00-07:00 IST unless something is wrong)
# Topic (keep private - anyone who knows it can read the messages):
TOPIC=$(tr -d '\r\n' < /mnt/c/SIH26052_data/ntfy_topic.txt)
L=${WATCH_LOG:-/mnt/c/SIH26052_data/train_wide32.log}
BAR=${WATCH_BAR:-1.788}
send() { curl -s -m 20 -H "Title: $1" -H "Priority: ${3:-default}" -d "$2" "https://ntfy.sh/$TOPIC" > /dev/null || true; }
status() {
  local last best ep
  last=$(grep -E "^epoch " "$L" | tail -1)
  ep=$(echo "$last" | awk '{print $2+1}')
  best=$(grep -oE "new best val_pesq=[0-9.]+" "$L" | tail -1 | cut -d= -f2)
  echo "Epoch ${ep:-?}/${WATCH_EPOCHS:-180}. Best val PESQ ${best:-?} (bar ${BAR}). Last: $(echo "$last" | grep -oE 'val_pesq=[0-9.]+ +val_stoi=[0-9.]+' | tr -s ' ')"
}
start=$(wc -l < "$L"); stale=0; next_status=$(( $(date +%s) + 7200 )); milestone=${WATCH_MILESTONE:-1.90}
[ "$1" = "--quiet" ] || send "SIH ${WATCH_NAME:-training}: phone alerts on" "$(status)"
while true; do
  new=$(tail -n +$((start+1)) "$L")
  start=$((start + $(printf "%s" "$new" | grep -c '')))
  if echo "$new" | grep -qiE "Traceback|CUDA error|out of memory|Killed|val=nan|train=nan"; then
    send "SIH ${WATCH_NAME:-training}: ERROR" "$(echo "$new" | grep -iE 'Error|Killed|nan' | tail -2) - the loop will retry; check the laptop." high
  fi
  if echo "$new" | grep -q "done\."; then
    send "SIH ${WATCH_NAME:-training}: FINISHED" "$(status). Next: export + test-set evaluation." high; exit 0
  fi
  if echo "$new" | grep -q "=== exit"; then
    sleep 90
    tail -n 3 "$L" | grep -q "=== start" || send "SIH ${WATCH_NAME:-training}: STOPPED" "Training exited and has not restarted. $(status)" high
  fi
  best=$(grep -oE "new best val_pesq=[0-9.]+" "$L" | tail -1 | cut -d= -f2)
  if [ -n "$best" ] && awk -v b="$best" -v t="$milestone" 'BEGIN{exit !(b>=t)}'; then
    send "SIH ${WATCH_NAME:-training}: PESQ passed $milestone" "$(status)"
    milestone=$(awk -v t="$milestone" 'BEGIN{printf "%.2f", t+0.05}')
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$L") ))
  if [ $age -gt 600 ]; then stale=$((stale+1)); else stale=0; fi
  if [ $stale -eq 2 ]; then send "SIH ${WATCH_NAME:-training}: STALLED" "No progress for $((age/60)) min. Is the laptop asleep or the lid closed?" high; fi
  now=$(date +%s)
  if [ $now -ge $next_status ]; then
    hour=$(TZ=Asia/Kolkata date +%-H)
    if [ $hour -ge 7 ]; then send "SIH ${WATCH_NAME:-training}: update" "$(status)"; fi
    next_status=$(( now + 7200 ))
  fi
  sleep 180
done
