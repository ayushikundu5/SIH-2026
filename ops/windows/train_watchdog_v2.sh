#!/bin/bash
# Kills a training process that has HUNG, so the self-healing loop
# (train_combat32_loop.sh) restarts it with --resume. A hung process never
# exits, so the loop alone cannot recover it. Replaces train_watchdog.sh
# (which waited ~20 min before acting).
#
# Two triggers:
#  1. the log shows a CUDA error after the most recent (re)start - seen twice
#     on 22 Sep 2026 (15:46, 22:37): "CUDA error: unknown error", after which
#     the process hangs forever instead of exiting. Acted on within ~1 min.
#  2. the log has been silent > 5 min on 4 consecutive 1-min checks. A step
#     line is written every ~20 s and validation takes ~1 min, so that much
#     silence is never normal; requiring 4 checks means waking from a
#     lid-close (log briefly old, training resumes within ~1 min) never trips it.
# At most 8 kills, then it gives up and leaves it to a human.
LOG=${1:-/mnt/c/SIH26052_data/train_combat32.log}
PAT=${2:-[s]rc.train --config configs/train_combat.yaml}
kills=0; stale=0
while [ $kills -lt 8 ]; do
  sleep 60
  pgrep -f "$PAT" > /dev/null || { stale=0; continue; }
  last_start=$(grep -n "=== start attempt" "$LOG" | tail -1 | cut -d: -f1)
  if tail -n +"${last_start:-1}" "$LOG" | grep -q "CUDA error"; then
    echo "=== watchdog: CUDA error in log, killing hung training $(date -Is)" >> "$LOG"
    pkill -9 -f "$PAT"; kills=$((kills+1)); stale=0; sleep 120; continue
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ $age -gt 300 ]; then stale=$((stale+1)); else stale=0; fi
  if [ $stale -ge 4 ]; then
    echo "=== watchdog: log silent ${age}s, killing hung training $(date -Is)" >> "$LOG"
    pkill -9 -f "$PAT"; kills=$((kills+1)); stale=0; sleep 120
  fi
done
echo "=== watchdog: gave up after $kills kills $(date -Is)" >> "$LOG"
