#!/usr/bin/env bash
# Self-healing training loop for Linux. Every attempt passes --resume, so a
# crash, a driver hiccup or a reboot costs at most the epoch in progress
# instead of the run.
#
#   CFG=configs/train_wide48.yaml TAG=wide48 nohup bash scripts/run_training_loop.sh &
#
# Stop it on purpose:   touch STOP_wide48     (then kill the python process)
# Start it again later: the same nohup line - it picks up from the last epoch.
set -u

CFG=${CFG:-configs/train_wide48.yaml}
TAG=${TAG:-wide48}
DATA_CFG=${DATA_CFG:-configs/data_combat.yaml}
MANIFEST=${MANIFEST:-manifests/manifest_combat.json}
PY=${PY:-.venv/bin/python}
LOG=${LOG:-train_${TAG}.log}
STOP=${STOP:-STOP_${TAG}}

for attempt in $(seq 1 30); do
    if [ -f "$STOP" ]; then
        echo "=== $STOP present, not starting $(date -Is)" >> "$LOG"
        break
    fi
    echo "=== start attempt $attempt $(date -Is)" >> "$LOG"
    "$PY" -m src.train --config "$CFG" --data-config "$DATA_CFG" \
          --manifest "$MANIFEST" --tag "$TAG" --resume "$@" >> "$LOG" 2>&1
    rc=$?
    echo "=== exit $rc $(date -Is)" >> "$LOG"
    [ "$rc" -eq 0 ] && break          # 0 = training finished all its epochs
    [ -f "$STOP" ] && break
    sleep 60                          # crashed: wait, then resume
done
echo "=== loop done $(date -Is)" >> "$LOG"
