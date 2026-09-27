#!/bin/bash
# fresh32: wide32 fine-tuned with the real combat-noise category. Self-healing:
# every attempt passes --resume, so a crash or reboot costs at most one epoch.
# Closing the lid PAUSES it; it continues when the laptop wakes.
# To stop on purpose: create C:\SIH26052_data\STOP_FRESH32, then kill python.
export SIH_DATA_ROOT=/root/sih/data
R=/mnt/c/SIH26052_data/wsl_run.sh
LOG=/mnt/c/SIH26052_data/train_fresh32.log
STOP=/mnt/c/SIH26052_data/STOP_FRESH32
for attempt in $(seq 1 30); do
  [ -f "$STOP" ] && { echo "=== STOP_FRESH32 present, not starting $(date -Is)" >> $LOG; break; }
  echo "=== start attempt $attempt $(date -Is)  SIH_DATA_ROOT=$SIH_DATA_ROOT" >> $LOG
  bash $R -m src.train --config configs/train_fresh32.yaml --data-config configs/data_combat.yaml \
       --manifest manifests/manifest_combat.json --tag fresh32 --resume "$@" >> $LOG 2>&1
  rc=$?
  echo "=== exit $rc $(date -Is)" >> $LOG
  [ $rc -eq 0 ] && break
  [ -f "$STOP" ] && break
  sleep 60
done
