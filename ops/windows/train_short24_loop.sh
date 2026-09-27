#!/bin/bash
# short24: width 24 at a 320-point window / 160-sample hop - the run that tests
# whether the 32 ms latency target can be met without losing quality. Measured
# latency of this framing is 27.4 ms total (results/bench_framing_probe.json).
#
# Self-healing: every attempt passes --resume, so a crash or reboot costs at
# most one epoch. Closing the lid PAUSES it; it continues when the laptop wakes.
# To stop on purpose: create C:\SIH26052_data\STOP_SHORT24, then kill python.
#
# SIH_NFFT / SIH_HOP are what make this a short-window run. They must be
# exported HERE rather than inherited: a process created through WMI gets the
# WMI service's environment, not the environment of whoever asked for it.
# Unset would silently train yet another 512-point model - which is why
# src/train.py prints the framing on every start, and why the checkpoint
# records it.
export SIH_NFFT=320
export SIH_HOP=160
export SIH_DATA_ROOT=/root/sih/data
R=/mnt/c/SIH26052_data/wsl_run.sh
LOG=/mnt/c/SIH26052_data/train_short24.log
STOP=/mnt/c/SIH26052_data/STOP_SHORT24
for attempt in $(seq 1 30); do
  [ -f "$STOP" ] && { echo "=== STOP_SHORT24 present, not starting $(date -Is)" >> $LOG; break; }
  echo "=== start attempt $attempt $(date -Is)  SIH_NFFT=$SIH_NFFT SIH_HOP=$SIH_HOP" >> $LOG
  bash $R -m src.train --config configs/train_short24.yaml --data-config configs/data_combat.yaml \
       --manifest manifests/manifest_combat.json --tag short24 --resume "$@" >> $LOG 2>&1
  rc=$?
  echo "=== exit $rc $(date -Is)" >> $LOG
  [ $rc -eq 0 ] && break
  [ -f "$STOP" ] && break
  sleep 60
done
