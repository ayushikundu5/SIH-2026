#!/bin/bash
# short32: width 32 at a 320-point window / 160 hop. Separates "shorter window"
# from "smaller model" in the 0.173 PESQ that short24 cost. See
# configs/train_short32.yaml for the reasoning and the prediction.
#
# Self-healing: every attempt passes --resume, so a crash or reboot costs at
# most one epoch. Closing the lid PAUSES it; it continues when the laptop wakes.
# To stop on purpose: create C:\SIH26052_data\STOP_SHORT32, then kill python.
#
# SIH_NFFT / SIH_HOP must be exported HERE rather than inherited: a process
# created through WMI gets the WMI service's environment, not the environment of
# whoever asked for it. src/train.py prints the framing on every start and the
# checkpoint records it, so a mistake is visible rather than silent.
export SIH_NFFT=320
export SIH_HOP=160
export SIH_DATA_ROOT=/root/sih/data
R=/mnt/c/SIH26052_data/wsl_run.sh
LOG=/mnt/c/SIH26052_data/train_short32.log
STOP=/mnt/c/SIH26052_data/STOP_SHORT32
for attempt in $(seq 1 30); do
  [ -f "$STOP" ] && { echo "=== STOP_SHORT32 present, not starting $(date -Is)" >> $LOG; break; }
  echo "=== start attempt $attempt $(date -Is)  SIH_NFFT=$SIH_NFFT SIH_HOP=$SIH_HOP" >> $LOG
  bash $R -m src.train --config configs/train_short32.yaml --data-config configs/data_combat.yaml \
       --manifest manifests/manifest_combat.json --tag short32 --resume "$@" >> $LOG 2>&1
  rc=$?
  echo "=== exit $rc $(date -Is)" >> $LOG
  [ $rc -eq 0 ] && break
  [ -f "$STOP" ] && break
  sleep 60
done
