#!/bin/bash
# Phone alerts for the short32 run. val_pesq milestones are OFF: this model
# frames at 320/160, so its validation PESQ is not comparable with the 512-point
# runs and a "milestone" would mean nothing. It IS comparable with short24.
export WATCH_LOG=/mnt/c/SIH26052_data/train_short32.log WATCH_NAME=short32 WATCH_EPOCHS=170 WATCH_MILESTONE=99 WATCH_BAR=n/a
exec bash /mnt/c/SIH26052_data/phone_updates.sh "$@"
