#!/bin/bash
# Phone alerts for the fresh32 fine-tune. val_pesq milestones are OFF: adding
# the combat_real category changed the validation mixtures, so val_pesq is not
# comparable with the earlier runs' numbers.
export WATCH_LOG=/mnt/c/SIH26052_data/train_fresh32.log WATCH_NAME=fresh32 WATCH_EPOCHS=170 WATCH_MILESTONE=99 WATCH_BAR=n/a
exec bash /mnt/c/SIH26052_data/phone_updates.sh "$@"
