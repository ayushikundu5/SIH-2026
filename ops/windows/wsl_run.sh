#!/bin/bash
# Run a command from the SIH project directory with the WSL venv's python.
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/SIH26052_data/wsl_run.sh -m pytest tests -q
cd "/mnt/c/Users/Ayushi Kundu/OneDrive/Desktop/SIH_2026" || exit 1
export PYTHONUNBUFFERED=1
exec /root/sih/.venv/bin/python "$@"
