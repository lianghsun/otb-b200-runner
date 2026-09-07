#!/usr/bin/env bash
# Camp on a SLURM partition and submit the OTB B200 job the instant a node frees.
# Run this ON the NCHC login node:
#
#     PART=<b200_partition> ACCOUNT=<your_account> GPUS=8 HF_TOKEN=hf_xxx ./grab_b200.sh
#
# It polls `sinfo` for idle nodes; when it sees enough, it submits otb_b200.slurm
# once and stops. If you already have a job queued/running on that partition it
# exits without submitting, so it is safe to re-run.
set -uo pipefail

PART="${PART:?set PART to the B200 partition name (see: sinfo)}"
ACCOUNT="${ACCOUNT:?set ACCOUNT to your NCHC allocation/計畫 (see: sacctmgr show assoc user=$(whoami) format=account)}"
GPUS="${GPUS:-8}"                # GPUs to request (gres=gpu:$GPUS)
NEED="${NEED:-1}"                 # idle nodes required before pouncing
JOB="${JOB:-otb_b200.slurm}"     # sbatch script to submit (same dir)
INTERVAL="${INTERVAL:-30}"       # seconds between checks
ME="$(whoami)"

command -v sinfo  >/dev/null || { echo "no sinfo on PATH — run this on the NCHC login node"; exit 1; }
command -v sbatch >/dev/null || { echo "no sbatch on PATH — run this on the NCHC login node"; exit 1; }
[ -f "$JOB" ] || { echo "job script '$JOB' not found in $(pwd)"; exit 1; }

echo "camping on '$PART' (need >=$NEED idle node(s)); checking every ${INTERVAL}s — Ctrl-C to stop"
while true; do
  if squeue -h -u "$ME" -p "$PART" -o "%i" | grep -q .; then
    echo "$(date '+%F %T')  you already have a job on '$PART' — nothing to grab, exiting."
    exit 0
  fi
  idle="$(sinfo -h -p "$PART" -t idle -o '%D' 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  echo "$(date '+%F %T')  idle on '$PART': ${idle:-0}"
  if [ "${idle:-0}" -ge "$NEED" ]; then
    echo "$(date '+%F %T')  >=$NEED idle — submitting $JOB (part=$PART acct=$ACCOUNT gpu:$GPUS)"
    if sbatch --partition="$PART" --account="$ACCOUNT" --gres="gpu:$GPUS" "$JOB"; then
      echo "$(date '+%F %T')  submitted — stop camping. Watch it with: squeue -u $ME"
      exit 0
    fi
    echo "$(date '+%F %T')  sbatch failed (someone else grabbed it? bad config?) — retrying next tick"
  fi
  sleep "$INTERVAL"
done
