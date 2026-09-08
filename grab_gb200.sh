#!/usr/bin/env bash
# Grab a free NCHC GB200 slot and submit the OTB runner. Run ON the login node:
#
#   HF_TOKEN=hf_xxx ./grab_gb200.sh                      # gb200-dev, pounce on 1 idle node
#   PART=gb200-r1 HF_TOKEN=hf_xxx ./grab_gb200.sh        # r1 (4 nodes, 10h backfill window)
#   MODE=queue PART=gb200-r1 HF_TOKEN=hf_xxx ./grab_gb200.sh   # just submit + let backfill schedule
#
# ONLY gb200-* is the free trial (H200 partitions are BILLED); this refuses any
# other partition, and otb_gb200.slurm additionally aborts on non-aarch64 nodes.
set -uo pipefail

ACCOUNT="${ACCOUNT:-ent115027}"     # the only usable Slurm account
PART="${PART:-gb200-dev}"           # gb200-dev | gb200-r1 | gb200-r2 (free only)
JOB="${JOB:-otb_gb200.slurm}"
MODE="${MODE:-pounce}"              # pounce = submit when enough nodes are idle; queue = submit once, rely on backfill
INTERVAL="${INTERVAL:-30}"
ME="$(whoami)"

# per-partition minimum job size (4 GPUs/node) + backfill-friendly time (< cap).
case "$PART" in
  gb200-dev) NODES="${NODES:-1}"; TIME="${TIME:-02:00:00}";;   # min 1 GPU, cap 2h
  gb200-r1)  NODES="${NODES:-4}"; TIME="${TIME:-10:00:00}";;   # min 16 GPU = 4 nodes, cap 24h
  gb200-r2)  NODES="${NODES:-8}"; TIME="${TIME:-08:00:00}";;   # min 32 GPU = 8 nodes, cap 12h
  *) echo "refusing PART='$PART' — only gb200-dev/r1/r2 are free (others are billed H200)"; exit 1;;
esac
GRES="gpu:GB200:4"

command -v sinfo >/dev/null && command -v sbatch >/dev/null || { echo "run on the NCHC login node (no sinfo/sbatch here)"; exit 1; }
[ -f "$JOB" ] || { echo "job script '$JOB' not found in $(pwd)"; exit 1; }

submit() {
  sbatch --account="$ACCOUNT" --partition="$PART" --nodes="$NODES" \
         --gres="$GRES" --time="$TIME" "$JOB"
}

# MaxSubmitJobsPU=2 is TOTAL (running + pending) across your gb200 jobs, not 2+2.
gb200_jobs() { squeue -h -u "$ME" -o '%P' | grep -c '^gb200' || true; }

if [ "$MODE" = "queue" ]; then
  if [ "$(gb200_jobs)" -ge 2 ]; then echo "already at the 2-job gb200 limit — not submitting."; exit 0; fi
  echo "queue mode: submitting to '$PART' (nodes=$NODES time=$TIME) and letting backfill schedule it"
  submit; exit $?
fi

echo "pounce on '$PART' (need $NODES idle node(s); gres=$GRES time=$TIME); every ${INTERVAL}s — Ctrl-C to stop"
while true; do
  mine="$(gb200_jobs)"
  if [ "$mine" -ge 2 ]; then echo "$(date '+%F %T')  at 2-job gb200 limit — exiting."; exit 0; fi
  idle="$(sinfo -h -p "$PART" -t idle -o '%D' | awk '{s+=$1} END{print s+0}')"
  echo "$(date '+%F %T')  idle nodes on '$PART': ${idle:-0} (need $NODES; you have $mine job(s))"
  if [ "${idle:-0}" -ge "$NODES" ]; then
    echo "$(date '+%F %T')  enough idle — submitting"
    if submit; then echo "$(date '+%F %T')  submitted — watch: squeue -u $ME"; exit 0; fi
    echo "$(date '+%F %T')  sbatch failed (grabbed / limit / config) — retry next tick"
  fi
  sleep "$INTERVAL"
done
