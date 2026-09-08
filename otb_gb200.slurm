#!/usr/bin/env bash
# SLURM batch job for the OpenTWBench runner on NCHC GB200 (Grace-Blackwell, Arm).
#
# ONLY gb200-* partitions are the free trial; H200 partitions are BILLED. The
# #SBATCH defaults below target the free gb200-dev; grab_gb200.sh overrides
# partition/nodes/gres/time on the command line. The architecture guard makes a
# wrong partition (that lands on x86_64 H200) ABORT instead of billing you.
#
#   sbatch otb_gb200.slurm                    # gb200-dev, 1 node, 2h
#   (or drive it via ./grab_gb200.sh)
#SBATCH --job-name=otb-gb200
#SBATCH --account=ent115027
#SBATCH --partition=gb200-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:GB200:4
#SBATCH --time=02:00:00
#SBATCH --output=otb-gb200-%j.log
set -euo pipefail

# === architecture guard: GB200 = aarch64 (free) · H200 = x86_64 (BILLED) ===
# A gb200-/dev prefix typo can land on an H200 node; refuse to run there.
arch="$(uname -m)"
if [ "$arch" != "aarch64" ]; then
  echo "ABORT: uname -m=$arch is not aarch64 — this is NOT a GB200 node." >&2
  echo "       H200 partitions are billed; refusing to run. Check --partition." >&2
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${HF_TOKEN:?set HF_TOKEN (scoped to the private dataset repo) before submitting}"
export RESULTS_BRANCH="${RESULTS_BRANCH:-b200-results}"
export TP="${TP:-$(nvidia-smi -L 2>/dev/null | wc -l)}"   # tensor-parallel over allocated GPUs

echo "=== OTB GB200 job on $(hostname) [$arch] — $(date) ==="
nvidia-smi -L || true
bash bootstrap.sh
