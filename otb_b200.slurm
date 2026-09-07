#!/usr/bin/env bash
# SLURM batch job for the OpenTWBench B200 runner.
#
# Partition / account / gres are NOT hard-coded here (they are cluster-specific
# and #SBATCH lines can't read env vars). Pass them on the sbatch command line,
# which overrides any #SBATCH directive:
#
#   sbatch --partition=<b200> --account=<acct> --gres=gpu:8 otb_b200.slurm
#
# Or just run ./grab_b200.sh, which builds that command line for you.
#SBATCH --job-name=otb-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=otb-b200-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${HF_TOKEN:?set HF_TOKEN (scoped to the private dataset repo) before submitting}"
export RESULTS_BRANCH="${RESULTS_BRANCH:-b200-results}"
# tensor-parallel over the GPUs SLURM actually gave us
export TP="${TP:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

echo "=== OTB B200 job on $(hostname) — $(date) ==="
nvidia-smi -L || true
bash bootstrap.sh
