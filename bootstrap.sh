#!/bin/bash
# OpenTWBench — B200 (CUDA / vLLM) runner for the models the Mac/MLX pass left
# unrun. Public repo; the benchmark itself stays private (pulled from a private
# HF repo with your token), and results are only per-question hashes + booleans.
#
#   git clone https://github.com/lianghsun/otb-b200-runner.git
#   cd otb-b200-runner
#   HF_TOKEN=hf_xxx bash bootstrap.sh
#
# Optional:  TP=8 HF_TOKEN=... bash bootstrap.sh      # tensor-parallel over 8 GPUs
#            MODELS='Qwen/Qwen2.5-72B-Instruct' HF_TOKEN=... bash bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! set HF_TOKEN first (reads the private benchmark + gated models):"
  echo "   HF_TOKEN=hf_xxx bash bootstrap.sh"
  exit 1
fi
export HF_TOKEN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "== 1. Python venv"
PYBIN="$(command -v python3.12 || command -v python3.11 || command -v python3)"
"$PYBIN" -m venv .venv
./.venv/bin/pip -q install --upgrade pip

echo "== 2. vLLM + deps"
# vLLM pulls a matching torch. On Blackwell (B200, sm_100) you need a build with
# CUDA 12.8+ support; a recent vLLM wheel provides it. If import fails below,
# install the Blackwell-compatible vLLM/torch for THIS box, then rerun.
./.venv/bin/pip -q install vllm || echo "   (vLLM install returned nonzero — see note below)"
./.venv/bin/pip -q install pyarrow "huggingface_hub[hf_xet]" hf_xet || \
  ./.venv/bin/pip -q install pyarrow huggingface_hub hf_xet
./.venv/bin/pip -q install hf_transfer || echo "   (hf_transfer unavailable — Xet/LFS fallback)"
if ! ./.venv/bin/python -c "import vllm, torch; print('   vLLM', vllm.__version__, '| torch', torch.__version__, '| GPUs', torch.cuda.device_count())"; then
  echo "!! vLLM/torch not importable on this box (likely a Blackwell/CUDA build mismatch)."
  echo "   Install a B200-compatible vLLM+torch into ./.venv, then: bash bootstrap.sh"
  exit 1
fi

echo "== 3. background auto-push: commit+push new results every 5 min"
push_now(){
  git add results 2>/dev/null || true
  git -c user.email=lianghsunh@gmail.com -c user.name="Liang-Hsun Huang" \
    commit -q -m "results $(date +%H:%M)" 2>/dev/null || true
  if err=$(git push origin HEAD:b200-results 2>&1); then
    echo "   [auto-push $(date +%H:%M)] ok"
  else
    echo "   [auto-push $(date +%H:%M)] !! PUSH FAILED — results safe on disk, fix creds:"
    printf '%s\n' "$err" | sed 's/^/       /'
  fi
}
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  # self-update CODE from main WITHOUT touching results/ or resetting the branch
  git fetch -q origin main 2>/dev/null || true
  git checkout -q origin/main -- eval_vllm.py README.md 2>/dev/null || true
  if git show-ref -q --verify refs/heads/b200-results; then
    git checkout -q b200-results 2>/dev/null || true
  else
    git checkout -q -b b200-results 2>/dev/null || true
  fi
  ( while true; do sleep 300; push_now; done ) &
  PUSHER=$!
  trap 'kill $PUSHER 2>/dev/null || true; push_now' EXIT
  echo "   auto-push armed (pid $PUSHER) — results stream to branch b200-results"
fi

echo "== 4. run the eval (per model: formosa then exam, then reclaim its weights)"
if [ -n "${MODELS:-}" ]; then
  ./.venv/bin/python eval_vllm.py --models ${MODELS}
else
  ./.venv/bin/python eval_vllm.py
fi

echo "== 5. done. Results auto-pushed to the b200-results branch throughout."
