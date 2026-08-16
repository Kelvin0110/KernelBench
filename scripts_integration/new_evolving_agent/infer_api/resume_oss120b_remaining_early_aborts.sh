#!/usr/bin/env bash
# Resume remaining unsuccessful OSS-120b problems on one GPU, sequential:
#   1) deletion+merge+refine subsets 9 (L1P56) and 41 (L3P32)
#   2) markov subset 18 (L2P42, 504 abort)
#
#   bash scripts_integration/new_evolving_agent/infer_api/resume_oss120b_remaining_early_aborts.sh [GPU]
#
# Default GPU=0. If that GPU is 0% util but holds idle memory, set
# RESUME_ALLOW_BUSY_GPU=1 (this wrapper does that when util<=5%).

set -euo pipefail

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -x /usr/local/cuda-12.6/bin/nvcc ]; then
  export CUDA_HOME="/usr/local/cuda-12.6"
elif [ -x "$HOME/opt/cuda-12.8/bin/nvcc" ]; then
  export CUDA_HOME="$HOME/opt/cuda-12.8"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME="/usr/local/cuda"
fi
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$REPO_ROOT/.venv/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

command -v nvcc >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=${CUDA_HOME:-unset})"; exit 1; }

util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
echo ">> GPU $GPU util=${util}% mem=${used} MiB CUDA_HOME=$CUDA_HOME nvcc=$(command -v nvcc)"

if [ "${util:-100}" -gt 5 ]; then
  echo "FATAL: GPU $GPU utilization ${util}% > 5%. Refusing."
  exit 1
fi
export RESUME_ALLOW_BUSY_GPU=1

LOG="base_agent_oss120b_remaining_resume_Aug_16.log"
{
  echo
  echo "================================================================================"
  echo "[remaining] start utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$GPU util=${util}% mem=${used}MiB"
  echo "[remaining] CUDA_HOME=$CUDA_HOME"
  echo "[remaining] jobs: dmr 9,41 then markov 18"
  echo "================================================================================"
} >> "$LOG"

echo ">> remaining resumes on GPU $GPU: dmr then markov"

bash "$HERE/resume_oss120b_deletion_merge_refine_early_aborts.sh" "$GPU"
bash "$HERE/resume_oss120b_markov_early_aborts.sh" "$GPU"

{
  echo "[remaining] all jobs complete utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOG"

echo ">> remaining resumes finished  log: $LOG"
