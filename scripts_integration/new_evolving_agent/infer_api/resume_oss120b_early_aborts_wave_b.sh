#!/usr/bin/env bash
# Wave B: second half of remaining OSS-120b early-abort resumes (10 problems).
# Runs sequentially on one GPU. Skip folding (already resumed) and
# deletion_merge_refine (parent batch still live).
#
#   bash scripts_integration/new_evolving_agent/infer_api/resume_oss120b_early_aborts_wave_b.sh [GPU]
#
# Default GPU=2.
# Order: merge_only (4) -> selective_recent5 (6)

set -euo pipefail

GPU="${1:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

LOG="base_agent_oss120b_resume_wave_b_Aug_13.log"

{
  echo
  echo "================================================================================"
  echo "[wave-b] start utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$GPU"
  echo "[wave-b] jobs: merge_only, selective_recent5 (10 problems)"
  echo "================================================================================"
} >> "$LOG"

echo ">> WAVE B  GPU $GPU  merge_only + selective_recent5"

bash "$HERE/resume_oss120b_merge_only_early_aborts.sh" "$GPU"
bash "$HERE/resume_oss120b_selective_recent5_early_aborts.sh" "$GPU"

{
  echo "[wave-b] all jobs complete utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOG"

echo ">> WAVE B finished  log: $LOG"
