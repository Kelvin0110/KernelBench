#!/usr/bin/env bash
# Wave A: first half of remaining OSS-120b early-abort resumes (9 problems).
# Runs sequentially on one GPU. Skip folding (already resumed) and
# deletion_merge_refine (parent batch still live).
#
#   bash scripts_integration/new_evolving_agent/infer_api/resume_oss120b_early_aborts_wave_a.sh [GPU]
#
# Default GPU=1.
# Order: baseline (1) -> skill_refine (2) -> deletion (3) -> markov (3)

set -euo pipefail

GPU="${1:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

LOG="base_agent_oss120b_resume_wave_a_Aug_13.log"

{
  echo
  echo "================================================================================"
  echo "[wave-a] start utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$GPU"
  echo "[wave-a] jobs: baseline, skill_refinement, deletion, markov (9 problems)"
  echo "================================================================================"
} >> "$LOG"

echo ">> WAVE A  GPU $GPU  baseline + skill_refine + deletion + markov"

bash "$HERE/resume_oss120b_baseline_early_aborts.sh" "$GPU"
bash "$HERE/resume_oss120b_skill_refinement_early_aborts.sh" "$GPU"
bash "$HERE/resume_oss120b_deletion_early_aborts.sh" "$GPU"
bash "$HERE/resume_oss120b_markov_early_aborts.sh" "$GPU"

{
  echo "[wave-a] all jobs complete utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOG"

echo ">> WAVE A finished  log: $LOG"
