#!/usr/bin/env bash
# Sequentially resume early-abort / mid-timeout problems for the OSS-120b
# markov_report run. All waves append to one log file. Earlier indices first.
#
#   bash scripts_integration/new_evolving_agent/infer_api/resume_oss120b_markov_early_aborts.sh [GPU]
#
# Default GPU=0 (GPU1 often holds live folding / other resumes).
# Remaining abort after Wave A: 18=L2P42 (504/500 abort @ itr 2).
# Subsets 39 and 48 already completed on the Aug-13/14 resume.

set -euo pipefail

GPU="${1:-0}"
RUN_NAME="base_agent_oss120b_markov_itr30_2026_08_07_14_07"
RESULTS_ROOT="runs_evolving/inference_oss_120b"
LOG="base_agent_oss120b_markov_itr30_resume_early_Aug_11.log"

# start:end pairs (1-based subset indices)
RANGES=(
  "18:18"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
[ -d "$RUN_DIR" ] || { echo "FATAL: no such run dir: $RUN_DIR"; exit 1; }
[ -f "$RUN_DIR/run_summary.json" ] || { echo "FATAL: missing run_summary.json in $RUN_DIR"; exit 1; }

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")"
if [ "${RESUME_ALLOW_BUSY_GPU:-0}" != "1" ] && [ "$used" -gt 1000 ]; then
  echo "FATAL: GPU $GPU busy (${used} MiB). Refusing. Set RESUME_ALLOW_BUSY_GPU=1 to override."
  exit 1
fi

echo ">> GPU $GPU  sequential early-abort resume for $RUN_NAME"
echo ">> log: $LOG (append)"
echo ">> waves: ${#RANGES[@]}"

{
  echo
  echo "================================================================================"
  echo "[resume-script] start utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=$GPU run=$RUN_NAME"
  echo "[resume-script] waves=${RANGES[*]}"
  echo "================================================================================"
} >> "$LOG"

wave=0
for pair in "${RANGES[@]}"; do
  wave=$((wave + 1))
  START="${pair%%:*}"
  END="${pair##*:}"

  {
    echo
    echo "--------------------------------------------------------------------------------"
    echo "[resume-script] wave $wave/${#RANGES[@]} problems ${START}..${END} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--------------------------------------------------------------------------------"
  } >> "$LOG"

  echo ">> wave $wave/${#RANGES[@]}: problems ${START}..${END}"

  CUDA_VISIBLE_DEVICES="$GPU" uv run python \
    scripts_integration/new_evolving_agent/evolve_kb_batch.py \
    --resume \
    --results-root "$RESULTS_ROOT" \
    --run-name "$RUN_NAME" \
    --max-iterations 30 \
    --start-problem "$START" \
    --end-problem "$END" \
    --nvidia-endpoint inference \
    --model gpt-oss-120b \
    --context-management markov_report \
    --evolving-report-max-tokens 65536 \
    --no-skill-deletion \
    --skill-merge-similarity 0.7 \
    --skill-merge-interval 50 \
    --hardware SONG_CPU6_A6000x4 \
    >> "$LOG" 2>&1

  {
    echo "[resume-script] wave $wave/${#RANGES[@]} done utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "$LOG"
  echo "   wave $wave done"
done

{
  echo
  echo "================================================================================"
  echo "[resume-script] all waves complete utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "================================================================================"
} >> "$LOG"

echo ">> all markov early-abort waves finished"
echo "   log: $LOG"
