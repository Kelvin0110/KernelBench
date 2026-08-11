#!/usr/bin/env bash
# Sequentially resume EARLY_TIMEOUT problem ranges for the terra truncation run.
# All waves append to one log file. Earlier indices run first (L1 causal resume).
#
#   bash scripts_integration/new_evolving_agent/infer_api/resume_terra_truncation_early_aborts.sh [GPU]
#
# Default GPU=1 (GPU0 often holds compress_trigger).

set -euo pipefail

GPU="${1:-1}"
RUN_NAME="base_agent_gpt_56_terra_truncation_itr30_2026_08_01_17_40"
RESULTS_ROOT="runs_evolving/inference_gpt_56_terra"
LOG="base_agent_gpt_56_terra_truncation_itr30_resume_early_Aug_11.log"

# start:end pairs (early aborts only; skip COMPLETE_MID_API / CLEAN)
RANGES=(
  "2:2"
  "9:10"
  "14:16"
  "19:22"
  "24:24"
  "27:28"
  "30:31"
  "34:42"
  "44:44"
  "46:47"
  "49:49"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
[ -d "$RUN_DIR" ] || { echo "FATAL: no such run dir: $RUN_DIR"; exit 1; }
[ -f "$RUN_DIR/run_summary.json" ] || { echo "FATAL: missing run_summary.json in $RUN_DIR"; exit 1; }

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")"
if [ "$used" -gt 1000 ]; then
  echo "FATAL: GPU $GPU busy (${used} MiB). Refusing."
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
    --model gpt-5.6-terra \
    --no-skill-deletion \
    --skill-merge-similarity 0.7 \
    --hardware SONG_CPU4_A6000x2 \
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

echo ">> all truncation early-abort waves finished"
echo "   log: $LOG"
