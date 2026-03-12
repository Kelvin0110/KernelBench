#!/bin/bash
# Container entry point for AIDE + KernelBench Docker integration.
# Modeled after aide/start.sh from the MLEBench integration.
#
# Required env vars: LEVEL, PROBLEM_ID
# Optional env vars: STEPS, HOURS, TIME_LIMIT_SECS, CODE_MODEL, FEEDBACK_MODEL,
#                    RUN_NAME, BACKEND, PRECISION, MOCK_EVAL, GPU_MEMORY_FRACTION
set -euo pipefail
set -x  # Echo commands for debugging

# Trap SIGTERM and SIGINT to gracefully shutdown background process
# This allows Python atexit/cleanup handlers to execute before terminating
trap 'echo "SIGTERM received, sending to background process"; kill -TERM $PYTHON_PID 2>/dev/null; wait $PYTHON_PID 2>/dev/null; exit 143' SIGTERM
trap 'echo "SIGINT received, sending to background process"; kill -TERM $PYTHON_PID 2>/dev/null; wait $PYTHON_PID 2>/dev/null; exit 130' SIGINT

# ---- Validate required env vars ----
: "${LEVEL:?ERROR: LEVEL env var is required}"
: "${PROBLEM_ID:?ERROR: PROBLEM_ID env var is required}"

# ---- Set defaults for optional vars ----
STEPS=${STEPS:-50}
HOURS=${HOURS:-2.0}
FINAL_EVAL_GRACE_SECS=${FINAL_EVAL_GRACE_SECS:-3600}  # 1 hour for final eval after AIDE stops

# Shell timeout: user can override; Python will validate and report adequacy
TIME_LIMIT_SECS=${TIME_LIMIT_SECS:-21600}  # Safe default: 6 hours (for up to 2h AIDE + 1h eval + buffer)

CODE_MODEL=${CODE_MODEL:-"openai/gpt-oss-120b"}
FEEDBACK_MODEL=${FEEDBACK_MODEL:-"openai/gpt-oss-120b"}
RUN_NAME=${RUN_NAME:-"docker_run"}
BACKEND=${BACKEND:-"cuda"}
PRECISION=${PRECISION:-"fp32"}
MOCK_EVAL=${MOCK_EVAL:-"0"}
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-"0.90"}

# ---- Hardware discovery (from MLEBench pattern) ----
if command -v nvidia-smi &> /dev/null && \
   nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
    HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[ \t]*//' | sed 's/[ \t]*$//' \
        | sort | uniq -c \
        | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
        | paste -sd ', ' -)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    echo "Detected GPU: ${GPU_NAME}"
else
    HARDWARE="CPU only (no GPU detected)"
    GPU_NAME=""
    echo "WARNING: No GPU detected"
fi
export HARDWARE GPU_NAME

# Verify PyTorch can see the GPU
python -c "
import torch
if torch.cuda.is_available():
    print(f'PyTorch CUDA OK: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: PyTorch cannot see CUDA GPU')
"

# ---- Prepare results directory ----
mkdir -p /app/run/logs
mkdir -p /app/run/workspaces

# ---- Convert time limit for display ----
format_time() {
    local secs=$1
    printf "%dh %dm %ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}
echo "================================================================"
echo "KernelBench + AIDE Docker Container"
echo "Problem: Level ${LEVEL}, ID ${PROBLEM_ID}"
echo "Time limit: $(format_time ${TIME_LIMIT_SECS})"
echo "Steps: ${STEPS}, Hours: ${HOURS}"
echo "Models: code=${CODE_MODEL}, feedback=${FEEDBACK_MODEL}"
echo "Backend: ${BACKEND}, Precision: ${PRECISION}"
echo "GPU memory fraction: ${GPU_MEMORY_FRACTION} (mock_eval=${MOCK_EVAL})"
echo "Final eval grace period: ${FINAL_EVAL_GRACE_SECS}s after AIDE time limit"
echo "================================================================"

# ---- Run with 3-tier timeout (adapted from Caesar pattern) ----
# Instead of a single `timeout` command that sends one signal, this runs the
# Python process in background and monitors it with escalating kill signals:
#   Tier 1: SIGTERM (Python sets stop_flag, finishes current node, runs final eval)
#   Tier 2: SIGKILL (hard kill after FINAL_EVAL_GRACE_SECS grace period)
cd /app
python scripts_integration/docker/docker_single_run.py \
    --level "${LEVEL}" \
    --problem_id "${PROBLEM_ID}" \
    --steps "${STEPS}" \
    --hours "${HOURS}" \
    --run_name "${RUN_NAME}" \
    --code_model "${CODE_MODEL}" \
    --feedback_model "${FEEDBACK_MODEL}" \
    --backend "${BACKEND}" \
    --precision "${PRECISION}" \
    --gpu-memory-fraction "${GPU_MEMORY_FRACTION}" \
    --results_dir "/app/run" &

PYTHON_PID=$!
START_TIME=$(date +%s)
DEADLINE=$((START_TIME + TIME_LIMIT_SECS))

# Monitor loop (check every 5s)
while kill -0 $PYTHON_PID 2>/dev/null; do
    NOW=$(date +%s)
    if [ $NOW -ge $DEADLINE ]; then
        echo "TIME LIMIT REACHED: sending stop signal to Python (AIDE search will finish, final eval will run)..."

        # Tier 1: SIGTERM — Python handler sets stop_flag, aborts current node, then
        # runs final evaluation. Do NOT kill yet — wait for graceful completion.
        kill -TERM $PYTHON_PID 2>/dev/null

        echo "Waiting up to ${FINAL_EVAL_GRACE_SECS}s for final evaluation to complete..."
        for i in $(seq 1 $FINAL_EVAL_GRACE_SECS); do
            kill -0 $PYTHON_PID 2>/dev/null || { echo "Python exited cleanly after ${i}s"; break; }
            sleep 1
        done

        # Tier 2: SIGKILL only if still alive after full grace period
        if kill -0 $PYTHON_PID 2>/dev/null; then
            echo "Final eval grace period expired (${FINAL_EVAL_GRACE_SECS}s). Escalating to SIGKILL..."
            kill -9 $PYTHON_PID 2>/dev/null
            sleep 2
        fi

        echo "Container exiting with timeout (code 124)"
        exit 124
    fi
    sleep 5
done

wait $PYTHON_PID
EXIT_CODE=$?
echo "Container exiting with code ${EXIT_CODE}"
exit ${EXIT_CODE}
