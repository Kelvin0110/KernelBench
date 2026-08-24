#!/usr/bin/env bash
# Launch ONE OSS-120b inference arm on a shared A6000 (CPU6).
# Disables the 42GB GPUMemoryReserver so multiple arms can share one GPU.
# Eval GPU phase is serialised by KB_GPU_EVAL_LOCK (default on).
#
#   bash scripts_integration/new_evolving_agent/infer_api/launch_oss120b_shared_gpu.sh \
#     <gpu> <run_name> <ctx_mode> [extra flags...]
#
# Example:
#   bash .../launch_oss120b_shared_gpu.sh 1 base_agent_oss120b_folding_itr30 folding --no-skill-deletion
#
# Lock-trim env (KB_EVAL_HOIST_INPUT_GEN, KB_EVAL_SKIP_DEAD_REF_TIMING, phase log)
# is set only on this child via `env`, so live arms that already started without
# those flags keep their original eval path.

set -euo pipefail

GPU="${1:?usage: launch_oss120b_shared_gpu.sh <gpu> <run_name> <ctx_mode> [extra flags...]}"
RUN_NAME="${2:?missing run_name}"
CTX="${3:?missing context-management mode}"
shift 3
EXTRA=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [ -x /usr/local/cuda-12.6/bin/nvcc ]; then
  export CUDA_HOME="/usr/local/cuda-12.6"
elif [ -x "$HOME/opt/cuda-12.8/bin/nvcc" ]; then
  export CUDA_HOME="$HOME/opt/cuda-12.8"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME="/usr/local/cuda"
fi
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$REPO_ROOT/.venv/bin:$PATH"
# Do not inherit Cursor sandbox UV_CACHE_DIR under /tmp (permission denied).
export UV_CACHE_DIR="$HOME/.cache/uv"
export PYTHONUNBUFFERED=1
export KB_GPU_RESERVE_GB=0
export KB_GPU_EVAL_LOCK="${KB_GPU_EVAL_LOCK:-1}"
export NVIDIA_EMBED_ENDPOINT=inference
export NVIDIA_SKILL_MERGE_EMBED_MODEL=qwen3-embedding-0.6b

command -v nvcc >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=${CUDA_HOME:-unset})"; exit 1; }

# evolve_kb_batch.py stamps run dirs at minute resolution (YYYY_MM_DD_HH_MM).
# Concurrent jobs with the same --run-name in the same UTC minute share one
# directory — pass distinct names (e.g. ..._t2 vs ..._t3) when launching in parallel.
STAMP="$(date -u +%Y_%m_%d_%H_%M_%S)"

EXTRA_JOINED="${EXTRA[*]:-}"
if [[ "$RUN_NAME" == *l2* || "$EXTRA_JOINED" == *"--enable-l2"* ]]; then
  LOG_DIR="$REPO_ROOT/eval_log/skill_management/new_inference_endpoint_and_SONG_CPU6"
elif [[ "$CTX" == "compress_trigger" || "$RUN_NAME" == *compaction* ]]; then
  LOG_DIR="$REPO_ROOT/eval_log/context_management/new_inference_endpoint"
else
  LOG_DIR="$REPO_ROOT/eval_log/context_management/new_inference_endpoint"
fi
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/${RUN_NAME}_gpu${GPU}_${STAMP}.log"
PHASE_LOG="$LOG_DIR/${RUN_NAME}_gpu${GPU}_${STAMP}.phase.jsonl"
RESULTS_ROOT="runs_evolving/inference_oss_120b"

echo ">> GPU $GPU  $RUN_NAME  ctx=$CTX  reserve=${KB_GPU_RESERVE_GB}GB  embed=$NVIDIA_SKILL_MERGE_EMBED_MODEL"
echo ">> CUDA_HOME=$CUDA_HOME  nvcc=$(command -v nvcc)"
echo ">> extra: ${EXTRA[*]:-}"
echo ">> log: $LOG"
echo ">> phase: $PHASE_LOG"
echo ">> hoist=1 skip_dead_ref=1"

CUDA_VISIBLE_DEVICES="$GPU" nohup env \
  UV_CACHE_DIR="$UV_CACHE_DIR" \
  CUDA_HOME="$CUDA_HOME" \
  PATH="$PATH" \
  PYTHONUNBUFFERED=1 \
  KB_GPU_RESERVE_GB=0 \
  KB_GPU_EVAL_LOCK="$KB_GPU_EVAL_LOCK" \
  KB_EVAL_HOIST_INPUT_GEN=1 \
  KB_EVAL_SKIP_DEAD_REF_TIMING=1 \
  KB_EVAL_PHASE_LOG="$PHASE_LOG" \
  NVIDIA_EMBED_ENDPOINT=inference \
  NVIDIA_SKILL_MERGE_EMBED_MODEL=qwen3-embedding-0.6b \
  uv run --no-sync python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name "$RUN_NAME" \
  --results-root "$RESULTS_ROOT" \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware SONG_CPU6_A6000x4 \
  --context-management "$CTX" \
  "${EXTRA[@]}" \
  >> "$LOG" 2>&1 &

echo "   pid=$!"
