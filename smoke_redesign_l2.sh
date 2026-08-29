#!/usr/bin/env bash
# Smoke the --redesign-l2 merge: shipped gate vs redesign, 3 problems x 3 iters.
#
# Both arms on gpt-oss-120b per the project's smoke rule -- the point is to catch
# a silently no-op'd feature, and a second model would confound "my flag is broken"
# with "this model behaves differently".
#
# Floors are dropped to near-zero for BOTH arms. At shipped floors (tasks 3,
# selections 50) nothing can possibly promote inside 3 problems, so the promotion
# path would never execute and the smoke run would pass while proving nothing.
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_HOME="$HOME/opt/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PWD/.venv/bin:$PATH"
export KB_GPU_RESERVE_GB=0
export KB_EVAL_HOIST_INPUT_GEN=1
export KB_EVAL_SKIP_DEAD_REF_TIMING=1
export KB_EVAL_UNLOCK_CORRECTNESS=0
export KB_GPU_EVAL_LOCK_SLOTS=1

COMMON=(--max-problems 3 --max-iterations 3 --model gpt-oss-120b
        --nvidia-endpoint inference --hardware NVIDIA_GH200x2_median
        --coder-timeout-sec 600 --results-root runs_evolving/smoke_redesign/
        --enable-l2 --l2-min-tasks 1 --l2-min-selections 1 --l2-min-rate 0.01
        --l2-min-hit-rate 0.01)

run () {  # run <tag> <gpu> [extra flags...]
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup .venv/bin/python \
    scripts_integration/new_evolving_agent/evolve_kb_batch.py \
    --run-name "smoke_${tag}" "${COMMON[@]}" "$@" \
    > "smoke_${tag}.log" 2>&1 &
  echo "  ${tag}: pid $! -> smoke_${tag}.log"
}

echo "launching:"
run shipped  0
run redesign 1 --redesign-l2
echo "done. tail -f smoke_*.log"
