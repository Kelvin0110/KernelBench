#!/usr/bin/env bash
# Launch a WAVE of heterogeneous arms (different context modes AND different
# skill-governance flags) on one GPU, staggered in time.
#
#   bash scripts_integration/new_evolving_agent/env/launch_wave.sh <gpu> <spec_file> [dry-run|status]
#
# launch_arm_reps.sh only takes <context-mode>:<reps>, so it cannot express a
# governance arm (--skill-deletion, --skill-merging, --enable-l2, ...). This
# takes a spec file instead, one arm per line:
#
#   # tag  |  context-mode  |  extra flags
#   -            | truncation |
#   merge_sim08  | truncation | --skill-merging --skill-merge-similarity 0.8
#   deletion     | truncation | --skill-deletion
#   folding      | folding    |
#
# A tag of "-" means the untagged baseline run name (CLAUDE.md 3.2).
#
# BASELINE: set HARDWARE=<folder under results/timing> to run on another server;
# resolution and validation live in hardware_env.sh, shared with the other
# launchers. Default is NVIDIA_GH200x2_2nd, the median-bearing baseline from
# e80838b -- the older NVIDIA_GH200x2/ predates fix(eval) 6a3e972 and has no
# median, so a run pointed at it divides candidate median by baseline mean. On
# this 50-problem subset that shifts 25 of 50 problems >5% (26_GELU_/22_Tanh ~4x).

KB_GPU_EVAL_LOCK_TIMEOUT_SEC="${KB_GPU_EVAL_LOCK_TIMEOUT_SEC:-5400}"
export KB_GPU_EVAL_LOCK_TIMEOUT_SEC

set -euo pipefail

GPU="${1:?usage: launch_wave.sh <gpu> <spec_file> [dry-run|status]}"
SPEC_FILE="${2:?missing spec file}"
MODE="${3:-launch}"

LAG_SEC="${LAG_SEC:-180}"
MAX_ARMS_PER_GPU="${MAX_ARMS_PER_GPU:-6}"
DIR_WAIT_SEC="${DIR_WAIT_SEC:-300}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
# shellcheck source=./hardware_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/hardware_env.sh"
kb_resolve_hardware
MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"

if [ "$LAG_SEC" -le 60 ]; then
  echo "FATAL: LAG_SEC=$LAG_SEC must be > 60 (run-name timestamp is minute-resolution)"; exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

RESULTS_ROOT="runs_evolving/gpt-oss-120b/"
STAMP="$(date -u +%b_%-d)"
MANIFEST_GLOB="wave_gpu${GPU}_*.manifest.tsv"
MANIFEST_DEFAULT="wave_gpu${GPU}_${STAMP}.manifest.tsv"

# ---------------------------------------------------------------------------
# status -- same audit as launch_arm_reps.sh, plus the orphaned-wait count that
# CLAUDE.md 3.4 requires (on pre-fix runs "proceeding UNLOCKED" cannot fire).
# ---------------------------------------------------------------------------
if [ "$MODE" = "status" ]; then
  MANIFEST="${MANIFEST:-$(ls -1t $MANIFEST_GLOB 2>/dev/null | head -1 || true)}"
  [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ] || { echo "no manifest matching $MANIFEST_GLOB"; exit 1; }
  echo "manifest: $MANIFEST"
  printf '%-4s %-26s %-9s %-9s %-8s %-8s %s\n' IDX TAG PID PROBLEM LOCKMAX ORPHWAIT RUNDIR
  while IFS=$'\t' read -r idx tag pid rundir log; do
    [ "$idx" = "idx" ] && continue
    alive="dead"; kill -0 "$pid" 2>/dev/null && alive="$pid"
    prob="$(grep -E "^\[batch\] \([0-9]+/" "$log" 2>/dev/null | tail -1 | grep -oE '\([0-9]+/[0-9]+\)' || echo '-')"
    lmax="$(grep -oE 'acquired after [0-9.]+s' "$log" 2>/dev/null | grep -oE '[0-9.]+' | sort -g | tail -1 || true)"
    w="$(grep -c "gpu-eval-lock.*waiting" "$log" 2>/dev/null || echo 0)"
    a="$(grep -c "gpu-eval-lock.*acquired" "$log" 2>/dev/null || echo 0)"
    printf '%-4s %-26s %-9s %-9s %-8s %-8s %s\n' "$idx" "$tag" "$alive" "$prob" "${lmax:-0}s" "$((w - a))" "$(basename "$rundir")"
  done < "$MANIFEST"
  echo
  unlocked=0
  while IFS=$'\t' read -r idx tag pid rundir log; do
    [ "$idx" = "idx" ] && continue
    n="$(grep -c "proceeding UNLOCKED" "$log" 2>/dev/null || true)"; unlocked=$((unlocked + ${n:-0}))
  done < "$MANIFEST"
  echo "contended (UNLOCKED) evals: $unlocked   <-- MUST stay 0"
  echo "ORPHWAIT (waiting with no acquire) should also be 0; >0 means evals died mid-wait"
  exit 0
fi

# ---- parse spec file ------------------------------------------------------
[ -f "$SPEC_FILE" ] || { echo "FATAL: no such spec file: $SPEC_FILE"; exit 1; }
declare -a Q_TAG=() Q_CTX=() Q_FLAGS=()
while IFS= read -r line; do
  line="${line%%#*}"
  [ -z "${line// /}" ] && continue
  tag="$(echo "${line%%|*}" | xargs)"
  rest="${line#*|}"
  ctx="$(echo "${rest%%|*}" | xargs)"
  flags=""
  case "$rest" in *\|*) flags="$(echo "${rest#*|}" | xargs)" ;; esac
  case "$ctx" in
    truncation|folding|markov_report|selective_retention|compress_trigger) ;;
    *) echo "FATAL: unknown context-management mode '$ctx' in spec line: $line"; exit 1 ;;
  esac
  Q_TAG+=("$tag"); Q_CTX+=("$ctx"); Q_FLAGS+=("$flags")
done < "$SPEC_FILE"
TOTAL="${#Q_TAG[@]}"
[ "$TOTAL" -gt 0 ] || { echo "FATAL: spec file has no arms"; exit 1; }

run_name_for_tag() {
  if [ "$1" = "-" ] || [ -z "$1" ]; then echo "base_agent_gpt_oss_120b_itr30_GH200"
  else echo "base_agent_gpt_oss_120b_${1}_itr30_GH200"; fi
}

# ---- preflight ------------------------------------------------------------
echo "=== preflight ==="
command -v nvcc  >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH -- add .venv/bin"; exit 1; }
grep -q "NVIDIA_INF_API_KEY" .env 2>/dev/null || { echo "FATAL: NVIDIA_INF_API_KEY not in .env"; exit 1; }
[ -f "subset_selection/selected_problems_50.csv" ] || { echo "FATAL: missing subset CSV"; exit 1; }
echo "  nvcc: $(nvcc --version | tail -1)"

kb_require_hardware "$REPO_ROOT"

nvidia-smi -i "$GPU" >/dev/null 2>&1 || { echo "FATAL: no GPU $GPU"; exit 1; }
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")"
echo "  GPU $GPU free: ${free_mib} MiB"
[ "$free_mib" -lt "$MIN_FREE_MIB" ] && { echo "FATAL: GPU $GPU has only ${free_mib} MiB free"; exit 1; }

existing=0
for p in $(pgrep -f "evolve_kb_batch" 2>/dev/null || true); do
  vis="$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' || true)"
  [ "$vis" = "$GPU" ] && existing=$((existing + 1))
done
existing=$((existing / 2))
echo "  arms already on GPU $GPU: $existing"
if [ $((existing + TOTAL)) -gt "$MAX_ARMS_PER_GPU" ]; then
  echo "FATAL: $existing existing + $TOTAL new > MAX_ARMS_PER_GPU=$MAX_ARMS_PER_GPU."
  echo "       Raise MAX_ARMS_PER_GPU to override."; exit 1
fi

# sklearn is imported lazily by the merge pass, which swallows its own
# exceptions -- a missing sklearn yields zero merges and a silent no-op arm.
if printf '%s\n' "${Q_FLAGS[@]}" | grep -q -- "--skill-merging"; then
  uv run --no-sync python -c "import sklearn; print('  sklearn', sklearn.__version__, '(merge arms OK)')" \
    || { echo "FATAL: --skill-merging requested but sklearn missing"; exit 1; }
fi

echo "  what a bare 'uv run' would do to the shared .venv:"
uv sync --dry-run 2>&1 | sed -r 's/\x1b\[[0-9;]*m//g' | grep -cE '^\s*[-+~] ' \
  | sed 's/^/    packages a bare `uv run` would change: /' || echo "    (none)"

echo "  probing load_inline(cuda_sources=...) ..."
uv run --no-sync python - <<'PY'
import sys, torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(), n); return x;}
'''
m = load_inline(name="wave_probe", cpp_sources="torch::Tensor f(torch::Tensor x);",
                cuda_sources=src, functions=["f"], verbose=False)
ok = m.f(torch.zeros(8, device="cuda")).sum().item() == 8.0
print("    CUDA extension build+run:", "OK" if ok else "WRONG RESULT")
sys.exit(0 if ok else 1)
PY
echo "=== preflight passed ==="
echo

echo "plan: $TOTAL arm(s) on GPU $GPU, staggered ${LAG_SEC}s, hardware=$HARDWARE"
for ((i=0; i<TOTAL; i++)); do
  printf '      %-26s ctx=%-20s %s\n' "$(run_name_for_tag "${Q_TAG[$i]}")" "${Q_CTX[$i]}" "${Q_FLAGS[$i]}"
done
echo "      ${MAX_PROBLEMS} problems x ${MAX_ITERATIONS} iterations"
echo "      KB_GPU_RESERVE_GB=0, KB_GPU_EVAL_LOCK_TIMEOUT_SEC=$KB_GPU_EVAL_LOCK_TIMEOUT_SEC"
echo

if [ "$MODE" = "dry-run" ]; then echo "(dry-run: nothing launched)"; exit 0; fi

MANIFEST="${MANIFEST:-$MANIFEST_DEFAULT}"
printf 'idx\ttag\tpid\trundir\tlog\n' > "$MANIFEST"

for ((i=0; i<TOTAL; i++)); do
  tag="${Q_TAG[$i]}"; ctx="${Q_CTX[$i]}"; flags="${Q_FLAGS[$i]}"
  RUN_NAME="$(run_name_for_tag "$tag")"
  LOG="${RUN_NAME}_${STAMP}_wave.log"
  : > "$LOG"

  before="$(ls -1d ${RESULTS_ROOT}${RUN_NAME}_2* 2>/dev/null | sort || true)"
  echo ">> [$((i+1))/$TOTAL] $RUN_NAME (ctx=$ctx) $flags"

  # shellcheck disable=SC2086 -- flags must word-split into separate argv entries
  CUDA_VISIBLE_DEVICES="$GPU" KB_GPU_RESERVE_GB=0 \
  KB_GPU_EVAL_LOCK_TIMEOUT_SEC="$KB_GPU_EVAL_LOCK_TIMEOUT_SEC" \
  setsid nohup uv run --no-sync python \
    scripts_integration/new_evolving_agent/evolve_kb_batch.py \
    --run-name "$RUN_NAME" \
    --results-root "$RESULTS_ROOT" \
    --max-problems "$MAX_PROBLEMS" \
    --max-iterations "$MAX_ITERATIONS" \
    --hardware "$HARDWARE" \
    --nvidia-endpoint inference \
    --model gpt-oss-120b \
    --context-management "$ctx" \
    --coder-timeout-sec 600 \
    $flags \
    >> "$LOG" 2>&1 &
  pid=$!
  echo "   pid=$pid  log=$LOG"

  rundir=""
  for _ in $(seq 1 "$DIR_WAIT_SEC"); do
    sleep 1
    after="$(ls -1d ${RESULTS_ROOT}${RUN_NAME}_2* 2>/dev/null | sort || true)"
    new="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | head -1)"
    [ -n "$new" ] && { rundir="$new"; break; }
  done
  [ -n "$rundir" ] || { echo "   WARNING: no run dir appeared within ${DIR_WAIT_SEC}s -- check $LOG"; rundir="?"; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$((i+1))" "${tag:--}" "$pid" "$rundir" "$LOG" >> "$MANIFEST"

  if [ "$i" -lt $((TOTAL - 1)) ]; then
    echo "   staggering ${LAG_SEC}s before the next arm"
    sleep "$LAG_SEC"
  fi
done

echo
echo "manifest: $MANIFEST"
echo "watch:    bash ${BASH_SOURCE[0]} $GPU $SPEC_FILE status"
