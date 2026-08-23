#!/usr/bin/env bash
# Launch replicates of one or more CONTEXT-MANAGEMENT arms on a single GPU,
# round-robin across modes and staggered in time.
#
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/launch_arm_reps.sh <gpu> <mode>:<reps> [<mode>:<reps> ...]
#
#   bash .../launch_arm_reps.sh 0 truncation:2 markov_report:2 folding:2
#   bash .../launch_arm_reps.sh 0 dry-run truncation:2 markov_report:2 folding:2
#   bash .../launch_arm_reps.sh 0 status
#
# Generalised from launch_merge_reps.sh; the preflight, run-dir resolution and
# collision handling there are reused verbatim because each was written against
# an observed failure. See that script's header for the full rationale on
# --no-sync, KB_GPU_RESERVE_GB and run-name collisions.
#
# ---------------------------------------------------------------------------
# GPU eval-lock contention -- the binding constraint above ~3 arms per GPU
# ---------------------------------------------------------------------------
# The lock serialises only the GPU phase of an eval (sub-second); the ~38s bulk
# is nvcc compilation and stays parallel. Measured on the 3 merge arms sharing
# GPU 1 (n=542 waits): median 30.9s, p90 265.6s, max 524.9s, zero timeouts.
#
# The default timeout is 1800s, and on expiry the lock does NOT fail -- it logs
# "proceeding UNLOCKED" and runs the eval under contention anyway, trading
# metric fidelity for liveness. A contended measurement is a silently deflated
# speedup (the gpu_lock docstring measures CV 26% -> 77%, worst sample 3.4x off),
# i.e. exactly the corruption the lock exists to prevent.
#
# Queue depth scales with contenders, so 6 arms puts the tail near 1800s where 3
# arms peaked at 525s. We therefore raise the timeout rather than let it expire:
# waiting longer costs only wall-clock, whereas expiring costs data validity.
# flock is released by the kernel on process exit, so a crashed arm cannot wedge
# the others; only a live-but-hung arm makes the longer timeout felt.
KB_GPU_EVAL_LOCK_TIMEOUT_SEC="${KB_GPU_EVAL_LOCK_TIMEOUT_SEC:-5400}"
export KB_GPU_EVAL_LOCK_TIMEOUT_SEC

set -euo pipefail

GPU="${1:?usage: launch_arm_reps.sh <gpu> <mode>:<reps> [<mode>:<reps> ...]}"
shift
MODE="launch"
case "${1:-}" in
  dry-run|status) MODE="$1"; shift ;;
esac
SPECS=("$@")

# ---- tunables -------------------------------------------------------------
LAG_SEC="${LAG_SEC:-180}"
CTX_DEFAULT_TAGS=1
MAX_ARMS_PER_GPU="${MAX_ARMS_PER_GPU:-6}"
DIR_WAIT_SEC="${DIR_WAIT_SEC:-300}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"

if [ "$LAG_SEC" -le 60 ]; then
  echo "FATAL: LAG_SEC=$LAG_SEC must be > 60 (run-name timestamp is minute-resolution)"; exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

# Hardware/baseline is a parameter so this script works on another server:
#   HARDWARE=<folder under results/timing> bash <this script> ...
# shellcheck source=./hardware_env.sh
# This server's baseline is results/timing/NVIDIA_GH200x2_median/ -- levels 1-2
# from the batch run plus level 3 re-measured in fresh processes (median-of-3).
# Without this, hardware_env.sh's folder-name auto-resolution would pick
# results/timing/NVIDIA_GH200x2/, which predates fix(eval) 6a3e972 and carries no
# median field on any of its 249 entries, so kb_require_hardware hard-FATALs.
KB_DEFAULT_HARDWARE="${KB_DEFAULT_HARDWARE:-NVIDIA_GH200x2_median}"
source "$(dirname "${BASH_SOURCE[0]}")/../hardware_env.sh"
kb_resolve_hardware


RESULTS_ROOT="runs_evolving/gpt-oss-120b/"
STAMP="$(date -u +%b_%-d)"
MANIFEST_GLOB="arm_reps_gpu${GPU}_*.manifest.tsv"
MANIFEST_DEFAULT="arm_reps_gpu${GPU}_${STAMP}.manifest.tsv"

# Naming per CLAUDE.md 3.2. truncation is the baseline and carries no tag, which
# matches the existing base_agent_gpt_oss_120b_itr30_GH200_* run.
mode_tag() {
  case "$1" in
    truncation)          echo "" ;;
    markov_report)       echo "markov" ;;
    folding)             echo "folding" ;;
    compress_trigger)    echo "compress" ;;
    selective_retention) echo "selective" ;;
    *) echo "FATAL: unknown context-management mode: $1" >&2; return 1 ;;
  esac
}
run_name_for() {
  local tag; tag="$(mode_tag "$1")" || return 1
  if [ -z "$tag" ]; then echo "base_agent_gpt_oss_120b_itr30_GH200"
  else echo "base_agent_gpt_oss_120b_${tag}_itr30_GH200"; fi
}

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
if [ "$MODE" = "status" ]; then
  MANIFEST="${MANIFEST:-$(ls -1t $MANIFEST_GLOB 2>/dev/null | head -1 || true)}"
  [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ] || {
    echo "no manifest found matching $MANIFEST_GLOB"
    ls -1t $MANIFEST_GLOB 2>/dev/null | sed 's/^/  /' || echo "  (none)"; exit 1; }
  echo "manifest: $MANIFEST"
  printf '%-4s %-20s %-9s %-9s %-8s %s\n' IDX MODE PID PROBLEM LOCKMAX RUNDIR
  while IFS=$'\t' read -r idx mode pid rundir log; do
    [ "$idx" = "idx" ] && continue
    alive="dead"; kill -0 "$pid" 2>/dev/null && alive="$pid"
    prob="$(grep -E "^\[batch\] \([0-9]+/" "$log" 2>/dev/null | tail -1 | grep -oE '\([0-9]+/[0-9]+\)' || echo '-')"
    # Worst lock wait so far -- the number that decides whether 6 arms is viable.
    lmax="$(grep -oE 'acquired after [0-9.]+s' "$log" 2>/dev/null | grep -oE '[0-9.]+' \
            | sort -g | tail -1 || true)"; lmax="${lmax:-0}"
    printf '%-4s %-20s %-9s %-9s %-8s %s\n' "$idx" "$mode" "$alive" "$prob" "${lmax}s" "$(basename "$rundir")"
  done < "$MANIFEST"
  echo
  unlocked=0
  while IFS=$'\t' read -r idx mode pid rundir log; do
    [ "$idx" = "idx" ] && continue
    n="$(grep -c "proceeding UNLOCKED" "$log" 2>/dev/null || true)"; unlocked=$((unlocked + ${n:-0}))
  done < "$MANIFEST"
  echo "contended (UNLOCKED) evals: $unlocked   <-- MUST stay 0; any >0 means deflated speedups"
  echo "lock timeout in force: ${KB_GPU_EVAL_LOCK_TIMEOUT_SEC}s"
  exit 0
fi

[ "${#SPECS[@]}" -gt 0 ] || { echo "FATAL: no <mode>:<reps> specs given"; exit 1; }

# ---- expand specs, round-robin across modes -------------------------------
# Round-robin rather than grouped: same-mode replicates end up 3*len(modes) apart
# instead of adjacent, and a partial failure leaves one of each mode rather than
# two of one.
declare -a Q_MODE=()
declare -a MODES=() COUNTS=()
for spec in "${SPECS[@]}"; do
  m="${spec%%:*}"; n="${spec##*:}"
  [ "$m" = "$spec" ] && n=1
  run_name_for "$m" >/dev/null || exit 1
  case "$n" in (*[!0-9]*|"") echo "FATAL: bad rep count in '$spec'"; exit 1 ;; esac
  MODES+=("$m"); COUNTS+=("$n")
done
maxn=0; for n in "${COUNTS[@]}"; do [ "$n" -gt "$maxn" ] && maxn="$n"; done
for ((r=0; r<maxn; r++)); do
  for ((j=0; j<${#MODES[@]}; j++)); do
    [ "$r" -lt "${COUNTS[$j]}" ] && Q_MODE+=("${MODES[$j]}")
  done
done
TOTAL="${#Q_MODE[@]}"

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
echo "=== preflight ==="
command -v nvcc  >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH -- add .venv/bin"; exit 1; }
kb_require_hardware "$REPO_ROOT"
grep -q "NVIDIA_INF_API_KEY" .env 2>/dev/null || { echo "FATAL: NVIDIA_INF_API_KEY not in .env"; exit 1; }
[ -f "subset_selection/selected_problems_50.csv" ] || { echo "FATAL: missing subset CSV"; exit 1; }
echo "  nvcc: $(nvcc --version | tail -1)"

nvidia-smi -i "$GPU" >/dev/null 2>&1 || { echo "FATAL: no GPU $GPU"; exit 1; }
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")"
echo "  GPU $GPU free: ${free_mib} MiB"
[ "$free_mib" -lt "$MIN_FREE_MIB" ] && { echo "FATAL: GPU $GPU has only ${free_mib} MiB free"; exit 1; }

# Count real arms only. A bare `pgrep -f evolve_kb_batch` also matches stale
# wait-loops whose own cmdline contains the pattern (two such orphans were found
# spinning for 13 and 9 days); filtering on CUDA_VISIBLE_DEVICES excludes them.
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

echo "  what a bare 'uv run' would do to the shared .venv:"
uv sync --dry-run 2>&1 | sed -r 's/\x1b\[[0-9;]*m//g' | grep -E '^\s*[-+~] ' | sed 's/^/    /' || echo "    (no venv changes pending)"

echo "  probing load_inline(cuda_sources=...) ..."
uv run --no-sync python - <<'PY'
import sys, torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(), n); return x;}
'''
m = load_inline(name="arm_reps_probe", cpp_sources="torch::Tensor f(torch::Tensor x);",
                cuda_sources=src, functions=["f"], verbose=False)
ok = m.f(torch.zeros(8, device="cuda")).sum().item() == 8.0
print("    CUDA extension build+run:", "OK" if ok else "WRONG RESULT")
sys.exit(0 if ok else 1)
PY
echo "=== preflight passed ==="
echo

echo "plan: $TOTAL arm(s) on GPU $GPU, round-robin, staggered ${LAG_SEC}s"
for ((j=0; j<${#MODES[@]}; j++)); do echo "      ${COUNTS[$j]} x ${MODES[$j]}  -> $(run_name_for "${MODES[$j]}")"; done
echo "      launch order: ${Q_MODE[*]}"
echo "      KB_GPU_RESERVE_GB=0, KB_GPU_EVAL_LOCK_TIMEOUT_SEC=$KB_GPU_EVAL_LOCK_TIMEOUT_SEC"
echo

if [ "$MODE" = "dry-run" ]; then echo "(dry-run: nothing launched)"; exit 0; fi

MANIFEST="${MANIFEST:-$MANIFEST_DEFAULT}"
printf 'idx\tmode\tpid\trundir\tlog\n' > "$MANIFEST"
declare -a SEEN_DIRS=()

for ((i=0; i<TOTAL; i++)); do
  m="${Q_MODE[$i]}"
  RUN_NAME="$(run_name_for "$m")"
  n=1; for ((k=0; k<i; k++)); do [ "${Q_MODE[$k]}" = "$m" ] && n=$((n+1)); done
  LOG="${RUN_NAME}_${STAMP}_rep${n}.log"
  : > "$LOG"

  # Same-run-name replicates must not share a UTC minute; see launch_merge_reps.sh.
  prev_same=""
  for ((k=i-1; k>=0; k--)); do [ "${Q_MODE[$k]}" = "$m" ] && { prev_same="${SEEN_DIRS[$k]}"; break; }; done
  if [ -n "$prev_same" ]; then
    pm="$(basename "$prev_same" | grep -oE '[0-9]{4}(_[0-9]{2}){4}$' || true)"
    while [ -n "$pm" ] && [ "$pm" = "$(date -u +%Y_%m_%d_%H_%M)" ]; do
      echo "   still inside UTC minute $pm; waiting for the boundary"; sleep 10
    done
  fi

  before="$(ls -1d ${RESULTS_ROOT}${RUN_NAME}_2* 2>/dev/null | sort || true)"
  echo ">> [$((i+1))/$TOTAL] $m rep$n on GPU $GPU -> $LOG"

  CUDA_VISIBLE_DEVICES="$GPU" KB_GPU_RESERVE_GB=0 \
  KB_GPU_EVAL_LOCK_TIMEOUT_SEC="$KB_GPU_EVAL_LOCK_TIMEOUT_SEC" \
  nohup uv run --no-sync python \
    scripts_integration/new_evolving_agent/evolve_kb_batch.py \
    --run-name "$RUN_NAME" \
    --results-root "$RESULTS_ROOT" \
    --max-problems 50 \
    --max-iterations 30 \
    --hardware "$HARDWARE" \
    --nvidia-endpoint inference \
    --model gpt-oss-120b \
    --context-management "$m" \
    --coder-timeout-sec 600 \
    >> "$LOG" 2>&1 &
  pid=$!
  echo "   pid=$pid"

  rundir=""
  for _ in $(seq 1 "$DIR_WAIT_SEC"); do
    sleep 1
    after="$(ls -1d ${RESULTS_ROOT}${RUN_NAME}_2* 2>/dev/null | sort || true)"
    new="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | head -1)"
    [ -n "$new" ] && { rundir="$new"; break; }
    kill -0 "$pid" 2>/dev/null || { echo "   FATAL: exited before creating a run dir:"; tail -20 "$LOG" | sed 's/^/     /'; exit 1; }
  done
  if [ -z "$rundir" ]; then
    echo "   FATAL: no new run dir in ${DIR_WAIT_SEC}s (slow start, or a run-name"
    echo "          collision that adopted an earlier replicate's directory). Killing."
    pkill -P "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true
    sleep 2; pkill -9 -P "$pid" 2>/dev/null || true; kill -9 "$pid" 2>/dev/null || true
    tail -20 "$LOG" | sed 's/^/     /'; exit 1
  fi
  SEEN_DIRS+=("$rundir")
  printf '%s\t%s\t%s\t%s\t%s\n' "$((i+1))" "$m" "$pid" "$rundir" "$LOG" >> "$MANIFEST"
  echo "   run dir: $rundir"

  [ $((i+1)) -lt "$TOTAL" ] && { echo "   waiting ${LAG_SEC}s"; sleep "$LAG_SEC"; }
done

echo
echo "=== launched $TOTAL arm(s) on GPU $GPU ==="
column -t -s$'\t' "$MANIFEST"
cat <<EOF

manifest: $MANIFEST
Status:   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/launch_arm_reps.sh $GPU status

WATCH THE LOCK. At 3 arms the worst wait was 525s against an 1800s timeout;
this run has $TOTAL arms on one GPU with the timeout raised to ${KB_GPU_EVAL_LOCK_TIMEOUT_SEC}s.
"contended (UNLOCKED) evals" in status must stay 0 -- any non-zero means some
speedups were measured under contention and are deflated.
EOF
