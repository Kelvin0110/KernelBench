#!/usr/bin/env bash
# Launch a WAVE of heterogeneous arms (different context modes AND different
# skill-governance flags) on one GPU, staggered in time.
#
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/launch_wave.sh <gpu> <spec_file> [dry-run|status]
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

# Critical-section trims. Both default OFF in src/kernelbench/eval.py so that a
# run already in flight is never perturbed -- eval.py is re-imported from disk by
# every spawned eval child, so an unconditional change would reach live arms.
# They are turned on HERE, i.e. only for newly launched waves.
#   KB_EVAL_SKIP_DEAD_REF_TIMING -- drop the reference timing window when its
#     result is provably unused: a fixed baseline is supplied (the governor
#     overwrites the measured ref_runtime with it anyway), or the kernel is not
#     correct (runtime stays -1.0, so the excessive-speedup flag cannot fire).
#   KB_EVAL_HOIST_INPUT_GEN -- build get_inputs() tensors on the CPU before
#     taking the lock. Pure-CPU torch.rand; the host-to-device transfer stays
#     locked. Measured at 19.2s/eval of lock hold on level 1 problem 34.
# Neither changes a recorded number. Both shorten the hold, so a wave launched
# with them sees less contention deflation than one launched without -- compare
# speedups across that boundary with the same care as across a baseline change.
KB_EVAL_SKIP_DEAD_REF_TIMING="${KB_EVAL_SKIP_DEAD_REF_TIMING:-1}"
KB_EVAL_HOIST_INPUT_GEN="${KB_EVAL_HOIST_INPUT_GEN:-1}"
#   KB_EVAL_UNLOCK_CORRECTNESS -- run the correctness trials outside the lock,
#     leaving only the timing window(s) held. A/B on L1P100: hold 1.96s -> 0.78s,
#     measured runtime unchanged.
#   KB_GPU_EVAL_LOCK_SLOTS -- counting semaphore: how many evals may hold the GPU
#     at once. 1 == the historical mutex. The 2026-08-23 probe measured degree 2/3
#     at 1.2%/0.7% median runtime inflation (2.3%/2.8% worst), against ~20-30%
#     replicate noise. NEVER mix slots=1 and slots>1 on one GPU -- different files.
#   CORRECTED 2026-08-27: UNLOCK_CORRECTNESS now defaults to 0, not 1. The A/B
#     above measured RUNTIME, not MEMORY. Unlocking correctness removes the bound
#     on how many evals are DEVICE-resident at once (correctness does the H2D of
#     the whole input set plus two model forwards), and each eval retains ~30 GB on
#     a level-1 problem / ~52 GB on L1P34. On 2026-08-23 that took a 146.8 GB card
#     to 144.8 GB and 1.8% of evals to CUDA OOM -- each recorded compiled=True
#     correct=False, so the governor debugs a kernel that was never broken.
#     Re-locking cut peak 141.4 -> 72.6 GB and OOM to 0 for ~5.7 s of extra hold,
#     with the lock still showing zero waits over 5 s. Do not flip this back.
KB_EVAL_UNLOCK_CORRECTNESS="${KB_EVAL_UNLOCK_CORRECTNESS:-0}"
KB_GPU_EVAL_LOCK_SLOTS="${KB_GPU_EVAL_LOCK_SLOTS:-3}"
#   KB_EVAL_MEM_GATE_FACTOR -- byte-sized admission gate on top of the slots:
#     reserves factor x input_bytes, so effective concurrency is
#     min(slots, budget/need). With correctness LOCKED the residents are bounded by
#     SLOTS, but SLOTS does not bound how much they NEED: 3 x ~52 GB on L1P34
#     overruns a 143.4 GB card. On this card (budget 0.85 x 143.4 = 122 GB, largest
#     input 7.0 GB) the admit-2 band is 5.80 < factor <= 8.71; measured retention is
#     ~7x input, so 7 is both the physical value and inside the band. 9 would admit
#     only 1 and needlessly serialise the biggest problems; 2.5 provably never binds.
#     VERIFY IT FIRES: mem_gate_waited_sec must be non-zero for some eval inside
#     subset problems 1-5. An all-zero column is how factor=2.5 went unnoticed.
KB_EVAL_MEM_GATE_FACTOR="${KB_EVAL_MEM_GATE_FACTOR:-7}"
export KB_EVAL_SKIP_DEAD_REF_TIMING KB_EVAL_HOIST_INPUT_GEN KB_EVAL_UNLOCK_CORRECTNESS
export KB_GPU_EVAL_LOCK_SLOTS KB_EVAL_MEM_GATE_FACTOR

set -euo pipefail

# grep -c prints "0" AND exits 1 when there are no matches, so `|| echo 0`
# emits two lines and breaks the arithmetic below. Count through this instead.
cnt() { local n; n="$(grep -c "$1" "$2" 2>/dev/null | head -1)"; echo "${n:-0}"; }


GPU="${1:?usage: launch_wave.sh <gpu> <spec_file> [dry-run|status]}"
SPEC_FILE="${2:?missing spec file}"
MODE="${3:-launch}"

LAG_SEC="${LAG_SEC:-180}"
MAX_ARMS_PER_GPU="${MAX_ARMS_PER_GPU:-6}"
DIR_WAIT_SEC="${DIR_WAIT_SEC:-300}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
# shellcheck source=./hardware_env.sh
# This server's baseline is results/timing/NVIDIA_GH200x2_median/ -- levels 1-2
# from the batch run plus level 3 re-measured in fresh processes (median-of-3).
# Without this, hardware_env.sh's folder-name auto-resolution would pick
# results/timing/NVIDIA_GH200x2/, which predates fix(eval) 6a3e972 and carries no
# median field on any of its 249 entries, so kb_require_hardware hard-FATALs.
KB_DEFAULT_HARDWARE="${KB_DEFAULT_HARDWARE:-NVIDIA_GH200x2_median}"
source "$(dirname "${BASH_SOURCE[0]}")/../hardware_env.sh"
kb_resolve_hardware
MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"

if [ "$LAG_SEC" -le 60 ]; then
  echo "FATAL: LAG_SEC=$LAG_SEC must be > 60 (run-name timestamp is minute-resolution)"; exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

# MODEL/RESULTS_ROOT/RUN_PREFIX are parameters so one spec file can be run
# against a second model on another GPU. Defaults reproduce the original
# hardcoded gpt-oss-120b behaviour exactly.
MODEL="${MODEL:-gpt-oss-120b}"
RESULTS_ROOT="${RESULTS_ROOT:-runs_evolving/gpt-oss-120b/}"
RUN_PREFIX="${RUN_PREFIX:-base_agent_gpt_oss_120b}"
case "$RESULTS_ROOT" in */) ;; *) RESULTS_ROOT="$RESULTS_ROOT/" ;; esac
mkdir -p "$RESULTS_ROOT"
STAMP="$(date -u +%b_%-d)"
MANIFEST_GLOB="wave_gpu${GPU}_*.manifest.tsv"
MANIFEST_DEFAULT="wave_gpu${GPU}_${RUN_PREFIX}_${STAMP}.manifest.tsv"

# ---------------------------------------------------------------------------
# status -- same audit as launch_arm_reps.sh, plus the orphaned-wait count that
# CLAUDE.md 3.4 requires (on pre-fix runs "proceeding UNLOCKED" cannot fire).
# ---------------------------------------------------------------------------
if [ "$MODE" = "status" ]; then
  MANIFEST="${MANIFEST:-$(ls -1t $MANIFEST_GLOB 2>/dev/null | head -1 || true)}"
  [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ] || { echo "no manifest matching $MANIFEST_GLOB"; exit 1; }
  echo "manifest: $MANIFEST"
  printf '%-4s %-26s %-9s %-9s %-8s %-8s %s\n' IDX TAG PID PROBLEM LOCKMAX ORPHWAIT RUNDIR
  unlocked=0
  while IFS=$'\t' read -r idx tag pid rundir log; do
    [ "$idx" = "idx" ] && continue
    alive="dead"; kill -0 "$pid" 2>/dev/null && alive="$pid"
    prob="$(grep -E "^\[batch\] \([0-9]+/" "$log" 2>/dev/null | tail -1 | grep -oE '\([0-9]+/[0-9]+\)' || echo '-')"
    # Lock messages are printed by the eval CHILD, whose stdout eval_runner
    # captures via redirect_stdout into terminal_output -- they NEVER reach the
    # arm log. Grepping "$log" (as this block used to) made LOCKMAX/ORPHWAIT and
    # the "MUST stay 0" line below vacuously clean regardless of reality.
    read -r lmax w u <<<"$(LOCK_RUNDIR="$rundir" ./.venv/bin/python - <<'PYEOF'
import json, glob, os, re
d = os.environ.get("LOCK_RUNDIR", "")
orph = unl = 0; mx = 0.0
for f in glob.glob(os.path.join(d, "workspaces", "*", "evaluation_terminal_output.jsonl")) if d else []:
    try: fh = open(f)
    except OSError: continue
    with fh:
        for line in fh:
            if not line.strip(): continue
            try: t = str(json.loads(line).get("terminal_output", ""))
            except Exception: continue
            ha = "acquired after" in t; hu = "proceeding UNLOCKED" in t
            if "waiting for another eval" in t and not (ha or hu): orph += 1
            unl += t.count("proceeding UNLOCKED")
            for m in re.finditer(r"acquired after ([0-9.]+)s", t):
                mx = max(mx, float(m.group(1)))
print(f"{mx:.0f}", orph, unl)
PYEOF
)"
    printf '%-4s %-26s %-9s %-9s %-8s %-8s %s\n' "$idx" "$tag" "$alive" "$prob" "${lmax:-0}s" "${w:-0}" "$(basename "$rundir")"
    unlocked=$((unlocked + ${u:-0}))
  done < "$MANIFEST"
  echo
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
  if [ "$1" = "-" ] || [ -z "$1" ]; then echo "${RUN_PREFIX}_itr30_GH200"
  else echo "${RUN_PREFIX}_${1}_itr30_GH200"; fi
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

echo "plan: $TOTAL arm(s) on GPU $GPU, staggered ${LAG_SEC}s"
echo "      model=$MODEL  hardware=$HARDWARE"
echo "      results-root=$RESULTS_ROOT"
for ((i=0; i<TOTAL; i++)); do
  printf '      %-26s ctx=%-20s %s\n' "$(run_name_for_tag "${Q_TAG[$i]}")" "${Q_CTX[$i]}" "${Q_FLAGS[$i]}"
done
echo "      ${MAX_PROBLEMS} problems x ${MAX_ITERATIONS} iterations"
echo "      KB_GPU_RESERVE_GB=0, KB_GPU_EVAL_LOCK_TIMEOUT_SEC=$KB_GPU_EVAL_LOCK_TIMEOUT_SEC"
echo "      SKIP_REF=$KB_EVAL_SKIP_DEAD_REF_TIMING HOIST=$KB_EVAL_HOIST_INPUT_GEN UNLOCK_CORR=$KB_EVAL_UNLOCK_CORRECTNESS LOCK_SLOTS=$KB_GPU_EVAL_LOCK_SLOTS MEM_GATE=$KB_EVAL_MEM_GATE_FACTOR, phase log per arm"
echo

if [ "$MODE" = "dry-run" ]; then echo "(dry-run: nothing launched)"; exit 0; fi

MANIFEST="${MANIFEST:-$MANIFEST_DEFAULT}"
printf 'idx\ttag\tpid\trundir\tlog\n' > "$MANIFEST"

for ((i=0; i<TOTAL; i++)); do
  tag="${Q_TAG[$i]}"; ctx="${Q_CTX[$i]}"; flags="${Q_FLAGS[$i]}"
  RUN_NAME="$(run_name_for_tag "$tag")"
  LOG="${RUN_NAME}_${STAMP}_wave.log"
  # One JSON line per eval: held_sec, waited_sec and a non-overlapping phase
  # breakdown. A FILE, never stdout -- eval stdout is spliced into the agent's
  # prompt (governor.py:1203), so printing telemetry would change LLM input.
  PHASE_LOG="${RUN_NAME}_${STAMP}_phase.jsonl"
  : > "$LOG"

  before="$(ls -1d ${RESULTS_ROOT}${RUN_NAME}_2* 2>/dev/null | sort || true)"
  echo ">> [$((i+1))/$TOTAL] $RUN_NAME (ctx=$ctx) $flags"

  # shellcheck disable=SC2086 -- flags must word-split into separate argv entries
  CUDA_VISIBLE_DEVICES="$GPU" KB_GPU_RESERVE_GB=0 \
  KB_GPU_EVAL_LOCK_TIMEOUT_SEC="$KB_GPU_EVAL_LOCK_TIMEOUT_SEC" \
  KB_EVAL_SKIP_DEAD_REF_TIMING="$KB_EVAL_SKIP_DEAD_REF_TIMING" \
  KB_EVAL_HOIST_INPUT_GEN="$KB_EVAL_HOIST_INPUT_GEN" \
  KB_EVAL_UNLOCK_CORRECTNESS="$KB_EVAL_UNLOCK_CORRECTNESS" \
  KB_GPU_EVAL_LOCK_SLOTS="$KB_GPU_EVAL_LOCK_SLOTS" \
  KB_EVAL_MEM_GATE_FACTOR="$KB_EVAL_MEM_GATE_FACTOR" \
  KB_EVAL_PHASE_LOG="$REPO_ROOT/$PHASE_LOG" \
  setsid nohup uv run --no-sync python \
    scripts_integration/new_evolving_agent/evolve_kb_batch.py \
    --run-name "$RUN_NAME" \
    --results-root "$RESULTS_ROOT" \
    --max-problems "$MAX_PROBLEMS" \
    --max-iterations "$MAX_ITERATIONS" \
    --hardware "$HARDWARE" \
    --nvidia-endpoint inference \
    --model "$MODEL" \
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
