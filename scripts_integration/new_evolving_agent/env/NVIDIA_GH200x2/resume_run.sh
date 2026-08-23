#!/usr/bin/env bash
# Resume an existing evolving-agent run, continuing from the first UNFINISHED
# problem (or an explicit range).
#
#   bash .../env/NVIDIA_GH200x2/resume_run.sh <gpu> <run_dir_name> <ctx_mode> [start] [end] [-- extra flags...]
#
# `start` may be "auto" (or omitted) -- it is then derived from batch_timing.jsonl
# as (completed problems + 1), i.e. re-run the problem that was in flight when the
# run died plus everything after it.
#
# Example -- resume a killed wave arm, keeping its treatment:
#   RESULTS_ROOT=runs_evolving/gpt-oss-120b/median/ \
#   bash .../env/NVIDIA_GH200x2/resume_run.sh 0 \
#     base_agent_gpt_oss_120b_deletion_itr30_GH200_2026_08_22_21_23 truncation auto -- --skill-deletion
#
# NARROW RANGES ARE SAFE -- two mechanisms cooperate
# --------------------------------------------------
# 1. Disk-level purge: removes only L1 entries *sourced from problems inside*
#    [start, end] (plus refine/merge descendants via parent_id / merge_meta).
#    Entries from later problems stay in shared_l1.jsonl. L2 has the equivalent
#    (rollback_l2_for_resume).
#
# 2. Prompt-level causal filter: while replaying the problem at subset index N,
#    `collect_causal_l1_entry_ids` (evolve_kb_batch.py) restricts the visible
#    catalog to entries whose provenance is strictly < N, via
#    KBGovernorConfig.l1_allowed_entry_ids. Entries whose provenance cannot be
#    resolved are excluded (conservative).
#
# So a replayed problem never sees skills learned after it, even though those
# skills remain on disk. Measured on the markov run: replaying index 39 shows
# 267/344 entries, provenance 1..38, zero future leakage.
#
# ORDER MATTERS for multiple narrow resumes: run the EARLIER index first, so
# its repaired skills are visible (and causally valid) to the later one.
#
# WHY THE `--` FLAG PASSTHROUGH IS LOAD-BEARING
# ---------------------------------------------
# evolve_kb_batch.py reconstructs a run's treatment from the CLI, not from disk.
# `_check_resume_config_mismatch` would normally abort when the flags disagree
# with the original run -- but it returns [] when run_summary.json is absent
# (evolve_kb_batch.py:675), which is exactly the killed-arm case. Before this
# script took extra flags, resuming a governance arm silently continued it as a
# plain truncation arm with no error. Pass the arm's flags after `--`, always.

set -euo pipefail

usage() { echo "usage: resume_run.sh <gpu> <run_dir_name> <ctx_mode> [start|auto] [end] [-- extra flags...]"; exit 1; }

GPU="${1:?$(usage)}"
RUN_NAME="${2:?missing run_dir_name (must include the timestamp suffix)}"
CTX="${3:?missing context-management mode}"
shift 3
START="auto"; END=""
[ $# -gt 0 ] && [ "$1" != "--" ] && { START="$1"; shift; }
[ $# -gt 0 ] && [ "$1" != "--" ] && { END="$1";   shift; }
[ "${1:-}" = "--" ] && shift
EXTRA_FLAGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

# Parameterised so this addresses runs under median/ (or any other results root)
# and a second model. The old hardcoded value could not reach runs_evolving/*/median/
# at all and exited "FATAL: no such run dir".
MODEL="${MODEL:-gpt-oss-120b}"
RESULTS_ROOT="${RESULTS_ROOT:-runs_evolving/gpt-oss-120b/}"
case "$RESULTS_ROOT" in */) ;; *) RESULTS_ROOT="$RESULTS_ROOT/" ;; esac
RUN_DIR="${RESULTS_ROOT}${RUN_NAME}"

MAX_PROBLEMS="${MAX_PROBLEMS:-50}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
DO_BACKUP="${DO_BACKUP:-1}"

# Critical-section trims + lock settings, matching launch_wave.sh. Both trims
# default OFF in src/kernelbench/eval.py so a run already in flight is never
# perturbed; they are turned on HERE, i.e. only for what this script launches.
KB_GPU_EVAL_LOCK_TIMEOUT_SEC="${KB_GPU_EVAL_LOCK_TIMEOUT_SEC:-5400}"
KB_EVAL_SKIP_DEAD_REF_TIMING="${KB_EVAL_SKIP_DEAD_REF_TIMING:-1}"
KB_EVAL_HOIST_INPUT_GEN="${KB_EVAL_HOIST_INPUT_GEN:-1}"
# How many evals may hold the GPU at once. 1 == the historical mutex and leaves
# the lock file path unchanged. Measured 2026-08-23 on this host (6 kernels x 5
# repeats, trims on): degree 2 inflates the measured runtime by 1.2% median /
# 2.3% worst, degree 3 by 0.7% / 2.8% -- an order of magnitude under the ~20-30%
# replicate noise. Never mix slots=1 and slots>1 against one GPU: the file names
# differ, so they would not interlock.
KB_GPU_EVAL_LOCK_SLOTS="${KB_GPU_EVAL_LOCK_SLOTS:-1}"
# Correctness trials out of the lock, leaving only the timing window(s) held.
# Measured A/B on L1P100 (warm build, trims on): hold 1.96s -> 0.78s with the
# measured runtime unchanged (2.42/2.44 vs 2.38/2.45 ms).
KB_EVAL_UNLOCK_CORRECTNESS="${KB_EVAL_UNLOCK_CORRECTNESS:-1}"
export KB_GPU_EVAL_LOCK_TIMEOUT_SEC KB_EVAL_SKIP_DEAD_REF_TIMING KB_EVAL_HOIST_INPUT_GEN KB_GPU_EVAL_LOCK_SLOTS KB_EVAL_UNLOCK_CORRECTNESS

# Hardware/baseline is a parameter so this script works on another server.
# Unlike the launchers, resume DEFAULTS TO THE ORIGINAL RUN'S baseline: replaying
# a range against a different one would make the repaired problems incomparable
# with the untouched ones in the same run dir.
# This server's baseline is results/timing/NVIDIA_GH200x2_median/ -- levels 1-2
# from the batch run plus level 3 re-measured in fresh processes (median-of-3).
# Without this, hardware_env.sh's folder-name auto-resolution would pick
# results/timing/NVIDIA_GH200x2/, which predates fix(eval) 6a3e972 and carries no
# median field on any of its 249 entries, so kb_require_hardware hard-FATALs.
KB_DEFAULT_HARDWARE="${KB_DEFAULT_HARDWARE:-NVIDIA_GH200x2_median}"
# shellcheck source=../hardware_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../hardware_env.sh"
if [ -z "${HARDWARE:-}" ] && [ -f "$RUN_DIR/run_summary.json" ]; then
  HARDWARE="$(./.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1])).get('hardware_server',''))" "$RUN_DIR/run_summary.json" 2>/dev/null || true)"
  [ -n "$HARDWARE" ] && echo ">> hardware from run_summary.json: $HARDWARE"
fi
kb_resolve_hardware
kb_require_hardware "$REPO_ROOT"

# --- preflight -------------------------------------------------------------
[ -d "$RUN_DIR" ] || { echo "FATAL: no such run dir: $RUN_DIR"; exit 1; }
command -v nvcc  >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH -- add .venv/bin"; exit 1; }

# A killed arm has NO run_summary.json (it is written only at run end), so the
# old hard requirement made every crash-resume impossible. Downgraded to a
# warning -- but note loudly that the config-mismatch guard is inert here, which
# is precisely why EXTRA_FLAGS must carry the arm's treatment.
if [ ! -f "$RUN_DIR/run_summary.json" ]; then
  echo ">> NOTE: no run_summary.json (run was killed mid-flight)."
  echo "         _check_resume_config_mismatch cannot verify the treatment; it returns []"
  echo "         when the summary is missing. The flags below are taken on trust:"
  echo "         extra flags: ${EXTRA_FLAGS[*]:-<none>}"
fi

# Free-memory guard rather than "any memory in use". The old `used > 1000`
# check refused to put a second arm on a GPU (an idle arm holds ~558 MiB),
# which makes a multi-arm resume wave impossible.
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")"
[ "$free_mib" -lt "$MIN_FREE_MIB" ] && { echo "FATAL: GPU $GPU has only ${free_mib} MiB free (< $MIN_FREE_MIB)"; exit 1; }

# --- resolve the resume point ----------------------------------------------
if [ "$START" = "auto" ]; then
  if [ -f "$RUN_DIR/batch_timing.jsonl" ]; then
    done_n="$(wc -l < "$RUN_DIR/batch_timing.jsonl")"
  else
    done_n=0   # killed before finishing problem 1 -- no batch_timing.jsonl at all
  fi
  START=$((done_n + 1))
  echo ">> auto start: $done_n problem(s) completed -> resuming at subset index $START"
fi
if [ "$START" -gt "$MAX_PROBLEMS" ]; then
  echo ">> nothing to do: start ($START) is past --max-problems ($MAX_PROBLEMS)"; exit 0
fi

STAMP="$(date -u +%b_%-d)"
LOG="${RUN_NAME%%_2026_*}_resume_${STAMP}.log"
PHASE_LOG="${RUN_NAME%%_2026_*}_resume_${STAMP}_phase.jsonl"

if [ "${DRYRUN:-0}" = "1" ]; then
  echo ">> DRYRUN: would resume $RUN_NAME on GPU $GPU"
  echo "   ctx=$CTX  start=$START  end=${END:-<end>}  model=$MODEL  hardware=$HARDWARE"
  echo "   results-root=$RESULTS_ROOT"
  echo "   extra flags: ${EXTRA_FLAGS[*]:-<none>}"
  echo "   HOIST=$KB_EVAL_HOIST_INPUT_GEN SKIP_REF=$KB_EVAL_SKIP_DEAD_REF_TIMING LOCK_SLOTS=$KB_GPU_EVAL_LOCK_SLOTS UNLOCK_CORR=$KB_EVAL_UNLOCK_CORRECTNESS"
  echo "   log=$LOG  phase-log=$PHASE_LOG"
  exit 0
fi

# Resuming rewrites results in-place and destructively purges L1/L2, so snapshot
# the run dir first. Set DO_BACKUP=0 for a barely-started arm where the tar is
# pure overhead.
if [ "$DO_BACKUP" = "1" ]; then
  BACKUP="${RUN_DIR}.preresume.$(date -u +%Y%m%d_%H%M%S).tar.gz"
  echo ">> backing up $RUN_DIR -> $BACKUP"
  tar czf "$BACKUP" -C "$RESULTS_ROOT" "$RUN_NAME"
  echo "   $(du -h "$BACKUP" | cut -f1)"
else
  BACKUP="(skipped, DO_BACKUP=0)"
fi

END_ARGS=()
[ -n "$END" ] && END_ARGS=(--end-problem "$END")
echo ">> GPU $GPU  resume $RUN_NAME  ctx=$CTX  problems ${START}..${END:-end}"
echo ">> model=$MODEL  hardware=$HARDWARE  results-root=$RESULTS_ROOT"
echo ">> extra flags: ${EXTRA_FLAGS[*]:-<none>}"
echo ">> KB_GPU_RESERVE_GB=0 HOIST=$KB_EVAL_HOIST_INPUT_GEN SKIP_REF=$KB_EVAL_SKIP_DEAD_REF_TIMING LOCK_SLOTS=$KB_GPU_EVAL_LOCK_SLOTS UNLOCK_CORR=$KB_EVAL_UNLOCK_CORRECTNESS"
echo ">> log: $LOG   phase log: $PHASE_LOG"

# shellcheck disable=SC2086  # EXTRA_FLAGS is an array; intentional word-splitting of none
CUDA_VISIBLE_DEVICES="$GPU" KB_GPU_RESERVE_GB=0 \
KB_EVAL_PHASE_LOG="$REPO_ROOT/$PHASE_LOG" \
  KB_GPU_EVAL_LOCK_SLOTS="$KB_GPU_EVAL_LOCK_SLOTS" \
  KB_EVAL_UNLOCK_CORRECTNESS="$KB_EVAL_UNLOCK_CORRECTNESS" \
setsid nohup uv run --no-sync python \
  scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name "$RUN_NAME" \
  --results-root "$RESULTS_ROOT" \
  --max-problems "$MAX_PROBLEMS" \
  --max-iterations "$MAX_ITERATIONS" \
  --start-problem "$START" \
  "${END_ARGS[@]}" \
  --hardware "$HARDWARE" \
  --nvidia-endpoint inference \
  --model "$MODEL" \
  --context-management "$CTX" \
  --coder-timeout-sec 600 \
  --backup-l1-on-resume \
  "${EXTRA_FLAGS[@]}" \
  >> "$LOG" 2>&1 &

PID=$!
echo "   pid=$PID"
if [ "${QUIET:-0}" = "1" ]; then
  echo "$PID"
  exit 0
fi
sleep 40
echo
echo "=== L1 purge summary (what the resume removed) ==="
grep -iE "purge|removed_count|kept_count|rollback" "$LOG" | head -8 || echo "(none logged yet)"
echo
echo "=== process ==="
pgrep -af "evolve_kb_batch.*$RUN_NAME" | cut -c1-140 || echo "NOT RUNNING -- check $LOG"
cat <<EOF

Backup: $BACKUP
Watch:  tail -f $LOG
Health: uv run --no-sync python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
EOF
