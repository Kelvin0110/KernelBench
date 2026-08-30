#!/usr/bin/env bash
# Resume a damaged range of an existing evolving-agent run on the repaired
# toolchain + LLM-timeout harness.
#
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/resume_run.sh <gpu> <run_dir_name> <ctx_mode> <start> [end]
#
# Example -- replay markov problems 39..50 (clean suffix, recovers L3P29/L3P48/L3P49):
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/resume_run.sh \
#     1 base_agent_gpt_oss_120b_markov_itr30_GH200_2026_08_07_13_58 markov_report 39
#
# NARROW RANGES ARE SAFE -- two mechanisms cooperate
# --------------------------------------------------
# 1. Disk-level purge: removes only L1 entries *sourced from problems inside*
#    [start, end] (plus refine/merge descendants via parent_id / merge_meta).
#    Entries from later problems stay in shared_l1.jsonl.
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
#   e.g.  ... markov_report 39 39     then     ... markov_report 47 48
#
# Caveat: problems between the replayed ranges are NOT re-run, so they keep
# results formed under the pre-repair L1. Usually minor; note it in the report.

set -euo pipefail

GPU="${1:?usage: resume_run.sh <gpu> <run_dir_name> <ctx_mode> <start> [end]}"
RUN_NAME="${2:?missing run_dir_name (must include the timestamp suffix)}"
CTX="${3:?missing context-management mode}"
START="${4:?missing --start-problem}"
# END is OPTIONAL and `--` may sit in its slot. Guard it, or END becomes "--" and the
# runner is handed a literal `--end-problem --`.
END="${5:-}"
[ "$END" = "--" ] && END=""
shift 4 2>/dev/null || true
[ $# -gt 0 ] && [ "${1:-}" != "--" ] && shift || true   # drop the optional END positional
EXTRA_ARGS=()
if [ "${1:-}" = "--" ]; then
  shift
  EXTRA_ARGS=("$@")
fi

# DRYRUN=1 resolves everything and prints the exact command WITHOUT launching.
# Added 2026-08-29: this script had no dry-run at all, so `DRYRUN=1 ...` silently
# launched a real 25-hour arm. CLAUDE.md 3.6 documents DRYRUN as working -- that is
# the OTHER host's copy; this one never had it.
DRYRUN="${DRYRUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

# Parameterised 2026-08-29. Was hardcoded to runs_evolving/gpt-oss-120b/, which made
# this script unable to address ANY terra run -- it exited "FATAL: no such run dir".
# Same defect CLAUDE.md 3.6 records as fixed on the other host; it was never ported here.
RESULTS_ROOT="${RESULTS_ROOT:-runs_evolving/gpt-oss-120b/}"
case "$RESULTS_ROOT" in */) ;; *) RESULTS_ROOT="$RESULTS_ROOT/" ;; esac
MODEL="${MODEL:-gpt-oss-120b}"
RUN_DIR="${RESULTS_ROOT}${RUN_NAME}"

# Hardware/baseline is a parameter so this script works on another server.
# Unlike the launchers, resume DEFAULTS TO THE ORIGINAL RUN'S baseline: replaying
# a range against a different one would make the repaired problems incomparable
# with the untouched ones in the same run dir.
# shellcheck source=./hardware_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../hardware_env.sh"
if [ -z "${HARDWARE:-}" ] && [ -f "$RUN_DIR/run_summary.json" ]; then
  HARDWARE="$(./.venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1])).get('hardware_server',''))" "$RUN_DIR/run_summary.json" 2>/dev/null || true)"
  [ -n "$HARDWARE" ] && echo ">> hardware from run_summary.json: $HARDWARE"
fi
ALLOW_MEAN_BASELINE=1 kb_require_hardware "$REPO_ROOT"

LOG="${RUN_NAME%%_2026_*}_resume_$(date -u +%b_%-d).log"

# --- preflight -------------------------------------------------------------
[ -d "$RUN_DIR" ]                 || { echo "FATAL: no such run dir: $RUN_DIR"; exit 1; }
# run_summary.json is written only at run END, so hard-requiring it refused every
# CRASH-resume -- the exact case this script exists for. CLAUDE.md 3.6 defect 2,
# fixed on the other host and never ported here. Now it is advisory: a missing
# summary just means the run never finished, which is normal for a resume. The
# batch_timing.jsonl check below is the real "is this a run dir" test.
if [ ! -f "$RUN_DIR/run_summary.json" ]; then
  echo ">> note: no run_summary.json (run did not finish) -- normal for a crash-resume"
  [ -f "$RUN_DIR/batch_timing.jsonl" ] || { echo "FATAL: $RUN_DIR has neither run_summary.json nor batch_timing.jsonl -- not a run dir"; exit 1; }
fi
command -v nvcc  >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH -- add .venv/bin"; exit 1; }

# CLAUDE.md 3.6 defect 4, ported 2026-08-29. The old test was `used > 1000 MiB`, but a
# single IDLE arm holds ~558 MiB, so arm 2 onward was refused and a multi-arm resume was
# impossible -- 8 of 12 arms exited here, silently, before their launch line. Gate on FREE
# memory instead, matching launch_wave.sh (MIN_FREE_MIB=20000, MAX_ARMS_PER_GPU).
# EVAL-LOCK ENV -- MUST MATCH launch_wave.sh, ported 2026-08-29.
# resume_run.sh set NONE of these, so every resumed arm silently ran with:
#   KB_EVAL_MEM_GATE_FACTOR unset -> 0     = memory gate OFF
#   KB_GPU_EVAL_LOCK_SLOTS  unset -> 1     = the historical mutex, not the 3-slot semaphore
#   KB_EVAL_HOIST_INPUT_GEN unset -> False = gate inert even if the factor were set
#   KB_GPU_RESERVE_GB       unset -> 42 GB PER ARM = 252 GB across 6 arms on a 143 GB card
# Two hazards at once: the reserver fights itself for headroom, and with the gate off
# nothing bounds concurrent device residents on the big-input problems -- the documented
# OOM path, where an OOM is recorded compiled=True correct=False and the governor then
# "debugs" a kernel that was never broken. It is ALSO a protocol seam: the resumed suffix
# would run under different eval settings than its own prefix, inside one run.
KB_GPU_RESERVE_GB="${KB_GPU_RESERVE_GB:-0}"
KB_GPU_EVAL_LOCK_TIMEOUT_SEC="${KB_GPU_EVAL_LOCK_TIMEOUT_SEC:-5400}"
KB_EVAL_SKIP_DEAD_REF_TIMING="${KB_EVAL_SKIP_DEAD_REF_TIMING:-1}"
KB_EVAL_HOIST_INPUT_GEN="${KB_EVAL_HOIST_INPUT_GEN:-1}"
KB_EVAL_UNLOCK_CORRECTNESS="${KB_EVAL_UNLOCK_CORRECTNESS:-0}"
KB_GPU_EVAL_LOCK_SLOTS="${KB_GPU_EVAL_LOCK_SLOTS:-3}"
KB_EVAL_MEM_GATE_FACTOR="${KB_EVAL_MEM_GATE_FACTOR:-7}"
KB_EVAL_MEM_GATE_TIMEOUT_SEC="${KB_EVAL_MEM_GATE_TIMEOUT_SEC:-1800}"
export KB_GPU_RESERVE_GB KB_GPU_EVAL_LOCK_TIMEOUT_SEC KB_EVAL_SKIP_DEAD_REF_TIMING \
       KB_EVAL_HOIST_INPUT_GEN KB_EVAL_UNLOCK_CORRECTNESS KB_GPU_EVAL_LOCK_SLOTS \
       KB_EVAL_MEM_GATE_FACTOR KB_EVAL_MEM_GATE_TIMEOUT_SEC
export KB_EVAL_PHASE_LOG="${KB_EVAL_PHASE_LOG:-$REPO_ROOT/${RUN_NAME}_resume_phase.jsonl}"

MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")"
[ "$free_mib" -lt "$MIN_FREE_MIB" ] && { echo "FATAL: GPU $GPU has only ${free_mib} MiB free (need $MIN_FREE_MIB). Refusing."; exit 1; }
MAX_ARMS_PER_GPU="${MAX_ARMS_PER_GPU:-12}"
# Count arms on THIS gpu. /proc/<pid>/environ of other users' processes is
# unreadable; without 2>/dev/null + `|| true` the EPERM trips `set -e` and the whole
# script dies before launching -- which is exactly what silently ate 8 arms.
existing=$( { for p in /proc/[0-9]*; do
  [ -r "$p/environ" ] || continue
  tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=${GPU}$" || continue
  tr '\0' ' '  < "$p/cmdline" 2>/dev/null | grep -q 'evolve_kb_batch.py' && echo x
done; } 2>/dev/null | wc -l || true )
existing=${existing:-0}
[ "$existing" -ge "$MAX_ARMS_PER_GPU" ] && { echo "FATAL: GPU $GPU already has $existing arms (max $MAX_ARMS_PER_GPU)."; exit 1; }
echo ">> GPU $GPU: ${free_mib} MiB free, $existing arm(s) already running"
echo ">> eval env: RESERVE_GB=$KB_GPU_RESERVE_GB SLOTS=$KB_GPU_EVAL_LOCK_SLOTS MEM_GATE=$KB_EVAL_MEM_GATE_FACTOR HOIST=$KB_EVAL_HOIST_INPUT_GEN UNLOCK_CORR=$KB_EVAL_UNLOCK_CORRECTNESS"

# Resuming rewrites results in-place and destructively purges L1. The run being
# repaired cost ~68 GPU-hours, so snapshot the whole run dir first.
BACKUP="${RUN_DIR}.prersume.$(date -u +%Y%m%d_%H%M%S).tar.gz"
echo ">> backing up $RUN_DIR -> $BACKUP"
tar czf "$BACKUP" -C "$RESULTS_ROOT" "$RUN_NAME"
echo "   $(du -h "$BACKUP" | cut -f1)"

END_ARGS=()
[ -n "$END" ] && END_ARGS=(--end-problem "$END")
echo ">> GPU $GPU  resume $RUN_NAME  ctx=$CTX  problems ${START}..${END:-end}"
echo ">> log: $LOG"

# GOVERNANCE FLAGS MUST REACH THE RUNNER. evolve_kb_batch.py rebuilds a run's
# treatment from the CLI, and _check_resume_config_mismatch returns [] when
# run_summary.json is absent (evolve_kb_batch.py:675) -- exactly the crash-resume
# case. Without this passthrough a deletion/merge/refinement/l2 arm resumes as a
# PLAIN TRUNCATION arm with nothing anywhere to catch it. CLAUDE.md 3.6 defect 3.
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  echo ">> extra flags: ${EXTRA_ARGS[*]}"
else
  echo ">> extra flags: (none -- correct ONLY for truncation/markov/folding/selective/compress arms)"
fi

if [ "$DRYRUN" = "1" ]; then
  echo ">> DRYRUN=1 -- would launch, but not launching:"
  echo "   CUDA_VISIBLE_DEVICES=$GPU uv run --no-sync python \\"
  echo "     scripts_integration/new_evolving_agent/evolve_kb_batch.py --resume \\"
  echo "     --run-name $RUN_NAME --results-root $RESULTS_ROOT \\"
  echo "     --start-problem $START ${END:+--end-problem $END} \\"
  echo "     --hardware $HARDWARE --model $MODEL --context-management $CTX \\"
  echo "     ${EXTRA_ARGS[*]}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" nohup uv run --no-sync python \
  scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name "$RUN_NAME" \
  --results-root "$RESULTS_ROOT" \
  --max-problems 50 \
  --max-iterations 30 \
  --start-problem "$START" \
  "${END_ARGS[@]}" \
  --hardware "$HARDWARE" \
  --nvidia-endpoint inference \
  --model "$MODEL" \
  --context-management "$CTX" \
  --coder-timeout-sec 600 \
  --backup-l1-on-resume \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG" 2>&1 &

echo "   pid=$!"
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
