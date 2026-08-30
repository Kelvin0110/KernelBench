#!/usr/bin/env bash
# Detached wave supervisor: keeps watch on a running wave and RESTARTS arms that die.
#
# WHY THIS EXISTS. nohup/setsid already survive an SSH disconnect -- a dropped
# connection cannot kill an arm. What it does NOT survive is an arm dying on its own
# (OOM-killer, unhandled exception, endpoint exhaustion), and on 2026-08-29 an arm
# could also "finish" EARLY and look healthy while writing garbage. So this watches
# for arms that VANISH and brings them back from their own resume point.
#
#   setsid bash wave_supervisor.sh <spec.tsv> [interval_sec] &
#
# spec.tsv columns (tab-separated): gpu  run_dir  ctx  start  flags  [results_root]
# results_root defaults to runs_evolving/gpt-5.6-terra/ when the column is absent, so
# older single-model specs still work. It is REQUIRED once a wave mixes models --
# a terra arm and a qwen arm live under different roots and the supervisor would
# otherwise look for every run dir under the terra tree and declare them all dead.
# `start` is only the ORIGINAL start; on restart the supervisor recomputes it from
# batch_timing.jsonl (completed+1) so it never redoes finished problems.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1
REPO="$PWD"
SPEC="${1:?usage: wave_supervisor.sh <spec.tsv> [interval_sec]}"
INTERVAL="${2:-300}"
LOG="$REPO/wave_supervisor.log"
HW="$REPO/scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd"
export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO/.venv/bin:$PATH"

say(){ echo "$(date -u '+%Y-%m-%d %H:%M:%SZ')  $*" >> "$LOG"; }
# TWO NAME SHAPES. A resumed arm carries --run-name <dir INCLUDING the timestamp>,
# because resume_run.sh is given the run dir. A launch_wave arm carries --run-name
# <base WITHOUT timestamp> and the runner appends _YYYY_MM_DD_HH_MM to form the dir.
# An exact match therefore reports every launch_wave arm as DEAD and tries to restart
# a live wave. Accept the dir itself OR any live run-name that is its timestamp prefix.
alive(){
  local dir="$1" rn
  while read -r rn; do
    [ -z "$rn" ] && continue
    [ "$rn" = "$dir" ] && return 0
    case "$dir" in "${rn}_"[0-9]*) return 0 ;; esac
  done < <(ps -eo cmd= | grep -oP '(?<=--run-name )\S+' | sort -u)
  return 1
}
# "finished" = the last recorded subset_index reached 50, not a line count (see above).
done50(){
  local last
  last=$(tail -1 "${2:-runs_evolving/gpt-5.6-terra/}$1/batch_timing.jsonl" 2>/dev/null \
    | "$REPO/.venv/bin/python" -c 'import sys,json;
try: print(json.loads(sys.stdin.read() or "{}").get("subset_index",0) or 0)
except Exception: print(0)' 2>/dev/null || echo 0)
  [ "${last:-0}" -ge 50 ]
}

say "SUPERVISOR START spec=$SPEC interval=${INTERVAL}s"
while :; do
  live=0; fin=0; restarted=0
  while IFS=$'\t' read -r gpu run ctx start flags root; do
    [ -z "${run:-}" ] && continue
    root="${root:-runs_evolving/gpt-5.6-terra/}"
    case "$root" in */) ;; *) root="$root/" ;; esac
    if alive "$run"; then live=$((live+1)); continue; fi
    if done50 "$run" "$root"; then fin=$((fin+1)); continue; fi
    # Dead but unfinished -> restart from completed+1, never from the original index.
    if [ ! -d "${root}$run" ]; then
      say "SKIP $run -- run dir does not exist yet (arm may still be starting)"
      continue
    fi
    # RESUME POINT = last record's subset_index + 1, NOT the line count.
    # A resume APPENDS replayed problems to batch_timing.jsonl, so after one resume the
    # file holds more lines than there are problems (r4_deletion: 61 lines, 50 distinct
    # indices, last index 25). Using wc -l would have restarted it at 62 -- past the end
    # of a 50-problem run -- instead of 26. Entries are appended in execution order, so
    # the LAST line is always the most recent pass.
    n=$(tail -1 "${root}$run/batch_timing.jsonl" 2>/dev/null \
        | "$REPO/.venv/bin/python" -c 'import sys,json;
try: print(json.loads(sys.stdin.read() or "{}").get("subset_index",0) or 0)
except Exception: print(0)' 2>/dev/null || echo 0)
    n=${n:-0}
    from=$((n+1))
    if [ "$from" -gt 50 ]; then say "SKIP $run -- last index $n, nothing left"; continue; fi
    say "RESTART $run (gpu$gpu) died at completed=$n -> resuming from $from  flags=[${flags:-none}]"
    case "$root" in *qwen*) mdl=qwen3.6-27b ;; *terra*) mdl=gpt-5.6-terra ;; *) mdl=gpt-oss-120b ;; esac
    RESULTS_ROOT="$root" MODEL="$mdl" DO_BACKUP=0 \
      bash "$HW/resume_run.sh" "$gpu" "$run" "$ctx" "$from" ${flags:+-- $flags} >> "$LOG" 2>&1
    restarted=$((restarted+1))
  done < "$SPEC"
  total=$(wc -l < "$SPEC")
  say "HEALTH live=$live finished=$fin restarted=$restarted of $total"
  [ $((live+fin)) -eq "$total" ] && [ "$live" -eq 0 ] && { say "ALL ARMS FINISHED -- supervisor exiting"; break; }
  sleep "$INTERVAL"
done
