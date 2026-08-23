#!/usr/bin/env bash
# RESUME a wave of arms from their existing run dirs, one arm per spec line.
#
#   RESULTS_ROOT=runs_evolving/gpt-oss-120b/median/ \
#     bash .../env/NVIDIA_GH200x2/resume_wave.sh <gpu> <spec_file> [dry-run]
#
# Same spec format as launch_wave.sh (`tag | context-mode | extra flags`), but
# instead of creating new run dirs it finds the NEWEST existing dir for each
# tag and continues it from the first unfinished problem (resume_run.sh `auto`).
#
# Why this exists rather than calling resume_run.sh six times by hand: the spec
# already carries each arm's governance flags, and those flags are load-bearing
# on resume. evolve_kb_batch.py rebuilds the treatment from the CLI, and
# `_check_resume_config_mismatch` returns [] when run_summary.json is absent
# (evolve_kb_batch.py:675) -- the killed-arm case. Drop a flag by hand and the
# arm silently continues as plain truncation with no error anywhere.

set -euo pipefail

GPU="${1:?usage: resume_wave.sh <gpu> <spec_file> [dry-run]}"
SPEC_FILE="${2:?missing spec file}"
MODE="${3:-launch}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

LAG_SEC="${LAG_SEC:-180}"
MAX_ARMS_PER_GPU="${MAX_ARMS_PER_GPU:-6}"
MODEL="${MODEL:-gpt-oss-120b}"
RESULTS_ROOT="${RESULTS_ROOT:-runs_evolving/gpt-oss-120b/}"
RUN_PREFIX="${RUN_PREFIX:-base_agent_gpt_oss_120b}"
case "$RESULTS_ROOT" in */) ;; *) RESULTS_ROOT="$RESULTS_ROOT/" ;; esac
export MODEL RESULTS_ROOT
STAMP="$(date -u +%b_%-d)"
MANIFEST="wave_gpu${GPU}_${RUN_PREFIX}_${STAMP}_resume.manifest.tsv"
RESUME_SH="$(dirname "${BASH_SOURCE[0]}")/resume_run.sh"

[ -f "$SPEC_FILE" ] || { echo "FATAL: no spec file $SPEC_FILE"; exit 1; }

# --- parse spec (identical rules to launch_wave.sh) -------------------------
TAGS=(); CTXS=(); FLAGS=()
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"
  [ -z "${line// /}" ] && continue
  tag="$(echo "${line%%|*}" | xargs)"
  rest="${line#*|}"
  ctx="$(echo "${rest%%|*}" | xargs)"
  flags=""
  [ "$rest" != "${rest#*|}" ] && flags="$(echo "${rest#*|}" | xargs)"
  case "$ctx" in
    truncation|folding|markov_report|selective_retention|compress_trigger) ;;
    *) echo "FATAL: unknown context-management mode '$ctx' in spec line: $line"; exit 1 ;;
  esac
  TAGS+=("$tag"); CTXS+=("$ctx"); FLAGS+=("$flags")
done < "$SPEC_FILE"
TOTAL="${#TAGS[@]}"
[ "$TOTAL" -gt 0 ] || { echo "FATAL: spec has no arms"; exit 1; }

run_name_for() { if [ "$1" = "-" ] || [ -z "$1" ]; then echo "${RUN_PREFIX}_itr30_GH200"; else echo "${RUN_PREFIX}_${1}_itr30_GH200"; fi; }

# --- resolve every run dir and its resume point BEFORE launching anything ----
echo "=== resume plan: $TOTAL arm(s) on GPU $GPU, staggered ${LAG_SEC}s ==="
DIRS=(); STARTS=()
for i in $(seq 0 $((TOTAL - 1))); do
  base="$(run_name_for "${TAGS[$i]}")"
  # newest timestamped dir for this arm
  dir="$(ls -1d ${RESULTS_ROOT}${base}_2* 2>/dev/null | sort | tail -1 || true)"
  [ -n "$dir" ] || { echo "FATAL: no existing run dir for '$base' under $RESULTS_ROOT"; exit 1; }
  name="$(basename "$dir")"
  if [ -f "$dir/batch_timing.jsonl" ]; then done_n="$(wc -l < "$dir/batch_timing.jsonl")"; else done_n=0; fi
  DIRS+=("$name"); STARTS+=("$((done_n + 1))")
  printf '  %-22s %-20s done=%-3s resume@%-3s %s\n' "${TAGS[$i]}" "${CTXS[$i]}" "$done_n" "$((done_n + 1))" "${FLAGS[$i]:-}"
done

existing=0
for p in $(pgrep -f evolve_kb_batch.py 2>/dev/null || true); do
  d="$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -oP '(?<=^CUDA_VISIBLE_DEVICES=).*' || true)"
  [ "$d" = "$GPU" ] && existing=$((existing + 1))
done
existing=$((existing / 2))   # uv wrapper + python child per arm
echo "  arms already on GPU $GPU: $existing"
[ $((existing + TOTAL)) -gt "$MAX_ARMS_PER_GPU" ] && {
  echo "FATAL: $existing existing + $TOTAL new > MAX_ARMS_PER_GPU=$MAX_ARMS_PER_GPU"; exit 1; }

if [ "$MODE" = "dry-run" ]; then echo "(dry-run: nothing launched)"; exit 0; fi

printf 'idx\ttag\tpid\trundir\tlog\n' > "$MANIFEST"
for i in $(seq 0 $((TOTAL - 1))); do
  name="${DIRS[$i]}"
  echo ">>> [$((i + 1))/$TOTAL] resuming $name at problem ${STARTS[$i]}"
  # shellcheck disable=SC2086  # FLAGS entries are pre-split flag strings
  pid="$(QUIET=1 DO_BACKUP="${DO_BACKUP:-1}" \
    bash "$RESUME_SH" "$GPU" "$name" "${CTXS[$i]}" auto -- ${FLAGS[$i]} | tail -1)"
  log="${name%%_2026_*}_resume_${STAMP}.log"
  printf '%s\t%s\t%s\t%s\t%s\n' "$((i + 1))" "${TAGS[$i]}" "$pid" "${RESULTS_ROOT}${name}" "$log" >> "$MANIFEST"
  echo "    pid=$pid  log=$log"
  [ "$i" -lt $((TOTAL - 1)) ] && sleep "$LAG_SEC"
done

echo
echo "manifest: $MANIFEST"
echo "audit lock contention with (the arm .log NEVER contains lock lines):"
echo "  bash $(dirname "${BASH_SOURCE[0]}")/../common/wave_status.sh"
