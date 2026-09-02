#!/usr/bin/env bash
# wave_backfill.sh -- keep each GPU topped up to TARGET_PER_GPU live arms by
# launching queued arms ONE AT A TIME as running arms finish.
#
# Runs detached (start it with setsid, see USAGE) so it survives an ssh drop.
#
# WHY A SEPARATE DAEMON FROM wave_supervisor.sh: the supervisor RESTARTS arms that
# died. This one LAUNCHES arms that never started. Mixing them would make "an arm is
# absent" ambiguous between the two responses.
#
# ------------------------------------------------------------------ two subtleties
# 1. ARM COUNTING. launch_wave.sh counts `pgrep evolve_kb_batch | wc / 2`, a
#    parent+child heuristic that skews whenever an eval child is mid-spawn. This
#    daemon counts DISTINCT --run-name tokens instead, which is the source of truth
#    CLAUDE.md 3.4 prescribes. It also excludes killed arms' leftover run dirs by
#    construction, since it never looks at directories.
# 2. MANIFEST CLOBBERING. launch_wave.sh's default manifest name is keyed only on
#    gpu+prefix+day and it truncates the file (`printf > "$MANIFEST"`). Two launches
#    on one day would leave only the second arm on record. Every launch here gets its
#    own MANIFEST, so each backfilled arm keeps a manifest row.
#
# USAGE
#   cd <repo root>
#   QUEUE=<path/to.queue> setsid nohup bash env/common/wave_backfill.sh >/dev/null 2>&1 &
#
# ENV
#   QUEUE            (required) spec-format file: `tag | context-mode | extra flags`
#   STATE            default $QUEUE.state -- tags already launched, one per line
#   LOG              default $QUEUE.log
#   TARGET_PER_GPU   default 9   -- launch only while a GPU is BELOW this
#   MAX_PER_GPU      default 12  -- passed through as MAX_ARMS_PER_GPU
#   INTERVAL         default 300 seconds
#   MODEL / RESULTS_ROOT / RUN_PREFIX / HW   -- passed to launch_wave.sh
#   SUPERVISOR_SPEC  optional wave_supervisor.sh spec.tsv; each launched arm is
#                    appended to it so the supervisor covers backfilled arms too
set -uo pipefail

REPO="${REPO:-/localhome/local-tianzheng/KernelBench}"
HW="${HW:-$REPO/scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd}"
QUEUE="${QUEUE:?set QUEUE=<queue file>}"
case "$QUEUE" in /*) ;; *) QUEUE="$REPO/$QUEUE" ;; esac
STATE="${STATE:-$QUEUE.state}"
LOG="${LOG:-$QUEUE.log}"
TARGET_PER_GPU="${TARGET_PER_GPU:-9}"
MAX_PER_GPU="${MAX_PER_GPU:-12}"
INTERVAL="${INTERVAL:-300}"
MODEL="${MODEL:-gpt-oss-120b}"
RESULTS_ROOT="${RESULTS_ROOT:-runs_evolving/gpt-oss-120b/}"
RUN_PREFIX="${RUN_PREFIX:-base_agent_gpt_oss_120b_r2}"
GPUS="${GPUS:-0 1}"
SUPERVISOR_SPEC="${SUPERVISOR_SPEC:-}"
MONITOR_GLOB="${MONITOR_GLOB:-}"
MONITOR_LOG="${MONITOR_LOG:-$REPO/wave_monitor.log}"

# wave_monitor.py EXITS (return 1) the moment its manifest set changes, and every
# backfill launch writes a new manifest -- so without this the monitor dies on the
# first backfill and the rest of the wave runs unwatched. Observed 2026-09-01.
# The /proc scan inspects argv[0..1] ONLY, so a `bash -c` whose TEXT mentions
# wave_monitor.py is not matched. A `pgrep -f wave_monitor` would match this very
# script's own command line -- that self-match already killed one tool session today.
monitor_pids(){
  local d a
  for d in /proc/[0-9]*; do
    a="$(tr '\0' '\n' < "$d/cmdline" 2>/dev/null | head -2 | tr '\n' ' ')"
    case "$a" in *python*wave_monitor.py*) printf '%s\n' "${d#/proc/}" ;; esac
  done
}
restart_monitor(){
  [ -n "$MONITOR_GLOB" ] || return 0
  local pid
  for pid in $(monitor_pids); do
    grep -q -- "$MONITOR_LOG" "/proc/$pid/cmdline" 2>/dev/null && kill "$pid" 2>/dev/null
  done
  sleep 3
  ( cd "$REPO" && setsid nohup "$REPO/.venv/bin/python" \
      scripts_integration/new_evolving_agent/env/common/wave_monitor.py \
      --interval 300 --log "$MONITOR_LOG" --manifests "$MONITOR_GLOB" \
      >/dev/null 2>&1 < /dev/null & )
  sleep 3
  log "MONITOR restarted for glob=$MONITOR_GLOB (pids: $(monitor_pids | tr '\n' ' '))"
}
case "$RESULTS_ROOT" in */) ;; *) RESULTS_ROOT="$RESULTS_ROOT/" ;; esac

touch "$STATE" "$LOG"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

# Singleton. NOT flock: `exec 9>lock` leaks the locked fd to EVERY child, so a
# stray `sleep` -- or worse, a backfilled ARM, which lives ~50 h -- keeps the lock
# held long after this daemon dies, and no replacement can ever start. Observed
# 2026-09-01: an orphaned `sleep` held it and blocked two restarts. A pid file has
# no inheritance semantics at all.
PIDFILE="$QUEUE.pid"
if [ -f "$PIDFILE" ]; then
  _old="$(cat "$PIDFILE" 2>/dev/null || true)"
  # guard against pid reuse: the pid must still BE a wave_backfill
  if [ -n "${_old:-}" ] && kill -0 "$_old" 2>/dev/null \
     && tr '\0' ' ' < "/proc/$_old/cmdline" 2>/dev/null | grep -q 'wave_backfill'; then
    log "another wave_backfill (pid $_old) is running -- exiting"; exit 0
  fi
  log "stale pid file (pid ${_old:-?} gone) -- taking over"
fi
printf '%s\n' "$$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# distinct --run-name tokens whose CUDA_VISIBLE_DEVICES is $1
arms_on_gpu() {
  local g="$1" p rn vis
  for p in $(pgrep -f 'evolve_kb_batch' 2>/dev/null || true); do
    [ -r "/proc/$p/cmdline" ] || continue
    rn="$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | grep -A1 -x -- '--run-name' | tail -1)"
    [ -n "$rn" ] || continue
    vis="$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
    [ "$vis" = "$g" ] && printf '%s\n' "$rn"
  done | sort -u | wc -l
}

run_dir_exists() {  # $1 = tag
  compgen -G "$REPO/$RESULTS_ROOT${RUN_PREFIX}_${1}_itr30_GH200_2*" >/dev/null 2>&1
}

launch_one() {  # $1 = gpu, $2 = spec line
  local gpu="$1" line="$2" tag spec rc
  tag="$(printf '%s' "$line" | cut -d'|' -f1 | tr -d '[:space:]')"
  spec="$(mktemp "${TMPDIR:-/tmp}/bf_${tag}_XXXX.spec")"
  printf '# auto-generated by wave_backfill.sh\n%s\n' "$line" > "$spec"
  log "LAUNCH $tag -> GPU $gpu"
  (
    cd "$REPO" || exit 1
    MODEL="$MODEL" RESULTS_ROOT="$RESULTS_ROOT" RUN_PREFIX="$RUN_PREFIX" \
    MAX_ARMS_PER_GPU="$MAX_PER_GPU" LAG_SEC=20 \
    MANIFEST="wave_gpu${gpu}_${RUN_PREFIX}_bf_${tag}.manifest.tsv" \
    bash "$HW/launch_wave.sh" "$gpu" "$spec"
  ) >> "$LOG" 2>&1
  rc=$?
  rm -f "$spec"
  # Pop the queue only on CONFIRMED evidence the run exists. A preflight FATAL
  # (GPU busy, sklearn gone) must leave the entry pending, not silently drop it.
  if run_dir_exists "$tag"; then
    printf '%s\n' "$tag" >> "$STATE"
    log "OK $tag launched on GPU $gpu (rc=$rc); run dir present"
    # Hand the arm to wave_supervisor.sh, which re-reads its spec every tick, so a
    # backfilled arm gets the same death-and-restart cover as a hand-launched one.
    # NEVER write a comment or blank line here: the supervisor's exit condition is
    # `total=$(wc -l < SPEC)`, which counts every line.
    if [ -n "$SUPERVISOR_SPEC" ] && [ -f "$SUPERVISOR_SPEC" ]; then
      local dir ctx flags
      dir="$(basename "$(ls -1d "$REPO/$RESULTS_ROOT${RUN_PREFIX}_${tag}_itr30_GH200_2"* 2>/dev/null | sort | tail -1)")"
      ctx="$(printf '%s' "$line" | cut -d'|' -f2 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
      flags="$(printf '%s' "$line" | cut -d'|' -f3- | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
      printf '%s\t%s\t%s\t1\t%s\t%s\n' "$gpu" "$dir" "$ctx" "$flags" "$RESULTS_ROOT" >> "$SUPERVISOR_SPEC"
      log "REGISTERED $dir -> $SUPERVISOR_SPEC (supervisor will restart it if it dies)"
    fi
    restart_monitor
    return 0
  fi
  log "FAILED $tag on GPU $gpu (rc=$rc); no run dir -- staying queued, retry next tick"
  return 1
}

log "=== wave_backfill start: queue=$QUEUE target=$TARGET_PER_GPU/gpu interval=${INTERVAL}s prefix=$RUN_PREFIX ==="

while :; do
  # pending = queue lines whose tag is not in STATE and has no run dir already
  mapfile -t pending < <(
    grep -v '^[[:space:]]*#' "$QUEUE" | grep -v '^[[:space:]]*$' | while IFS= read -r l; do
      t="$(printf '%s' "$l" | cut -d'|' -f1 | tr -d '[:space:]')"
      grep -qxF "$t" "$STATE" && continue
      run_dir_exists "$t" && { printf '%s\n' "$t" >> "$STATE"; continue; }
      printf '%s\n' "$l"
    done
  )

  if [ "${#pending[@]}" -eq 0 ]; then
    log "queue drained -- $(wc -l < "$STATE") launched. exiting."
    break
  fi

  # pick the emptiest GPU
  best_gpu=""; best_n=999
  for g in $GPUS; do
    n="$(arms_on_gpu "$g")"
    [ "$n" -lt "$best_n" ] && { best_n="$n"; best_gpu="$g"; }
  done

  if [ "$best_n" -ge "$TARGET_PER_GPU" ]; then
    counts=""; for g in $GPUS; do counts="$counts gpu$g=$(arms_on_gpu "$g")"; done
    log "hold:$counts (target $TARGET_PER_GPU) pending=${#pending[@]}"
  else
    launch_one "$best_gpu" "${pending[0]}"
    sleep 60   # let the new arm register before the next count
  fi

  sleep "$INTERVAL"
done
