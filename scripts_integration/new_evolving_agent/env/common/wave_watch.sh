#!/usr/bin/env bash
# Durable health watchdog for a running wave. Meant to be launched detached and
# left alone for the life of the run:
#
#   cd /localhome/local-tianzheng/KernelBench
#   setsid nohup bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/wave_watch.sh \
#     >/dev/null 2>&1 < /dev/null &
#
#   tail -f wave_watch.log          # heartbeat, one line per poll
#   grep ALARM wave_watch.log       # only the things that need a human
#
# Env: WATCH_INTERVAL_SEC (900)  EXPECT_ARMS (9)  TIMEOUT_RATE_ALARM (2.0)
#
# Why a script rather than an in-session monitor: a session-scoped watcher dies
# with the session, and a 50x30 wave runs for ~2 days.
#
# What it will NOT do: declare the run dead because its own tooling broke. An
# empty read from wave_status.sh previously made ${n:-0} evaluate to 0 and
# tripped an "ALL ARMS FINISHED OR DIED" branch on a perfectly healthy 9-arm
# run, purely because the script had been moved. Every zero-arm reading is now
# cross-checked against pgrep before it is believed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"

STATUS="$HERE/wave_status.sh"
LOG="$REPO_ROOT/wave_watch.log"
INTERVAL="${WATCH_INTERVAL_SEC:-900}"
EXPECT="${EXPECT_ARMS:-9}"
RATE_ALARM="${TIMEOUT_RATE_ALARM:-2.0}"

[ -f "$STATUS" ] || { echo "FATAL: no wave_status.sh at $STATUS" | tee -a "$LOG"; exit 1; }

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

# Ground truth, independent of the status tool: count real arm processes.
real_arms() {
  local c=0 p a0
  for p in $(pgrep -f "evolve_kb_batch.py" 2>/dev/null || true); do
    [ -r "/proc/$p/cmdline" ] || continue
    a0="$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)"
    [ "$(basename "${a0:-x}")" = "uv" ] && continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "evolve_kb_batch.py" || continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--run-name" || continue
    c=$((c + 1))
  done
  echo "$c"
}

timeout_stats() {
  ./.venv/bin/python - <<'PY' 2>/dev/null || echo "0 0 0"
import json, glob
ev = to = 0
# Both layouts: the flat legacy root AND runs_evolving/<model>/<series>/...
# The old single hardcoded glob missed every median-series run, so the eval
# timeout rate printed 0/0 no matter what was happening.
_pats = ('runs_evolving/*/*/workspaces/*/evaluation_terminal_output.jsonl',
         'runs_evolving/*/*/*/workspaces/*/evaluation_terminal_output.jsonl')
for f in sorted({p for _pat in _pats for p in glob.glob(_pat)}):
    try:
        fh = open(f)
    except OSError:
        continue
    with fh:
        for l in fh:
            if not l.strip():
                continue
            ev += 1
            try:
                if 'evaluation timeout after' in str(json.loads(l).get('terminal_output', '')):
                    to += 1
            except Exception:
                pass
print(ev, to, round(100 * to / max(ev, 1), 2))
PY
}

say "watchdog start: expect=$EXPECT interval=${INTERVAL}s status=$STATUS"
last_n="$EXPECT"; alarmed_rate=0; last_bucket=-1

while true; do
  line="$(bash "$STATUS" check 2>/dev/null)"
  n="$(echo "$line"    | sed -n 's/.*ARMS=\([0-9]*\).*/\1/p')"
  orph="$(echo "$line" | sed -n 's/.*ORPH=\([0-9-]*\).*/\1/p')"
  unl="$(echo "$line"  | sed -n 's/.*UNLOCK=\([0-9]*\).*/\1/p')"
  cud="$(echo "$line"  | sed -n 's/.*CUDAH=\([0-9]*\).*/\1/p')"
  lmax="$(echo "$line" | sed -n 's/.*LOCKMAX=\([0-9]*\).*/\1/p')"

  if [ -z "$line" ] || [ -z "$n" ]; then
    say "ALARM tooling: wave_status.sh returned nothing (arms per pgrep: $(real_arms)). NOT an arm failure."
    sleep "$INTERVAL"; continue
  fi

  read -r ev to rate <<<"$(timeout_stats)"

  [ "${orph:-0}" -gt 0 ] && say "ALARM orphaned waits=$orph -- evals killed mid-wait; those numbers are suspect"
  [ "${unl:-0}" -gt 0 ] && say "ALARM proceeded UNLOCKED=$unl -- contended timing, speedups deflated"
  [ "${cud:-0}" -gt 0 ] && say "ALARM CUDA_HOME hits=$cud -- nvcc may be missing; kernels can silently fall back to PyTorch"

  if [ "$(awk -v r="${rate:-0}" -v t="$RATE_ALARM" 'BEGIN{print (r>t)?1:0}')" = "1" ] && [ "$alarmed_rate" = "0" ]; then
    say "ALARM eval-timeout rate ${rate}% (${to}/${ev}) above ${RATE_ALARM}%"
    alarmed_rate=1
  fi

  if [ "${n:-0}" -lt "$last_n" ]; then
    rn="$(real_arms)"
    if [ "$rn" -lt "$last_n" ]; then
      say "ALARM arm count dropped: $rn/$EXPECT (pgrep-confirmed) -- names: $(echo "$line" | sed -n 's/.*NAMES=//p')"
      last_n="$rn"
    else
      say "status tool undercounted ($n) but pgrep sees $rn -- ignoring"
    fi
  fi

  minp="$(bash "$STATUS" 2>/dev/null | awk '$2 ~ /^[0-9]+$/ {gsub(/[()]/,"",$3); split($3,a,"/"); if(a[1]!="" && (m==""||a[1]<m)) m=a[1]} END{print m+0}')"
  say "ok arms=${n}/${EXPECT} slowest=${minp}/50 timeouts=${to}/${ev} (${rate}%) lockmax=${lmax}s orph=${orph} unlock=${unl}"

  bucket=$(( ${minp:-0} / 5 ))
  if [ "$bucket" -gt "$last_bucket" ]; then last_bucket="$bucket"; fi

  if [ "${n:-0}" -eq 0 ]; then
    rn="$(real_arms)"
    if [ "$rn" -eq 0 ]; then
      say "wave COMPLETE (or all arms gone): pgrep confirms 0 arms. timeouts ${to}/${ev} (${rate}%)"
      exit 0
    fi
    say "status tool reported 0 arms but pgrep sees $rn -- continuing"
  fi
  sleep "$INTERVAL"
done
