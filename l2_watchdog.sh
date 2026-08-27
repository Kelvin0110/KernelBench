#!/usr/bin/env bash
# Detached health watchdog for the GPU0 L2 wave.
#
# Runs independently of any SSH session or Claude Code monitor: start it with
#   setsid nohup bash l2_watchdog.sh >/dev/null 2>&1 &
# so it becomes its own session leader with no controlling terminal, exactly
# like the arms themselves (PPID=1, TTY=?).
#
# Deliberately does NOT reuse env/common/wave_watch.sh: that globs run
# directories without intersecting the live process list, so a killed arm's
# directory keeps contributing to its numbers (CLAUDE.md 3.5). Everything here
# is keyed off `ps` first.
#
# Appends one block per interval to l2_watchdog.log and shouts on:
#   - an arm disappearing
#   - CUDA OOM appearing in any arm log
#   - mem-gate waits that hit the timeout (charged against the eval deadline)
#   - `proceeding UNLOCKED` (contended evals)
#   - GPU memory above a threshold
set -u
cd "$(dirname "$0")"

LOG=l2_watchdog.log
INTERVAL="${INTERVAL:-900}"        # 15 min
MEM_WARN_PCT="${MEM_WARN_PCT:-92}"
TOTAL_GIB=142.75

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

# `grep -c` prints 0 AND exits 1 when there is no match, so `grep -c x || echo 0`
# emits TWO lines and any arithmetic on it is a syntax error. Same family as the
# pgrep -c trap in CLAUDE.local.md. Always funnel counts through this.
count_in() {  # count_in <pattern> <file>
  local n
  n=$(grep -ci -- "$1" "$2" 2>/dev/null | head -1)
  printf '%s' "${n:-0}"
}

say "watchdog start pid=$$ interval=${INTERVAL}s"
prev_arms=""

while true; do
  # --- live arms, from ps (never from directory globs) ---
  arms=$(ps -eo cmd= | grep -oP '(?<=--run-name )base_agent_gpt_oss\S+' | sort -u)
  n=$(echo "$arms" | grep -c . || true)

  if [ -n "$prev_arms" ] && [ "$arms" != "$prev_arms" ]; then
    gone=$(comm -23 <(echo "$prev_arms") <(echo "$arms") | tr '\n' ' ')
    [ -n "${gone// /}" ] && say "ALERT arm(s) DISAPPEARED: $gone"
    new=$(comm -13 <(echo "$prev_arms") <(echo "$arms") | tr '\n' ' ')
    [ -n "${new// /}" ] && say "arm(s) appeared: $new"
  fi
  prev_arms="$arms"

  # --- GPU memory ---
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
  pct=$(awk -v u="$used" -v t="$TOTAL_GIB" 'BEGIN{printf "%.1f", u/1024/t*100}')
  over=$(awk -v p="$pct" -v w="$MEM_WARN_PCT" 'BEGIN{print (p>w)?1:0}')
  [ "$over" = "1" ] && say "ALERT GPU0 memory ${pct}% (${used} MiB)"

  # --- OOM and contention, from the arm logs of LIVE arms only ---
  oom=0; unlocked=0
  for a in $arms; do
    f=$(ls -t "${a}"_*wave.log 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    oom=$(( oom + $(count_in "CUDA out of memory" "$f") ))
    unlocked=$(( unlocked + $(count_in "proceeding UNLOCKED" "$f") ))
  done
  [ "$oom" -gt 0 ] && say "ALERT CUDA OOM occurrences in live arm logs: $oom"
  [ "$unlocked" -gt 0 ] && say "ALERT proceeding UNLOCKED (contended evals): $unlocked"

  # --- mem-gate timeouts: this wait is charged against the 600s eval deadline ---
  gate_to=$(cat base_agent_gpt_oss_120b_*_phase.jsonl 2>/dev/null \
    | grep -o '"mem_gate_waited_sec": *[0-9.]*' \
    | awk -F: '{ if ($2+0 >= 600) c++ } END { print c+0 }')

  # --- progress ---
  prog=""
  for a in $arms; do
    d=$(ls -dt runs_evolving/gpt-oss-120b/*/"${a}"_* 2>/dev/null | head -1)
    [ -z "$d" ] && continue
    # An arm that has not finished problem 1 has no batch_timing.jsonl yet. The
    # `<` redirect fails in the SHELL before wc runs, so a 2>/dev/null on wc does
    # not suppress it -- test for the file instead.
    if [ -f "$d/batch_timing.jsonl" ]; then p=$(wc -l < "$d/batch_timing.jsonl"); else p=0; fi
    short=${a#base_agent_gpt_oss_120b_}; short=${short%_itr30_GH200}
    prog="$prog ${short}=${p}"
  done

  say "arms=$n mem=${pct}% gate_timeouts=${gate_to} oom=${oom} unlocked=${unlocked} |$prog"

  # --- L2 tier state: promotions, judge decisions, pre-seed survival ---
  # A zero here is otherwise silent (governor.py swallows promotion-pass
  # exceptions into one line), and a pre-seeded rule set being demoted to empty
  # would quietly gut the three arms that depend on it.
  l2line=""
  for a in $arms; do
    d=$(ls -dt runs_evolving/gpt-oss-120b/*/"${a}"_* 2>/dev/null | head -1)
    [ -z "$d" ] && continue
    [ -f "$d/l2_standing.jsonl" ] || continue
    short=${a#base_agent_gpt_oss_120b_}; short=${short%_itr30_GH200}
    st=$(wc -l < "$d/l2_standing.jsonl")
    pr=0; ps=0
    if [ -f "$d/l2_promotions.jsonl" ]; then
      pr=$(count_in '"event": "promote"' "$d/l2_promotions.jsonl")
      ps=$(count_in '"event": "preseed"' "$d/l2_promotions.jsonl")
    fi
    # A pre-seeded arm whose standing set has fallen to zero has been demoted.
    if [ "$ps" -gt 0 ] && [ "$st" -eq 0 ]; then
      say "ALERT $short was pre-seeded with $ps rule(s) but standing is now 0 -- demoted"
    fi
    l2line="$l2line ${short}=st${st}/pr${pr}"
  done
  [ -n "$l2line" ] && say "  L2:$l2line"

  jd=$(cat base_agent_gpt_oss_120b_l2_judge_*wave.log 2>/dev/null | grep -c "\[l2\] judge" | head -1)
  jf=$(cat base_agent_gpt_oss_120b_l2_judge_*wave.log 2>/dev/null | grep -c "judge FAILED\|judge SKIPPED" | head -1)
  [ "${jf:-0}" -gt 0 ] && say "ALERT judge failed/skipped ${jf} time(s) -- it fails closed, so this looks like 'rejected everything'"
  say "  judge: decisions=${jd:-0} failures=${jf:-0}"

  # Stop when every arm is gone (wave finished or was killed).
  if [ "$n" -eq 0 ]; then
    say "no arms left; watchdog exiting"
    exit 0
  fi
  sleep "$INTERVAL"
done
