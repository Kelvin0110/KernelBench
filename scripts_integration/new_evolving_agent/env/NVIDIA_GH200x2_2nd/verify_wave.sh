#!/usr/bin/env bash
# Verify every live arm's ACTUAL command line against the wave spec files.
#
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/verify_wave.sh <spec> [<spec>...]
#
# run_summary.json is only written when a batch finishes, so for the first three
# days the process cmdline is the only authoritative record of what an arm is
# actually running. A silently wrong --hardware or a dropped governance flag is
# invisible in the logs and only shows up as a quietly wrong number at the end.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

declare -A WANT_CTX WANT_FLAGS
for spec in "$@"; do
  [ -f "$spec" ] || { echo "no such spec: $spec"; exit 1; }
  while IFS= read -r line; do
    line="${line%%#*}"; [ -z "${line// /}" ] && continue
    tag="$(echo "${line%%|*}" | xargs)"; rest="${line#*|}"
    ctx="$(echo "${rest%%|*}" | xargs)"; flags=""
    case "$rest" in *\|*) flags="$(echo "${rest#*|}" | xargs)" ;; esac
    if [ "$tag" = "-" ] || [ -z "$tag" ]; then n="base_agent_gpt_oss_120b_itr30_GH200"
    else n="base_agent_gpt_oss_120b_${tag}_itr30_GH200"; fi
    WANT_CTX["$n"]="$ctx"; WANT_FLAGS["$n"]="$flags"
  done < "$spec"
done

echo "expected arms: ${#WANT_CTX[@]}"
echo
bad=0; seen=0
for p in $(pgrep -f "evolve_kb_batch.py" 2>/dev/null || true); do
  [ -r "/proc/$p/cmdline" ] || continue
  a0="$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)"
  [ "$(basename "${a0:-x}")" = "uv" ] && continue
  cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)"
  case "$cmd" in *evolve_kb_batch.py*) ;; *) continue ;; esac
  name="$(echo "$cmd" | grep -oE '\-\-run-name [^ ]+' | awk '{print $2}')"
  [ -n "${WANT_CTX[$name]+x}" ] || { echo "UNEXPECTED arm running: $name"; bad=1; continue; }
  seen=$((seen+1))

  ctx="$(echo "$cmd" | grep -oE '\-\-context-management [^ ]+' | awk '{print $2}')"
  hw="$(echo "$cmd" | grep -oE '\-\-hardware [^ ]+' | awk '{print $2}')"
  probs="$(echo "$cmd" | grep -oE '\-\-max-problems [0-9]+' | awk '{print $2}')"
  iters="$(echo "$cmd" | grep -oE '\-\-max-iterations [0-9]+' | awk '{print $2}')"

  errs=""
  [ "$ctx" = "${WANT_CTX[$name]}" ] || errs="$errs ctx($ctx!=${WANT_CTX[$name]})"
  [ "$hw" = "${HARDWARE:-NVIDIA_GH200x2_2nd}" ] || errs="$errs hw($hw)"
  [ "$probs" = "${MAX_PROBLEMS:-50}" ] || errs="$errs problems($probs)"
  [ "$iters" = "${MAX_ITERATIONS:-30}" ] || errs="$errs iters($iters)"
  for f in ${WANT_FLAGS[$name]}; do
    case "$f" in --*) case "$cmd" in *"$f"*) ;; *) errs="$errs missing($f)" ;; esac ;; esac
  done

  short="$(echo "$name" | sed 's/base_agent_gpt_oss_120b_//;s/_itr30_GH200//;s/^itr30_GH200$/truncation/')"
  if [ -n "$errs" ]; then printf '  %-16s MISMATCH:%s\n' "$short" "$errs"; bad=1
  else printf '  %-16s OK  ctx=%-20s %s\n' "$short" "$ctx" "${WANT_FLAGS[$name]}"; fi
done

echo
missing=$(( ${#WANT_CTX[@]} - seen ))
echo "live: $seen / ${#WANT_CTX[@]}   (not yet up: $missing)"
[ "$bad" -eq 0 ] && echo "VERDICT: every live arm matches its spec." \
                 || { echo "VERDICT: MISMATCH -- fix before letting these run 3 days."; exit 1; }
