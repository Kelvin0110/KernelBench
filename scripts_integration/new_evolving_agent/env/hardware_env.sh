#!/usr/bin/env bash
# Shared hardware/baseline resolution for the launcher scripts, so the same
# scripts work on a different server by pointing $HARDWARE at that host's folder
# under results/timing/.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/hardware_env.sh"
#   kb_resolve_hardware            # sets $HARDWARE, defaulting per $KB_DEFAULT_HARDWARE
#   kb_require_hardware            # preflight: folder exists, baseline carries median
#
# Env:
#   HARDWARE               folder under results/timing/ (default NVIDIA_GH200x2_2nd)
#   ALLOW_MEAN_BASELINE=1  downgrade the missing-median check from fatal to warning
#
# Why the median check is fatal by default: get_timing_stats() started recording
# a median in fix(eval) 6a3e972, and runtime_from_stats() prefers it. A baseline
# JSON written before that has no median, so the candidate's median is divided by
# the baseline's mean. On the GH200 50-problem subset that shifts 25 of 50
# problems by >5% and inflates 26_GELU_/22_Tanh roughly 4x -- a silent metric
# error, not a crash, which is exactly the kind this repo has been bitten by.

# Default hardware = the name of the directory the CALLING script lives in.
# Server-specific launchers sit in env/<HARDWARE>/, so copying that folder for a
# new machine and renaming it is all that is needed -- no edit to any script.
# Falls back to NVIDIA_GH200x2_2nd when sourced from somewhere else.
_kb_caller_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" 2>/dev/null && pwd)"
_kb_caller_name="$(basename "${_kb_caller_dir:-}")"
if [ -n "$_kb_caller_name" ] && [ -d "$(dirname "$(dirname "$(dirname "$_kb_caller_dir")")")/../results/timing/$_kb_caller_name" ] 2>/dev/null; then
  KB_DEFAULT_HARDWARE="${KB_DEFAULT_HARDWARE:-$_kb_caller_name}"
fi
KB_DEFAULT_HARDWARE="${KB_DEFAULT_HARDWARE:-NVIDIA_GH200x2_2nd}"

kb_resolve_hardware() {
  HARDWARE="${HARDWARE:-$KB_DEFAULT_HARDWARE}"
  export HARDWARE
}

kb_require_hardware() {
  local repo="${1:-$PWD}"
  kb_resolve_hardware
  if [ ! -d "$repo/results/timing/$HARDWARE" ]; then
    echo "FATAL: missing results/timing/$HARDWARE"
    echo "       set HARDWARE=<folder> to one of:"
    ls -1 "$repo/results/timing" 2>/dev/null | grep -v '\.md$' | sed 's/^/         /'
    exit 1
  fi

  local py="$repo/.venv/bin/python"
  [ -x "$py" ] || py="python3"
  "$py" - "$repo" "$HARDWARE" "${ALLOW_MEAN_BASELINE:-0}" <<'PY' || exit 1
import json, sys
repo, hw, allow = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
path = f"{repo}/results/timing/{hw}/baseline_time_torch.json"
try:
    b = json.load(open(path))
except Exception as exc:
    print(f"FATAL: cannot read {path}: {exc}")
    sys.exit(1)
entry = next(iter(next(iter(b.values())).values()))
if "median" in entry:
    print(f"  baseline {hw}: median present (OK)")
    sys.exit(0)
msg = (f"baseline {hw} has no 'median' field (written before fix(eval) 6a3e972).\n"
       "         Speedup would compare candidate median against baseline mean.\n"
       "         Regenerate it, or set ALLOW_MEAN_BASELINE=1 to accept the skew.")
if allow:
    print(f"  WARNING: {msg}")
    sys.exit(0)
print(f"FATAL: {msg}")
sys.exit(1)
PY
}
