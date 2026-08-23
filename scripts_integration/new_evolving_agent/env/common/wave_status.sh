#!/usr/bin/env bash
# One-shot health view of every live evolving-agent arm, across both GPUs.
#
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/wave_status.sh
#
# Columns:
#   PROBLEM   per-problem progress from the [batch] lines
#   ITERS     governor iterations recorded so far
#   LOCKMAX   worst GPU-eval-lock wait observed (s)
#   ORPHWAIT  waits with no matching acquire/UNLOCKED -- MUST be 0. Non-zero
#             means an eval was killed while queued and the governor recorded a
#             compile failure for a kernel that was never broken (CLAUDE.md 3.4).
#   UNLOCK    evals that gave up waiting and ran contended -- MUST be 0; those
#             speedups are deflated.
#   CUDAH     'CUDA_HOME' hits in the log -- MUST be 0, else nvcc is missing and
#             kernels silently fall back to plain PyTorch while scoring correct.

set -uo pipefail

MODE="${1:-table}"   # table | check  ("check" prints one parseable summary line)

# grep -c prints "0" AND exits 1 when there are no matches, so `|| echo 0`
# emits two lines and breaks the arithmetic below. Count through this instead.
cnt() { local n; n="$(grep -c "$1" "$2" 2>/dev/null | head -1)"; echo "${n:-0}"; }

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

T_ARMS=0; T_ORPH=0; T_UNLOCK=0; T_CUDAH=0; T_TMOUT=0; T_LMAX=0; NAMES=""
if [ "$MODE" != "check" ]; then
  printf '%-34s %-4s %-9s %-6s %-8s %-9s %-7s %-6s %-6s\n' \
    ARM GPU PROBLEM ITERS LOCKMAX ORPHWAIT UNLOCK TMOUT CUDAH
fi

found=0
for p in $(pgrep -f "evolve_kb_batch" 2>/dev/null || true); do
  env_file="/proc/$p/environ"
  [ -r "$env_file" ] || continue
  vis="$(tr '\0' '\n' < "$env_file" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
  [ -n "$vis" ] || continue
  name="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -oE '\-\-run-name [^ ]+' | awk '{print $2}')"
  [ -n "$name" ] || continue
  # Two processes per arm: the `uv run` wrapper and the python child it execs.
  # Both mention evolve_kb_batch.py, so discriminate on argv[0] and keep the
  # child -- otherwise every arm is listed twice.
  argv0="$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)"
  [ "$(basename "${argv0:-}")" = "uv" ] && continue
  tr '\0' ' ' < "/proc/$p/cmdline" | grep -q "evolve_kb_batch.py" || continue

  log="$(ls -1t ${name}_*.log 2>/dev/null | head -1)"
  if [ -z "$log" ]; then
    printf '%-34s %-4s %-9s %-6s %-8s %-9s %-7s %-6s\n' "${name:0:34}" "$vis" "?" "?" "?" "?" "?" "?"
    found=1; continue
  fi
  prob="$(grep -E "^\[batch\] \([0-9]+/" "$log" 2>/dev/null | tail -1 | grep -oE '\([0-9]+/[0-9]+\)')"
  iters="$(cnt "^\[kb-governor\] iter=" "$log")"

  # The lock/timeout messages are printed by the eval CHILD, whose stdout
  # eval_runner.run_kernelbench_eval captures via redirect_stdout into
  # terminal_output. They never reach the arm log -- grepping it (as CLAUDE.md
  # 3.4 still says to) reports clean no matter what happens. Read the jsonl.
  # Honour RESULTS_ROOT, else search BOTH layouts: the flat legacy root and the
  # per-series subdir (runs_evolving/<model>/median/...). The old glob hardcoded
  # runs_evolving/gpt-oss-120b/ with no median/ segment, so every median-series
  # run resolved to "" and LOCKMAX/ORPHWAIT/UNLOCK/TMOUT all printed 0 -- falsely
  # clean, for exactly the runs we care about.
  rundir="$(ls -1dt ${RESULTS_ROOT:-runs_evolving}/${name}_2* \
                    runs_evolving/*/${name}_2* \
                    runs_evolving/*/*/${name}_2* 2>/dev/null | head -1)"
  read -r w a u lmax nto ch <<<"$(LOCK_RUNDIR="$rundir" ./.venv/bin/python - <<'PYEOF'
import json, glob, os, re
d = os.environ.get("LOCK_RUNDIR", "")
w = a = u = nto = ch = 0
mx = 0.0
if d:
    for f in glob.glob(os.path.join(d, "workspaces", "*", "evaluation_terminal_output.jsonl")):
        try:
            fh = open(f)
        except OSError:
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    t = str(json.loads(line).get("terminal_output", ""))
                except Exception:
                    continue
                # Orphan = this eval waited but never acquired or gave up:
                # the signature of a child killed mid-wait. Counting raw
                # "waiting" minus "acquired" lines instead goes NEGATIVE, because
                # gpu_lock prints "waiting" only when a poll observes wait>=5s but
                # prints "acquired after" from the final elapsed -- a ~5.1s wait
                # can acquire before any poll crosses the threshold.
                hw = "waiting for another eval" in t
                ha = "acquired after" in t
                hu = "proceeding UNLOCKED" in t
                if hw and not (ha or hu):
                    w += 1
                u += t.count("proceeding UNLOCKED")
                nto += t.count("evaluation timeout after")
                ch += t.count("CUDA_HOME")
                for m in re.finditer(r"acquired after ([0-9.]+)s", t):
                    mx = max(mx, float(m.group(1)))
print(w, a, u, f"{mx:.0f}", nto, ch)
PYEOF
)"
  w=${w:-0}; a=${a:-0}; u=${u:-0}; lmax=${lmax:-0}; nto=${nto:-0}; ch=${ch:-0}
  ch=$(( ch + $(cnt "CUDA_HOME" "$log") ))
  # The baseline arm carries no tag, so stripping the prefix leaves "itr30_GH200".
  short="$(echo "$name" | sed 's/base_agent_gpt_oss_120b_//;s/_itr30_GH200//;s/^itr30_GH200$/truncation/')"
  [ -z "$short" ] && short=truncation
  T_ARMS=$((T_ARMS+1)); T_ORPH=$((T_ORPH + w)); T_UNLOCK=$((T_UNLOCK+u)); T_CUDAH=$((T_CUDAH+ch)); T_TMOUT=$((T_TMOUT+nto))
  [ "${lmax:-0}" -gt "$T_LMAX" ] 2>/dev/null && T_LMAX=${lmax:-0}
  NAMES="$NAMES$short,"
  if [ "$MODE" != "check" ]; then
    printf '%-34s %-4s %-9s %-6s %-8s %-9s %-7s %-6s %-6s\n' \
      "${short:0:34}" "$vis" "${prob:-init}" "$iters" "${lmax:-0}s" "$w" "$u" "$nto" "$ch"
  fi
  found=1
done

if [ "$MODE" = "check" ]; then
  echo "ARMS=$T_ARMS ORPH=$T_ORPH UNLOCK=$T_UNLOCK TMOUT=$T_TMOUT LOCKMAX=$T_LMAX CUDAH=$T_CUDAH NAMES=${NAMES%,}"
  exit 0
fi

[ "$found" -eq 1 ] || echo "(no live arms)"
echo
echo "ORPHWAIT / UNLOCK / CUDAH must all read 0. Any non-zero means the affected"
echo "arm's numbers are suspect -- investigate before using it in the analysis."
