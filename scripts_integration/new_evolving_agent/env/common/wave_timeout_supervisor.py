#!/usr/bin/env python3
"""Restart arms whose EVAL-TIMEOUT rate has gone bad, so they pick up a higher deadline.

WHY THIS EXISTS
---------------
`evaluation_timeout_s` bounds WORK ONLY: GPU-lock wait and memory-gate wait are
published by the eval child and discounted by the parent (execution.py:392), which
the failure text states outright -- "evaluation timeout after 600.0s (excluding 0s
GPU-lock wait)". CPU starvation is the one delay that cannot be discounted, because
it is indistinguishable from the work simply being slow, and KB_EVAL_HOIST_INPUT_GEN=1
put get_inputs() on the host CPU.

On 2026-09-02 the default was raised 600 -> 3600 (submodule 33c2ab4, repo c9bcde6).
That is PARENT-SIDE: KBGovernorConfig is built once at run start and held for the
arm's whole life, so a running arm keeps whatever it launched with. The only way an
arm adopts 3600 is to be stopped and resumed. This script does that, but only when
the loss is bad enough to be worth the churn.

A timed-out eval is not a neutral loss: it is recorded as `worker_error` and the
governor then "debugs" a kernel that was never broken -- the same phantom-failure
class that 7ac0e87 removed for lock waits.

SAFETY RULES (each one is here because getting it wrong corrupts a run)
----------------------------------------------------------------------
* RESUME INDEX. batch_timing.jsonl records FAILED problems too, with status="error"
  and fewer than max-iterations of work. So `last_index + 1` silently skips a
  half-done problem and keeps its truncated best-of-N as though it were a full run.
  Rule (CLAUDE.md 3.6): if the last record is not status=ok, restart AT that index.
  It is not a line count either -- a resume APPENDS, so the file outgrows the run.
* FLAGS. evolve_kb_batch.py rebuilds an arm's treatment from the CLI, and
  _check_resume_config_mismatch returns [] when run_summary.json is absent -- exactly
  the killed-arm case. Resuming without the flags silently continues a governance arm
  as plain truncation. We therefore derive flags from the arm's OWN /proc cmdline
  rather than a hand-maintained spec that can drift.
* MODEL. resume_run.sh defaults to gpt-oss-120b. wave_supervisor.sh guesses the model
  from the results-root string and has NO luna case, so it would resume a luna arm as
  gpt-oss -- wrong model AND wrong context window (1,050,000 vs 128,000). We read
  --model off the live process instead.
* ORPHANS. Killing an arm leaves its eval child holding 30-52 GB of GPU memory. We
  sweep only that arm's own descendants, never a blanket compute-apps kill, which
  would take out every other arm on the card.

Default is DRY-RUN. Pass --apply to actually restart.
"""
from __future__ import annotations

import argparse, glob, json, os, re, signal, subprocess, sys, time
from datetime import datetime, timezone

REPO = "/localhome/local-tianzheng/KernelBench"
HW = f"{REPO}/scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2"
STATE = f"{REPO}/.wave_timeout_supervisor_state.json"
EVAL_TO = re.compile(r"evaluation timeout after", re.I)

# args resume_run.sh supplies itself; everything else on the cmdline is treatment
_RESUME_OWNED_VAL = {
    "--run-name", "--results-root", "--max-problems", "--max-iterations",
    "--start-problem", "--end-problem", "--hardware", "--nvidia-endpoint",
    "--model", "--context-management", "--coder-timeout-sec", "--evaluation-timeout-sec",
}
_RESUME_OWNED_BARE = {"--resume", "--backup-l1-on-resume"}


def _say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def live_arms() -> dict[str, dict]:
    """Map run-name -> metadata, taken from the live process (never from a spec file)."""
    out: dict[str, dict] = {}
    for p in glob.glob("/proc/[0-9]*"):
        try:
            cmd = open(f"{p}/cmdline", "rb").read().decode(errors="replace").split("\0")
            cmd = [c for c in cmd if c]
            if not any("evolve_kb_batch.py" in c for c in cmd):
                continue
            pid = int(os.path.basename(p))
            env = dict(
                l.split("=", 1)
                for l in open(f"{p}/environ", "rb").read().decode(errors="replace").split("\0")
                if "=" in l
            )

            def arg(flag: str):
                try:
                    return cmd[cmd.index(flag) + 1]
                except Exception:
                    return None

            rn = arg("--run-name")
            if not rn:
                continue
            # Slice from the script path onward FIRST. The parent is launched as
            # `uv run --no-sync python .../evolve_kb_batch.py ...`, so scanning the whole
            # cmdline captures `--no-sync` (and swallows `python` as its value) and feeds
            # them back to the batch runner on resume, which breaks the restart.
            try:
                cmd = cmd[next(k for k, c in enumerate(cmd) if "evolve_kb_batch.py" in c) + 1:]
            except StopIteration:
                continue
            # treatment flags = whatever resume_run.sh does not supply itself
            flags, i = [], 0
            while i < len(cmd):
                a = cmd[i]
                if a in _RESUME_OWNED_VAL:
                    i += 2; continue
                if a in _RESUME_OWNED_BARE or not a.startswith("--"):
                    i += 1; continue
                flags.append(a)
                if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                    flags.append(cmd[i + 1]); i += 2
                else:
                    i += 1
            rec = dict(
                pid=pid, run=rn, gpu=env.get("CUDA_VISIBLE_DEVICES"),
                model=arg("--model"), root=arg("--results-root"),
                ctx=arg("--context-management"), flags=flags,
                already_raised=("--evaluation-timeout-sec" in cmd),
            )
            if rn not in out or pid < out[rn]["pid"]:
                out[rn] = rec
        except Exception:
            pass
    return out


def run_dir(a: dict) -> str | None:
    c = [d for d in glob.glob(os.path.join(a["root"] or "", a["run"] + "*")) if os.path.isdir(d)]
    return max(c, key=os.path.getmtime) if c else None


def timeout_stats(d: str, window: int) -> tuple[int, int, int]:
    """(timeouts_in_window, evals_in_window, problems_done) over the most recent evals."""
    recs = []
    for f in glob.glob(os.path.join(d, "workspaces", "*", "metrics_by_iteration.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            w = r.get("wall_time_utc")
            if not w:
                continue
            try:
                t = datetime.fromisoformat(str(w).replace("Z", "+00:00"))
            except Exception:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            m = r.get("metrics_iteration") or {}
            recs.append((t, bool(EVAL_TO.search(str(m.get("error") or "")))))
    recs.sort(key=lambda x: x[0])
    tail = recs[-window:]
    bt = os.path.join(d, "batch_timing.jsonl")
    done = 0
    if os.path.exists(bt):
        done = len({json.loads(l).get("subset_index") for l in open(bt) if l.strip()})
    return sum(1 for _, b in tail if b), len(tail), done


def resume_index(d: str) -> int:
    """CLAUDE.md 3.6: redo the problem unless it ended cleanly."""
    bt = os.path.join(d, "batch_timing.jsonl")
    try:
        last = None
        for line in open(bt):
            if line.strip():
                last = line
        r = json.loads(last or "{}")
        i = int(r.get("subset_index", 0) or 0)
        return max(1, i if str(r.get("status", "")) != "ok" else i + 1)
    except Exception:
        return 1


def descendants(pid: int) -> set[int]:
    kids: dict[int, list[int]] = {}
    for p in glob.glob("/proc/[0-9]*"):
        try:
            q = int(os.path.basename(p))
            pp = int(open(f"{p}/stat").read().split(") ", 1)[1].split()[1])
            kids.setdefault(pp, []).append(q)
        except Exception:
            pass
    seen, stack = set(), [pid]
    while stack:
        x = stack.pop()
        for k in kids.get(x, []):
            if k not in seen:
                seen.add(k); stack.append(k)
    return seen


def stop_arm(a: dict) -> None:
    """SIGTERM the arm, then sweep ONLY its own GPU-resident descendants."""
    doomed = descendants(a["pid"]) | {a["pid"]}
    for pid in sorted(doomed):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    for _ in range(30):
        if not os.path.exists(f"/proc/{a['pid']}"):
            break
        time.sleep(1)
    try:
        gpu_pids = {
            int(x.split(",")[0])
            for x in subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip().splitlines() if x.strip()
        }
    except Exception:
        gpu_pids = set()
    for pid in sorted(doomed & gpu_pids):          # never a blanket sweep
        _say(f"    sweeping orphaned GPU child pid={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=80, help="most-recent evals per arm to score")
    ap.add_argument("--min-evals", type=int, default=30, help="need this many before judging")
    ap.add_argument("--threshold-pct", type=float, default=8.0, help="restart above this timeout %%")
    ap.add_argument("--max-restarts", type=int, default=2, help="per arm, ever")
    ap.add_argument("--cooldown-h", type=float, default=8.0, help="min hours between restarts")
    ap.add_argument("--skip-past", type=int, default=47, help="leave arms this far along alone")
    ap.add_argument("--max-per-pass", type=int, default=1, help="never churn the whole wave at once")
    ap.add_argument("--new-timeout", type=int, default=3600)
    ap.add_argument("--apply", action="store_true", help="actually restart (default: dry-run)")
    a = ap.parse_args()

    st = {}
    if os.path.exists(STATE):
        try:
            st = json.load(open(STATE))
        except Exception:
            st = {}

    arms = live_arms()
    _say(f"scanning {len(arms)} live arms (window={a.window}, threshold={a.threshold_pct}%)")

    # report-only: a foreign GPU process means someone outside our lock is on the card
    try:
        mine = set()
        for r in arms.values():
            mine |= descendants(r["pid"]) | {r["pid"]}
        foreign = [
            x for x in subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip().splitlines()
            if x.strip() and int(x.split(",")[0]) not in mine
        ]
        if foreign:
            _say(f"  NOTE {len(foreign)} foreign GPU process(es) (outside KB_GPU_EVAL_LOCK): {foreign[:3]}")
    except Exception:
        pass

    acted = 0
    for rn, arm in sorted(arms.items()):
        d = run_dir(arm)
        if not d:
            continue
        to, n, done = timeout_stats(d, a.window)
        pct = 100.0 * to / n if n else 0.0
        tag = rn.replace("base_agent_", "").replace("_itr30_GH200", "")
        flag = ""
        if arm["already_raised"]:
            flag = " [already raised]"
        _say(f"  {tag:<34} {done:>2}/50  timeouts {to:>3}/{n:<3} = {pct:5.1f}%{flag}")

        if arm["already_raised"] or n < a.min_evals or pct < a.threshold_pct:
            continue
        if done >= a.skip_past:
            _say(f"    SKIP -- {done}/50, too near the end to churn"); continue
        s = st.get(rn, {"restarts": 0, "last": 0})
        if s["restarts"] >= a.max_restarts:
            _say(f"    SKIP -- already restarted {s['restarts']}x"); continue
        if time.time() - s["last"] < a.cooldown_h * 3600:
            _say(f"    SKIP -- cooldown"); continue
        if acted >= a.max_per_pass:
            _say(f"    DEFER -- max {a.max_per_pass} restart(s) per pass"); continue

        frm = resume_index(d)
        if frm > 50:
            _say(f"    SKIP -- nothing left"); continue
        cmd = [
            "bash", f"{HW}/resume_run.sh", str(arm["gpu"]), os.path.basename(d),
            str(arm["ctx"]), str(frm), "--",
            "--evaluation-timeout-sec", str(a.new_timeout), *arm["flags"],
        ]
        # resume_run.sh deliberately defaults to the SINGLE-ARM eval profile
        # (KB_GPU_EVAL_LOCK_SLOTS=1, KB_EVAL_MEM_GATE_FACTOR=0). Inheriting that inside a
        # live multi-arm wave is a correctness bug, not a slowdown:
        #   * slots=1 uses a DIFFERENT lock file from slots=N (gpu_lock.py:159-164 appends
        #     .slotK), so the restarted arm would NOT interlock with the others on its GPU
        #     -- its timing windows could overlap theirs, and contention deflates speedup.
        #   * mem gate 0 removes device-memory admission control, which is what keeps three
        #     ~49 GiB L1P34 residents off a 143 GiB card.
        # All of these are ${VAR:-default} in resume_run.sh, so exporting them wins.
        env = dict(os.environ, RESULTS_ROOT=arm["root"] or "", MODEL=arm["model"] or "",
                   DO_BACKUP="0", MAX_ARMS_PER_GPU="12",
                   KB_GPU_EVAL_LOCK_SLOTS="3", KB_EVAL_MEM_GATE_FACTOR="7",
                   KB_EVAL_HOIST_INPUT_GEN="1", KB_EVAL_SKIP_DEAD_REF_TIMING="1",
                   KB_EVAL_UNLOCK_CORRECTNESS="0", KB_GPU_RESERVE_GB="0")
        _say(f"    TRIGGER {pct:.1f}% >= {a.threshold_pct}%  -> stop and resume at {frm}")
        _say(f"      model={arm['model']} root={arm['root']} flags={arm['flags'] or ['none']}")
        if not a.apply:
            _say(f"      DRY-RUN, would run: {' '.join(cmd)}")
            continue
        stop_arm(arm)
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        ok = r.returncode == 0
        _say(f"      resume {'OK' if ok else 'FAILED rc=' + str(r.returncode)}")
        if not ok:
            _say(f"      stderr: {(r.stderr or '')[-400:]}")
        st[rn] = {"restarts": s["restarts"] + 1, "last": time.time()}
        json.dump(st, open(STATE, "w"), indent=2)
        acted += 1
    _say(f"done; {acted} restart(s) this pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
