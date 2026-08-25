#!/usr/bin/env python3
"""Per-arm health monitor for a live wave.

Emits an event line only when something CHANGES, so the log stays signal:

    DONE  <arm>   -- run_summary.json appeared; that arm finished cleanly
    ALERT <arm>   -- process vanished WITHOUT run_summary.json (died)
    WARN  <arm>   -- new errors appeared (OOM / API / traceback / unlocked eval)
    PROG  <arm>   -- advanced to a new problem
    WAVE COMPLETE -- every arm reached run_summary.json

Two traps this deliberately avoids, both documented in CLAUDE.md 3.4:

* **Never glob run directories.** A killed arm leaves a live-looking directory
  whose pre-kill records keep contributing to any glob-based rollup. Arms come
  from the wave manifests and are intersected with the live process list.
* **Lock/eval errors never reach the arm log.** eval_runner wraps
  eval_kernel_against_ref in redirect_stdout(), so "out of memory",
  "proceeding UNLOCKED" and friends land in the per-iteration
  evaluation_terminal_output.jsonl, never in the arm's own log. Grepping the arm
  log for them reads vacuously clean regardless of reality.

Scanning is incremental -- byte offsets per file -- so cost does not grow with
the wave's length.

    python3 wave_monitor.py [--interval 300] [--once] [--log FILE]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
for _ in range(4):
    REPO = os.path.dirname(REPO)

# Errors are scanned from two places with DIFFERENT pattern sets, because the two
# streams mean different things.
#
# The arm log is the runner talking: a traceback or an API error there is a real
# problem with the wave. The eval record is mostly the CANDIDATE KERNEL talking --
# eval_runner captures the child's stdout, so a failed kernel's own compile
# traceback and its "CUDA error" land there on every bad attempt. Those are the
# agent working as designed, not wave ill-health, and counting them produced 77
# "tracebacks" on a control arm that finished perfectly. Only harness-level
# failures are scanned there.
LOG_PATTERNS = [
    ("fatal", re.compile(r"^FATAL|CUDA_HOME", re.M)),
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("api_error", re.compile(r"RateLimitError|APIStatusError|APITimeoutError|"
                             r"ContentPolicyViolation|InternalServerError|"
                             r"429 Too Many|502 Bad Gateway", re.I)),
    ("coder_error", re.compile(r"coder_call_error")),
]
EVAL_PATTERNS = [
    ("cuda_oom", re.compile(r"CUDA out of memory|CUDA error: out of memory", re.I)),
    ("unlocked_eval", re.compile(r"proceeding UNLOCKED")),
    ("mem_gate_timeout", re.compile(r"memory gate: waited")),
    ("worker_error", re.compile(r"worker_error")),
]
# torch inductor autotuner noise -- benign, explicitly NOT an error (CLAUDE.md 3.5)
BENIGN = re.compile(r"No valid triton configs|OutOfMemoryError: triton_mm")

# LLM-side failures are recorded ONLY in metrics_iteration["error"] -- they reach
# neither the arm log nor the eval terminal output, so they are invisible to both
# pattern sets above and have to be counted from the metrics stream.
API_ERR_RE = re.compile(r"coder_call_error|ContentPolicyViolation|RateLimitError|"
                        r"InternalServerError|APITimeoutError|APIStatusError", re.I)

PROBLEM_RE = re.compile(r"^\[batch\] \((\d+)/(\d+)\)", re.M)

# The mem-gate valve, mirrored from src/kernelbench/eval.py's floor under
# wait_reporting_active(). Raised 1800 -> 3600 on 2026-08-25 after the valve fired
# for real (r2_markov waited 1800s for 49 GiB and proceeded UNGATED). Keep in step
# with eval.py: a stale value here silently mis-scores both the valve-hit count and
# the headroom alarm.
MEM_GATE_VALVE_SEC = 3600.0


def load_arms(pattern):
    """(tag, pid, rundir, log) for every arm in the matching manifests."""
    arms = []
    for man in sorted(glob.glob(os.path.join(REPO, pattern))):
        for line in open(man):
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] == "idx":
                continue
            _, tag, pid, rundir, log = p[:5]
            if rundir in ("", "?"):
                continue
            name = os.path.basename(rundir.rstrip("/"))
            arms.append({
                "tag": tag,
                "pid": int(pid) if pid.isdigit() else -1,
                "rundir": rundir if os.path.isabs(rundir) else os.path.join(REPO, rundir),
                "log": log if os.path.isabs(log) else os.path.join(REPO, log),
                "name": name,
                # RUN_NAME is the dir name minus the _YYYY_MM_DD_HH_MM the runner
                # appends; it is what appears after --run-name in the cmdline.
                "run_name": STAMP_RE.sub("", name),
                "manifest": man,
            })
    return arms


def manifest_stamps(pattern):
    """mtime per manifest, so a relaunch UNDER a running monitor is detectable.

    launch_wave.sh names the manifest from GPU + RUN_PREFIX + UTC day, so killing
    a wave and relaunching it the same day REWRITES the same file with new pids
    and new run dirs. A monitor started before that keeps watching the dead wave
    and would emit an ALERT per arm. Detect it and say so instead.
    """
    return {m: os.path.getmtime(m)
            for m in sorted(glob.glob(os.path.join(REPO, pattern)))}


STAMP_RE = re.compile(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}$")


def _cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            return fh.read().decode("utf-8", "replace").split("\0")
    except (OSError, ValueError):
        return []


def alive(arm):
    """True only if the pid exists AND is still THIS arm.

    A bare os.kill(pid, 0) is not enough after a wave is killed and relaunched:
    pids are recycled, so an unrelated process inheriting the number reads as a
    healthy arm forever. Match the run name in the cmdline instead.
    """
    pid = arm["pid"]
    if pid <= 0:
        return False
    argv = _cmdline(pid)
    if argv:
        return arm["run_name"] in argv
    try:
        os.kill(pid, 0)          # /proc unreadable -- fall back
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def arm_config(arm):
    """(hardware, results_root) as the arm was actually launched.

    Read from the live cmdline, never from a launcher or a spec: this project's
    recurring failure is a baseline that differs from the one assumed, and it is
    a silent metric error rather than a crash.
    """
    argv = _cmdline(arm["pid"])
    out = {}
    for flag in ("--hardware", "--results-root", "--model"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                out[flag] = argv[i + 1]
    return out


def scan_new(path, offsets):
    """Bytes appended to `path` since the last call (empty if unchanged)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    off = offsets.get(path, 0)
    if size <= off:
        if size < off:
            offsets[path] = 0
            off = 0
        else:
            return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(off)
            chunk = fh.read()
    except OSError:
        return ""
    offsets[path] = off + len(chunk)
    return chunk.decode("utf-8", "replace")


def count_errors(text, patterns):
    if not text:
        return {}
    out = {}
    for label, pat in patterns:
        n = 0
        for m in pat.finditer(text):
            window = text[max(0, m.start() - 120):m.end() + 120]
            if not BENIGN.search(window):
                n += 1
        if n:
            out[label] = n
    return out


def arm_state(arm, offsets):
    rundir, st = arm["rundir"], {}
    st["done"] = os.path.exists(os.path.join(rundir, "run_summary.json"))
    st["alive"] = alive(arm)
    st["errors"] = count_errors(scan_new(arm["log"], offsets), LOG_PATTERNS)

    try:
        with open(arm["log"], "r", errors="replace") as fh:
            m = PROBLEM_RE.findall(fh.read())
        st["problem"] = int(m[-1][0]) if m else 0
        st["total"] = int(m[-1][1]) if m else 0
    except OSError:
        st["problem"], st["total"] = 0, 0

    evals = 0
    for f in glob.glob(os.path.join(rundir, "workspaces", "*",
                                    "evaluation_terminal_output.jsonl")):
        chunk = scan_new(f, offsets)
        if not chunk:
            continue
        for line in chunk.splitlines():
            if not line.strip():
                continue
            evals += 1
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            for k, v in count_errors(str(rec.get("terminal_output") or ""), EVAL_PATTERNS).items():
                st["errors"][k] = st["errors"].get(k, 0) + v
    st["new_evals"] = evals
    return st


def wave_health(arms):
    """Compiled / correct / hack rates across the wave.

    Fields live under `metrics_iteration`, NOT at the record root -- reading them
    flat silently yields 0 compiled / 0 correct on a perfectly healthy wave.
    """
    n = comp = corr = hack = api = 0
    best = 0.0
    for arm in arms:
        for f in glob.glob(os.path.join(arm["rundir"], "workspaces", "*",
                                        "metrics_by_iteration.jsonl")):
            try:
                fh = open(f, errors="replace")
            except OSError:
                continue
            with fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        m = (json.loads(line).get("metrics_iteration") or {})
                    except ValueError:
                        continue
                    if not m:
                        continue
                    n += 1
                    comp += bool(m.get("compiled"))
                    corr += bool(m.get("correct"))
                    hack += bool(m.get("is_hack"))
                    if m.get("correct") and m.get("speedup"):
                        best = max(best, float(m["speedup"]))
                    if API_ERR_RE.search(str(m.get("error") or "")):
                        api += 1
    return n, comp, corr, hack, best, api


def gate_health(arms):
    """Regression detectors for the 2026-08-25 mem-gate fix.

    Three things were wrong and are now fixed; each has a counter here so a
    regression is visible instead of silent:
      * the gate's wait was billed to the 600s eval deadline -> eval timeouts
      * the timeout branch print()ed into the coder's prompt -> contamination
      * on timeout the gate proceeds UNGATED -> OOM risk on the big problems
    Phase logs are matched to arms by run name, never globbed blindly.
    """
    waited = to = contam = oom = 0
    waits = []
    names = {a["run_name"] for a in arms}
    for f in glob.glob(os.path.join(REPO, "*_phase.jsonl")):
        base = os.path.basename(f)
        if not any(base.startswith(n + "_") for n in names):
            continue
        try:
            fh = open(f, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                w = float(r.get("mem_gate_waited_sec") or 0)
                if w > 0.05:
                    waited += 1
                    waits.append(w)
                if w >= MEM_GATE_VALVE_SEC - 1:   # proceeded UNGATED
                    to += 1
    for arm in arms:
        for f in glob.glob(os.path.join(arm["rundir"], "workspaces", "*",
                                        "evaluation_terminal_output.jsonl")):
            try:
                fh = open(f, errors="replace")
            except OSError:
                continue
            with fh:
                for line in fh:
                    if "memory gate: waited" in line:
                        contam += 1
                    if "out of memory" in line.lower() and "triton_mm" not in line:
                        oom += 1
    return waited, to, contam, oom, waits


def eval_timeout_rate(arms):
    """Iterations killed by the eval deadline, split by CAUSE.

    A raised timeout rate is NOT by itself a gate regression, and saying so was
    actively misleading in the log. Every timeout message carries
    "(excluding Ns GPU-lock wait)" when a wait was published and discounted, so:

      * excluded > 0  -> the deadline WAS extended and the eval still overran, i.e.
        it genuinely needed that much WORK. Expected on the big problems, where a
        slow candidate kernel x 25 timing trials can exceed 600s on its own.
      * excluded == 0 with a long gate wait -> the wait never reached the parent.
        THAT is the regression, and it is the only thing worth alarming on.

    Returns (iterations, timeouts, timeouts_without_an_excluded_wait, max_excluded).
    """
    n = t = 0
    bare = [0]
    max_excl = [0]
    for arm in arms:
        for f in glob.glob(os.path.join(arm["rundir"], "workspaces", "*",
                                        "metrics_by_iteration.jsonl")):
            try:
                fh = open(f, errors="replace")
            except OSError:
                continue
            with fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        m = (json.loads(line).get("metrics_iteration") or {})
                    except ValueError:
                        continue
                    if not m:
                        continue
                    n += 1
                    err = str(m.get("error") or "")
                    if "timeout" in err.lower():
                        t += 1
                        hit = re.search(r"excluding (\d+)s", err)
                        if hit:
                            max_excl[0] = max(max_excl[0], int(hit.group(1)))
                        else:
                            bare[0] += 1
    return n, t, bare[0], max_excl[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log", default=os.path.join(REPO, "wave_monitor.log"))
    ap.add_argument("--manifests", default="wave_gpu*.manifest.tsv",
                    help="glob for the manifests to watch, relative to the repo root. "
                         "Scope this to the CURRENT wave -- a finished wave's manifest "
                         "is still on disk and would be watched forever.")
    ap.add_argument("--health-every", type=int, default=12,
                    help="emit a HEALTH rollup every N cycles (0 = never)")
    args = ap.parse_args()

    # Singleton per log file. Two daemons appending to one log interleave their
    # events, so every DONE / ALERT / GATE line is emitted twice -- which reads as
    # two arms finishing, or an alert storm, and silently doubles any rate the
    # operator eyeballs from the log. Keyed on the LOG path, so a second monitor
    # watching a DIFFERENT wave is still allowed. flock is released by the kernel
    # on process death, so a crashed monitor never wedges the next one out.
    lock_path = args.log + ".lock"
    try:
        import fcntl

        _lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                other = os.read(_lock_fd, 32).decode().strip() or "?"
            except OSError:
                other = "?"
            print("FATAL: another wave_monitor is already writing %s (pid %s). "
                  "Stop it first, or pass a different --log." % (args.log, other),
                  file=sys.stderr)
            return 1
        os.ftruncate(_lock_fd, 0)
        os.write(_lock_fd, str(os.getpid()).encode())
    except ImportError:  # non-POSIX: degrade to the old unguarded behaviour
        _lock_fd = None

    out = open(args.log, "a", buffering=1)

    def emit(kind, msg):
        line = "%s  %-5s %s" % (
            time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()), kind, msg)
        print(line, file=out, flush=True)
        print(line, flush=True)

    arms = load_arms(args.manifests)
    if not arms:
        emit("ALERT", "no arms found matching %s -- nothing to watch" % args.manifests)
        return 1

    offsets, prev, first, cycle, prev_api = {}, {}, True, 0, [None]
    base_gate = [None]
    stamps = manifest_stamps(args.manifests)
    emit("START", "watching %d arms from %d manifest(s), interval %ds"
         % (len(arms), len(stamps), args.interval))

    # Record what the arms are ACTUALLY scored against, and refuse to be quiet if
    # they disagree. A split baseline within one wave makes arm-vs-arm comparison
    # meaningless, and it is invisible in every downstream artifact.
    cfgs = {a["name"]: arm_config(a) for a in arms}
    for flag, label in (("--hardware", "baseline"), ("--results-root", "results-root"),
                        ("--model", "model")):
        vals = {}
        for name, c in cfgs.items():
            vals.setdefault(c.get(flag, "?"), []).append(name)
        if len(vals) == 1:
            emit("INFO", "%s: %s (uniform across %d arms)"
                 % (label, next(iter(vals)), len(arms)))
        else:
            emit("ALERT", "%s IS NOT UNIFORM -- %s. Arm-vs-arm comparison across "
                 "this wave is INVALID until resolved."
                 % (label, "; ".join("%s=%d arm(s) %s" % (v, len(n), sorted(n)[:3])
                                     for v, n in sorted(vals.items()))))

    while True:
        now = manifest_stamps(args.manifests)
        if not first and now != stamps:
            emit("ALERT", "MANIFEST REWRITTEN -- the wave was relaunched under this "
                 "monitor, so it is still watching the OLD pids and run dirs and "
                 "every arm will read as dead. Restart the monitor. (%s)"
                 % ", ".join(sorted(os.path.basename(k) for k in
                                    set(now) ^ set(stamps)
                                    or [k for k in now if now[k] != stamps.get(k)])))
            return 1
        tot = {"done": 0, "alive": 0, "err": 0}
        for arm in arms:
            key, st = arm["name"], arm_state(arm, offsets)
            p = prev.get(key, {})
            tot["done"] += st["done"]
            tot["alive"] += st["alive"]
            tot["err"] += sum(st["errors"].values())

            if st["done"] and not p.get("done"):
                emit("DONE", "%s  finished %d/%d (run_summary.json written)"
                     % (key, st["problem"], st["total"]))
            elif not st["alive"] and not st["done"] and p.get("alive", True):
                emit("ALERT", "%s  PROCESS GONE at problem %d/%d with NO "
                     "run_summary.json -- arm died, pid %d"
                     % (key, st["problem"], st["total"], arm["pid"]))
            if st["errors"]:
                detail = ", ".join("%s=%d" % kv for kv in sorted(st["errors"].items()))
                emit("WARN", "%s  %s errors: %s"
                     % (key, "existing" if first else "NEW", detail))
            if st["problem"] > p.get("problem", 0) and not first:
                emit("PROG", "%s  problem %d/%d" % (key, st["problem"], st["total"]))
            prev[key] = st

        if first:
            emit("INFO", "baseline: %d/%d alive, %d done, %d pre-existing errors"
                 % (tot["alive"], len(arms), tot["done"], tot["err"]))
        if tot["done"] == len(arms):
            emit("DONE", "WAVE COMPLETE -- all %d arms wrote run_summary.json" % len(arms))
            return 0
        if not tot["alive"] and tot["done"] < len(arms):
            emit("ALERT", "WAVE STALLED -- 0 arms alive but only %d/%d finished"
                 % (tot["done"], len(arms)))
            return 1

        if args.health_every and (cycle % args.health_every == 0 or first or args.once):
            n, comp, corr, hack, best, api = wave_health(arms)
            if n:
                emit("HEALTH", "%d/%d alive, %d done | %d iters, compiled %d%%, "
                     "correct %d%%, hacks %d, llm_errors %d, best %.2fx"
                     % (tot["alive"], len(arms), tot["done"], n,
                        round(100 * comp / n), round(100 * corr / n), hack, api, best))
                if prev_api[0] is None:
                    # seed on the first rollup; otherwise pre-existing failures
                    # always look like a jump from zero
                    if api:
                        emit("INFO", "%d pre-existing LLM-side failure(s) at start"
                             % api)
                elif api > prev_api[0]:
                    emit("WARN", "LLM-side failures rose %d -> %d "
                         "(coder_call_error / content-policy / rate-limit; "
                         "counted from metrics_iteration.error)" % (prev_api[0], api))
                prev_api[0] = api
                gw, gto, gcon, goom, gwaits = gate_health(arms)
                ni, nto, bare, max_excl = eval_timeout_rate(arms)
                # These counters are CUMULATIVE over the wave, so a wave that was
                # already damaged before the monitor started would latch every
                # warning on forever and hide a fresh regression. Baseline at the
                # first rollup and judge only what has happened SINCE.
                if base_gate[0] is None:
                    base_gate[0] = (gcon, goom, ni, nto, bare)
                    emit("GATE", "baseline: waited %d, valve-hit %d, prompt-contam %d, "
                         "OOM %d, eval-timeouts %d/%d (%.1f%%, %d bare) -- all PRE-EXISTING; "
                         "warnings below judge growth from here"
                         % (gw, gto, gcon, goom, nto, ni,
                            100.0 * nto / ni if ni else 0.0, bare))
                else:
                    b_con, b_oom, b_ni, b_nto, b_bare = base_gate[0]
                    d_ni, d_nto = ni - b_ni, nto - b_nto
                    rate = 100.0 * d_nto / d_ni if d_ni else 0.0
                    emit("GATE", "since start: +%d iters, +%d eval-timeouts (%.1f%%), "
                         "+%d prompt-contam, +%d OOM, +%d bare | cumulative waited %d, valve-hit %d%s"
                         % (d_ni, d_nto, rate, gcon - b_con, goom - b_oom,
                            bare - b_bare, gw, gto,
                            "  (max wait %.0fs)" % max(gwaits) if gwaits else ""))
                    # A bare timeout is NOT on its own a regression. execution.py only
                    # appends "(excluding Ns GPU-lock wait)" when a wait was actually
                    # published, so an eval that queued for NOTHING and simply needed
                    # >600s of work is legitimately bare -- and most evals queue for
                    # nothing (755 of 1174 on this wave). Alarming on the first one
                    # cried wolf at 22:01Z. If publishing had truly broken, essentially
                    # EVERY timeout taken while queueing would be bare, so require the
                    # bare share to dominate before calling it a regression.
                    d_bare = bare - b_bare
                    d_tmo = nto - b_nto
                    bare_share = (100.0 * d_bare / d_tmo) if d_tmo else 0.0
                    if d_bare >= 3 and bare_share > 50.0:
                        emit("ALERT", "+%d of +%d eval timeouts (%.0f%%) carry NO excluded "
                             "wait since start. The mem-gate wait is no longer reaching "
                             "the parent via gpu_lock.report_external_wait, so queueing is "
                             "being billed to the 600s work budget -- the 2026-08-25 fix "
                             "has regressed." % (d_bare, d_tmo, bare_share))
                    elif d_bare:
                        emit("INFO", "+%d bare eval timeout(s) of +%d (%.0f%%) -- below the "
                             "regression threshold (>=3 and >50%%). Evals that queued for "
                             "nothing get no exclusion suffix, so a few bare timeouts are "
                             "just slow kernels." % (d_bare, d_tmo, bare_share))
                    elif rate > 2.0 and d_ni > 200:
                        emit("INFO", "eval-timeout rate %.1f%% since start, but 0 of them "
                             "lack an excluded wait (max excluded %ds, so the deadline "
                             "stretched to %ds). These evals needed >600s of WORK -- "
                             "expected on the big problems, NOT a gate regression."
                             % (rate, max_excl, 600 + max_excl))
                    if gwaits:
                        head = MEM_GATE_VALVE_SEC - max(gwaits)
                        if head < 600:
                            emit("ALERT", "max gate wait %.0fs is within %.0fs of the "
                                 "%.0fs valve. At the valve an eval proceeds UNGATED -- "
                                 "3 x 49GiB residents on L1P34 is ~147GiB on a 143GiB "
                                 "card, i.e. the OOM path."
                                 % (max(gwaits), head, MEM_GATE_VALVE_SEC))
                    if goom > b_oom:
                        emit("ALERT", "+%d CUDA OOM since start -- recorded compiled=True "
                             "correct=False, so the governor will debug kernels that were "
                             "never broken. Lower KB_GPU_EVAL_LOCK_SLOTS or raise "
                             "KB_EVAL_MEM_GATE_FACTOR." % (goom - b_oom))
                    if gcon > b_con:
                        emit("ALERT", "+%d evals injected mem-gate text into the coder "
                             "prompt since start. eval stdout AND stderr are spliced into "
                             "KERNEL_BENCH_EVAL_TERMINAL_OUTPUT -- this MUTATES LLM INPUT "
                             "and means the 2026-08-25 fix regressed." % (gcon - b_con))
                if comp and round(100 * comp / n) < 60:
                    emit("WARN", "compile rate %d%% is abnormally low -- check nvcc "
                         "and CUDA_HOME on this host" % round(100 * comp / n))
        cycle += 1
        first = False
        if args.once:
            emit("INFO", "%d/%d alive, %d done" % (tot["alive"], len(arms), tot["done"]))
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
