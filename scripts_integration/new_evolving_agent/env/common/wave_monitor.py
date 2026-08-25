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
            arms.append({
                "tag": tag,
                "pid": int(pid) if pid.isdigit() else -1,
                "rundir": rundir if os.path.isabs(rundir) else os.path.join(REPO, rundir),
                "log": log if os.path.isabs(log) else os.path.join(REPO, log),
                "name": os.path.basename(rundir.rstrip("/")),
            })
    return arms


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


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
    st["alive"] = alive(arm["pid"])
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

    offsets, prev, first, cycle, prev_api = {}, {}, True, 0, [0]
    emit("START", "watching %d arms, interval %ds" % (len(arms), args.interval))

    while True:
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
                if api > prev_api[0]:
                    emit("WARN", "LLM-side failures rose %d -> %d "
                         "(coder_call_error / content-policy / rate-limit; "
                         "counted from metrics_iteration.error)" % (prev_api[0], api))
                prev_api[0] = api
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
