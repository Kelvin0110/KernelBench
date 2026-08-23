#!/usr/bin/env python3
"""Sample wave health into a JSONL, one snapshot per interval.

Deliberately cheap and side-effect free: it reads /proc, nvidia-smi and line
counts only. It never writes into a run dir and never imports anything the eval
child imports, so it is safe to start and stop while arms are live.

WHY LINE COUNTS. The phase records emitted by src/kernelbench/eval.py carry no
timestamp, so a phase log alone cannot answer "what was the lock doing in the
last hour". Snapshotting each file's line count every interval turns the append
order into time windows: records [prev_count, cur_count) belong to that window.
Adding a timestamp field to eval.py would be the direct fix, but eval.py is
re-imported from disk by every eval spawn, and editing it under a live wave is
what killed the waves on 2026-08-20 and 2026-08-23.

    nohup python3 wave_collect.py --out wave_samples.jsonl --interval 300 &
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone

RUN_NAME_RE = re.compile(r"--run-name\s+(\S+)")


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def live_arms() -> dict:
    """run_name -> gpu, for processes that are actually alive right now.

    A killed arm leaves a complete-looking run dir behind, so any glob over run
    dirs silently mixes its pre-kill (higher contention) evals into what you
    think is the current state. The process list is the only source of truth.
    """
    arms = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        cmdline = _read(f"/proc/{pid}/cmdline").replace("\0", " ")
        if "evolve_kb_batch.py" not in cmdline:
            continue
        m = RUN_NAME_RE.search(cmdline)
        if not m:
            continue
        environ = _read(f"/proc/{pid}/environ")
        gpu = ""
        for kv in environ.split("\0"):
            if kv.startswith("CUDA_VISIBLE_DEVICES="):
                gpu = kv.split("=", 1)[1]
        # uv wrapper + python child share a run name; keep one entry
        arms[m.group(1)] = gpu
    return arms


def gpus() -> list:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            rows.append({"idx": int(p[0]), "mem_mib": int(p[1]),
                         "util_gpu": int(p[2]), "util_mem": int(p[3])})
    return rows


def host() -> dict:
    load = _read("/proc/loadavg").split()
    mem = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            mem[k] = int(v.strip().split()[0]) // 1024 // 1024
    return {"load1": float(load[0]) if load else -1.0,
            "load5": float(load[1]) if len(load) > 1 else -1.0,
            "mem_total_gb": mem.get("MemTotal", -1),
            "mem_avail_gb": mem.get("MemAvailable", -1),
            "cores": os.cpu_count()}


def counts(patterns: list) -> dict:
    out = {}
    for pat in patterns:
        for f in glob.glob(pat):
            n = 0
            try:
                with open(f, "rb") as fh:
                    for _ in fh:
                        n += 1
            except OSError:
                continue
            out[os.path.basename(f)] = n
    return out


def progress(results_roots: list, arms: dict) -> dict:
    """problems completed per LIVE arm, from batch_timing.jsonl."""
    out = {}
    for run in arms:
        for root in results_roots:
            # isdir matters: resume drops a <run>.preresume.<stamp>.tar.gz next to
            # the run dir, and a bare glob picks the tarball as the "run dir",
            # reporting 0 problems and 0 evals for a perfectly healthy arm.
            for d in sorted(p for p in glob.glob(os.path.join(root, run + "*"))
                            if os.path.isdir(p)):
                bt = os.path.join(d, "batch_timing.jsonl")
                n = 0
                try:
                    with open(bt) as fh:
                        n = sum(1 for _ in fh)
                except OSError:
                    pass
                evals = 0
                for f in glob.glob(os.path.join(d, "workspaces", "*",
                                                "evaluation_terminal_output.jsonl")):
                    try:
                        with open(f, "rb") as fh:
                            evals += sum(1 for _ in fh)
                    except OSError:
                        pass
                out[run] = {"problems_done": n, "evals": evals, "dir": os.path.basename(d)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="wave_samples.jsonl")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--phase-glob", default="*_phase.jsonl")
    ap.add_argument("--results-root", action="append",
                    default=["runs_evolving/gpt-oss-120b/median",
                             "runs_evolving/gpt-5.6-terra/median"])
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    while True:
        arms = live_arms()
        snap = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "arms": arms,
            "n_arms": len(arms),
            "arms_per_gpu": {g: sum(1 for v in arms.values() if v == g)
                             for g in sorted(set(arms.values()))},
            "gpus": gpus(),
            "host": host(),
            "phase_lines": counts([args.phase_glob]),
            "progress": progress(args.results_root, arms),
        }
        with open(args.out, "a") as fh:
            fh.write(json.dumps(snap) + "\n")
        if args.once or not arms:
            # no live arms -> nothing left to watch; exit so the job does not linger
            if not arms:
                break
            if args.once:
                break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
