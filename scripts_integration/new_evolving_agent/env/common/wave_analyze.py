#!/usr/bin/env python3
"""Read wave_collect samples + phase logs and say what the bottleneck is.

    python3 wave_analyze.py --samples wave_samples.jsonl --window-h 7

Answers one question: can this host take more arms per GPU, and if so how many.

The parallelism ceiling is whichever of four resources saturates first:

  lock   sum(held) / (wall * slots)   -- the GPU eval lock
  cpu    load1 / cores                -- unlocked nvcc/ninja, which N arms do at once
  gpu    utilization.gpu              -- actual device work
  llm    inferred from the residual   -- if evals are cheap and arms are still slow,
                                         the agent is waiting on the API and adding
                                         arms is free until the endpoint throttles

Report the headroom of each; the smallest one is the ceiling. Adding arms beyond
it converts throughput into queueing, which is what the pre-2026-08-23 mutex did.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from datetime import datetime, timedelta, timezone


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def load_samples(path):
    out = []
    try:
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="wave_samples.jsonl")
    ap.add_argument("--phase-glob", default="*_phase.jsonl")
    ap.add_argument("--window-h", type=float, default=7.0)
    ap.add_argument("--target-util", type=float, default=0.60,
                    help="utilisation to size the recommendation against")
    args = ap.parse_args()

    samples = load_samples(args.samples)
    if len(samples) < 2:
        print(f"need >=2 samples in {args.samples} (have {len(samples)}); "
              "let the collector run longer")
        return

    now = datetime.fromisoformat(samples[-1]["ts"])
    lo = now - timedelta(hours=args.window_h)
    win = [s for s in samples if datetime.fromisoformat(s["ts"]) >= lo]
    if len(win) < 2:
        win = samples
    t0 = datetime.fromisoformat(win[0]["ts"])
    t1 = datetime.fromisoformat(win[-1]["ts"])
    wall_h = (t1 - t0).total_seconds() / 3600
    if wall_h <= 0:
        print("zero-length window")
        return

    first, last = win[0], win[-1]
    print(f"=== window {t0:%Y-%m-%d %H:%M} -> {t1:%H:%M} UTC  ({wall_h:.2f} h, "
          f"{len(win)} samples) ===\n")

    # ---- throughput ----------------------------------------------------
    arms_per_gpu = last["arms_per_gpu"]
    n_arms = last["n_arms"]
    d_ev = d_pr = 0
    for run, cur in last["progress"].items():
        prev = first["progress"].get(run)
        if prev:
            d_ev += max(0, cur["evals"] - prev["evals"])
            d_pr += max(0, cur["problems_done"] - prev["problems_done"])
    print(f"arms: {n_arms} total, per GPU {arms_per_gpu}")
    print(f"throughput: {d_ev} evals ({d_ev/wall_h:.1f}/h aggregate, "
          f"{d_ev/wall_h/max(n_arms,1):.2f}/h per arm), {d_pr} problems completed")
    if d_ev:
        print(f"            -> {n_arms*wall_h*60/max(d_ev,1)*1:.1f} arm-min per eval; "
              f"a solo gpt-oss arm historically does ~23 evals/h (1500 evals / 63-65 h)")

    # ---- lock, from phase records in this window -----------------------
    # Phase records carry no timestamp; the collector's line counts bound them.
    lock_util_by_gpu = {}
    recs_win = []
    for f in glob.glob(args.phase_glob):
        b = os.path.basename(f)
        start = first["phase_lines"].get(b, 0)
        end = last["phase_lines"].get(b, 0)
        if end <= start:
            continue
        try:
            lines = open(f).readlines()[start:end]
        except OSError:
            continue
        for ln in lines:
            try:
                recs_win.append((b, json.loads(ln)))
            except json.JSONDecodeError:
                pass

    if not recs_win:
        print("\nno phase records in window -- is KB_EVAL_PHASE_LOG set on the arms?")
    else:
        held = [r["held_sec"] for _, r in recs_win]
        wait = [r["waited_sec"] for _, r in recs_win]
        slots = max((r.get("lock_slots", 1) for _, r in recs_win), default=1)
        n_gpu = max(1, len(arms_per_gpu))
        # held seconds are spread across n_gpu independent locks
        util = sum(held) / (wall_h * 3600 * slots * n_gpu)
        lock_util_by_gpu["all"] = util
        print(f"\nlock ({len(recs_win)} evals, slots={slots}, {n_gpu} GPU locks):")
        print(f"  held  median {st.median(held):6.2f}s  p90 {pct(held,.9):7.2f}s  "
              f"max {max(held):8.2f}s")
        print(f"  wait  median {st.median(wait):6.2f}s  p90 {pct(wait,.9):7.2f}s  "
              f"max {max(wait):8.2f}s   >5s: {sum(1 for x in wait if x>5)}/{len(wait)}")
        print(f"  UTILISATION {util:.1%} of capacity (sum held / wall x slots x gpus)")
        ph = {}
        for _, r in recs_win:
            for k, v in r.get("phases", {}).items():
                ph.setdefault(k, []).append(v)
        inside = {k: st.median(v) for k, v in ph.items() if st.median(v) > 0.005}
        print(f"  phase medians: {json.dumps({k: round(v,2) for k,v in inside.items()})}")
        # unlocked_correctness is deliberately OFF: it removes the bound on how
        # many evals are device-resident and OOM'd the box on 2026-08-23. The
        # intended config is hoisted + skipped ref window + slots>1.
        bad = [b for b, r in recs_win
               if not (r.get("hoisted") and r.get("lock_slots", 1) > 1)]
        if bad:
            print(f"  !! {len(bad)} eval(s) NOT fully optimised -- check the arm env")

    # ---- host / gpu ----------------------------------------------------
    cores = last["host"]["cores"] or 1
    load = [s["host"]["load1"] for s in win]
    cpu_util = st.median(load) / cores
    print(f"\nhost: load1 median {st.median(load):.1f} / {cores} cores = {cpu_util:.1%}"
          f"   mem avail {last['host']['mem_avail_gb']} GB / {last['host']['mem_total_gb']} GB")
    for idx in sorted({g["idx"] for s in win for g in s["gpus"]}):
        u = [g["util_gpu"] for s in win for g in s["gpus"] if g["idx"] == idx]
        m = [g["mem_mib"] for s in win for g in s["gpus"] if g["idx"] == idx]
        print(f"gpu{idx}: util median {st.median(u):.0f}%  p90 {pct(u,.9):.0f}%   "
              f"mem median {st.median(m)/1024:.1f} GB  max {max(m)/1024:.1f} GB")

    # ---- verdict --------------------------------------------------------
    print("\n=== headroom ===")
    limits = {}
    if lock_util_by_gpu:
        limits["lock"] = lock_util_by_gpu["all"]
    limits["cpu"] = cpu_util
    gpu_utils = [st.median([g["util_gpu"] for s in win for g in s["gpus"] if g["idx"] == i]) / 100
                 for i in sorted({g["idx"] for s in win for g in s["gpus"]})]
    if gpu_utils:
        limits["gpu"] = max(gpu_utils)
    mem_frac = 1 - (last["host"]["mem_avail_gb"] / max(last["host"]["mem_total_gb"], 1))
    limits["host_mem"] = mem_frac

    per_gpu_now = max(arms_per_gpu.values()) if arms_per_gpu else 0
    for k, v in sorted(limits.items(), key=lambda kv: -kv[1]):
        scale = (args.target_util / v) if v > 0 else float("inf")
        cap = per_gpu_now * scale
        print(f"  {k:9s} {v:6.1%} used  -> supports ~{cap:.0f} arms/GPU at "
              f"{args.target_util:.0%} target" if v > 0 else
              f"  {k:9s} {v:6.1%} used  -> no measurable pressure")
    binding = max(limits.items(), key=lambda kv: kv[1])
    rec = per_gpu_now * (args.target_util / binding[1]) if binding[1] > 0 else per_gpu_now * 4
    print(f"\nbinding resource: {binding[0]} at {binding[1]:.1%}")
    print(f"currently {per_gpu_now} arms/GPU -> recommendation: "
          f"{'HOLD' if rec < per_gpu_now * 1.25 else f'raise to ~{min(rec, per_gpu_now*3):.0f} arms/GPU'}")
    print("\nCaveats: LLM-endpoint latency is not measured here and is the one limit that")
    print("degrades silently -- check the `coder` phase gap before and after any increase.")
    print("Problems 1-5 carry 74% of the benchmark's input-generation cost, so a window")
    print("that straddles them is not steady state.")


if __name__ == "__main__":
    main()
