#!/usr/bin/env python3
"""Re-score every arm's headline metrics under ONE uniform excessive-speedup threshold.

WHY THIS EXISTS
---------------
`src/kernelbench/eval.py` changed `excessive_speedup_threshold` from 10 -> 30 at
2026-08-24T15:11:45 (commit 588a6a5). eval.py is re-imported by every eval spawn
(`execution.py` uses start_method="spawn"), so the change reached all live arms
mid-wave without a restart and without a line in any log. Evals before that instant
flagged `is_hack=True` for any speedup > 10; evals after, only for > 30.

`is_hack` gates which iterations may form a "best", so the same kernel with the same
speedup scores differently depending on *when* it ran. Arms are staggered across
problems, so the seam does not fall in the same place for every arm -- it is a
systematic, one-directional distortion of arm-vs-arm comparison.

WHY IT IS SAFE TO RE-DERIVE OFFLINE
-----------------------------------
The recorded `extra.speedup` IS the quantity the threshold was applied to. Verified
on this wave: predicting the stored `is_hack` from `speedup > threshold-in-force-at-
that-timestamp` agrees on 10388/10388 correct evals (100.0000%, zero mismatches).
So re-scoring is exact, not approximate, and needs no re-evaluation.

The 109 hack-flagged evals that are NOT speedup-driven (static-check rejections) need
no special handling: every one is `correct=False`, so it can never form a best.

WHAT IT CANNOT FIX
------------------
Only the *reported metric*. At run time `is_hack=True` also vetoed `is_new_best`, so
a suppressed kernel was never banked and the agent kept debugging from a worse state.
Re-scoring recovers the number, not the search trajectory.

METRIC DEFINITIONS -- matched to src/kernelbench/{score,performance_stats}.py:
  fast_p@p   = |{problems: correct and speedup > p}| / n_attempted   (strict >,
               failed problems stay in the denominator)
  geomean    = geometric mean over problems that are correct AND speedup > 0
               (failed problems EXCLUDED from the denominator)

Usage:
  python3 rescore_hack_threshold.py --threshold 30 --output-dir <dir> [--all-dirs]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

SEAM_UTC = "2026-08-24T15:11:45"   # commit 588a6a5 -- threshold 10 -> 30
OLD_T, NEW_T = 10.0, 30.0
DEFAULT_ROOTS = ["runs_evolving/gpt-oss-120b/median", "runs_evolving/gpt-5.6-terra/median"]
THRESHOLDS = [0.0, 1.0, 2.0]


def live_arms() -> set:
    """Run names of processes alive right now.

    A killed arm leaves a complete-looking run dir behind, so a bare glob mixes its
    pre-kill evals into what looks like current state. /proc is the only truth.
    """
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
        except OSError:
            continue
        if "evolve_kb_batch.py" not in cmd:
            continue
        m = re.search(r"--run-name\s+(\S+)", cmd)
        if m:
            out.add(m.group(1))
    return out


def geometric_mean(xs: list) -> float:
    xs = [x for x in xs if x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# A run_summary.json is written ONLY after the last problem (evolve_kb_batch.py), so
# its presence plus a full batch_timing.jsonl is the artifact-level proof that an arm
# actually finished -- as opposed to a killed arm, which leaves a complete-looking dir.
# Hardcoding "PARTIAL" was wrong once the wave completed: a stale warning on a final
# result is worse than no warning, because it invites discounting a valid comparison.
PROTOCOL_PROBLEMS = 50


def _incomplete(arms: dict) -> list:
    bad = []
    for name, a in arms.items():
        d = a.get("dir") or ""
        bt = os.path.join(d, "batch_timing.jsonl")
        n = sum(1 for _ in open(bt)) if os.path.isfile(bt) else 0
        if n < PROTOCOL_PROBLEMS or not os.path.isfile(os.path.join(d, "run_summary.json")):
            bad.append(f"{name} ({n}/{PROTOCOL_PROBLEMS})")
    return sorted(bad)


def _status_line(arms: dict) -> str:
    bad = _incomplete(arms)
    if bad:
        return ("PARTIAL -- " + ", ".join(bad) + "; ANALYSIS_RULES.md:158 forbids partial "
                "prefixes as a final comparative result. Magnitude only.")
    return (f"COMPLETE -- all {len(arms)} arms finished {PROTOCOL_PROBLEMS}/{PROTOCOL_PROBLEMS} "
            "with run_summary.json. Levels are quotable; n=1 replicate per cell, so per "
            "ANALYSIS_RULES/open-item-9 this still cannot support an arm-vs-arm winner claim.")


def scan_arm(run_dir: str, threshold: float) -> dict:
    """problem -> {best_stored, best_uniform, n_evals, n_flipped}."""
    probs = {}
    ws = os.path.join(run_dir, "workspaces")
    if not os.path.isdir(ws):
        return probs
    for name in sorted(os.listdir(ws)):
        f = os.path.join(ws, name, "evaluation_terminal_output.jsonl")
        if not os.path.isfile(f):
            continue
        bs = bu = 0.0
        n = flipped = 0
        try:
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                e = r.get("extra") or {}
                n += 1
                if not e.get("correct"):
                    continue
                s = e.get("speedup")
                if s is None:
                    continue
                s = float(s)
                stored_hack = bool(e.get("is_hack"))
                uniform_hack = s > threshold
                if stored_hack != uniform_hack:
                    flipped += 1
                if not stored_hack:
                    bs = max(bs, s)
                if not uniform_hack:
                    bu = max(bu, s)
        except OSError:
            continue
        if n:
            probs[name] = {"best_stored": bs, "best_uniform": bu,
                           "n_evals": n, "n_flipped": flipped}
    return probs


def metrics(bests: list) -> dict:
    """bests: one best-speedup per problem (0.0 => no correct non-hack kernel)."""
    n = len(bests)
    out = {"n_problems": n}
    for p in THRESHOLDS:
        out[f"fast_p_best@{p}"] = (sum(1 for b in bests if b > p) / n) if n else 0.0
    correct = [b for b in bests if b > 0]
    out["n_correct"] = len(correct)
    out["best_geomean"] = geometric_mean(correct)
    out["best_median"] = sorted(correct)[len(correct) // 2] if correct else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=NEW_T)
    ap.add_argument("--runs-root", action="append", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--completed-only", action="store_true",
                    help="keep only arms with run_summary.json AND >=50 batch_timing "
                         "entries. Neither --all-dirs (everything, incl. in-flight) nor "
                         "the live-process default gives you 'the finished arms' once a "
                         "new wave is running -- an arm at problem 2 would collapse the "
                         "aligned set to near zero and silently shrink every denominator.")
    ap.add_argument("--all-dirs", action="store_true",
                    help="do NOT intersect with the live process list (includes killed arms)")
    args = ap.parse_args()
    roots = args.runs_root or DEFAULT_ROOTS

    live = live_arms()
    if not args.all_dirs and not live:
        print("no live arms found; pass --all-dirs to score run dirs regardless", file=sys.stderr)
        return 2

    arms = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue          # skips <run>.preresume.<stamp>.tar.gz
            if not args.all_dirs and name not in live:
                continue
            probs = scan_arm(d, args.threshold)
            if probs:
                arms[name] = {"dir": d, "root": root, "problems": probs}

    if args.completed_only:
        keep = {n: a for n, a in arms.items() if not _incomplete({n: a})}
        dropped = sorted(set(arms) - set(keep))
        if dropped:
            print(f"--completed-only: dropped {len(dropped)} in-flight arm(s): "
                  + ", ".join(d[-34:] for d in dropped))
        arms = keep

    if not arms:
        print("no arms matched", file=sys.stderr)
        return 2

    # per-arm, own denominator
    for name, a in arms.items():
        st = [p["best_stored"] for p in a["problems"].values()]
        un = [p["best_uniform"] for p in a["problems"].values()]
        a["stored"] = metrics(st)
        a["uniform"] = metrics(un)
        a["n_flipped_evals"] = sum(p["n_flipped"] for p in a["problems"].values())
        a["n_evals"] = sum(p["n_evals"] for p in a["problems"].values())
        a["model"] = "terra" if "terra" in name else "gpt-oss-120b"

    # aligned intersection, computed WITHIN each model group (cross-model is not a
    # valid contrast: different endpoints, different latency, different GPUs)
    aligned = {}
    for model in sorted({a["model"] for a in arms.values()}):
        group = {n: a for n, a in arms.items() if a["model"] == model}
        common = set.intersection(*(set(a["problems"]) for a in group.values())) if group else set()
        aligned[model] = {"n_aligned": len(common), "arms": {}}
        for n, a in group.items():
            st = [a["problems"][p]["best_stored"] for p in sorted(common)]
            un = [a["problems"][p]["best_uniform"] for p in sorted(common)]
            aligned[model]["arms"][n] = {"stored": metrics(st), "uniform": metrics(un)}

    os.makedirs(args.output_dir, exist_ok=True)
    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "uniform_threshold": args.threshold,
        "seam_utc": SEAM_UTC,
        "seam_note": f"eval.py excessive_speedup_threshold {OLD_T} -> {NEW_T} (commit 588a6a5)",
        "baseline": "results/timing/NVIDIA_GH200x2_median (speedups are governor-computed "
                    "against this fixed baseline; re-scoring does not touch the baseline)",
        "status": _status_line(arms),
        "incomplete_arms": _incomplete(arms),
        "live_filtered": not args.all_dirs,
        "aligned_within_model": aligned,
        "arms": {n: {k: v for k, v in a.items() if k != "problems"} for n, a in arms.items()},
        "per_problem": {n: a["problems"] for n, a in arms.items()},
    }
    jp = os.path.join(args.output_dir, "rescore_hack_threshold.json")
    with open(jp, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)

    cp = os.path.join(args.output_dir, "rescore_hack_threshold.csv")
    cols = ["run_name", "model", "n_problems", "n_evals", "n_flipped_evals"]
    for side in ("stored", "uniform"):
        cols += [f"{side}_fast_p_best@{p}" for p in THRESHOLDS] + [f"{side}_best_geomean", f"{side}_n_correct"]
    with open(cp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for n, a in sorted(arms.items()):
            row = {"run_name": n, "model": a["model"], "n_problems": a["stored"]["n_problems"],
                   "n_evals": a["n_evals"], "n_flipped_evals": a["n_flipped_evals"]}
            for side in ("stored", "uniform"):
                for p in THRESHOLDS:
                    row[f"{side}_fast_p_best@{p}"] = round(a[side][f"fast_p_best@{p}"], 4)
                row[f"{side}_best_geomean"] = round(a[side]["best_geomean"], 4)
                row[f"{side}_n_correct"] = a[side]["n_correct"]
            w.writerow(row)

    # ---- console report -------------------------------------------------
    print(f"uniform threshold {args.threshold}x   seam {SEAM_UTC}   arms {len(arms)}"
          f"   {'live-filtered' if not args.all_dirs else 'ALL DIRS'}")
    print("STATUS: " + doc["status"] + "\n")
    for model in sorted(aligned):
        al = aligned[model]
        print(f"=== {model}: aligned on {al['n_aligned']} problems common to all "
              f"{len(al['arms'])} arms ===")
        print(f"{'arm':46s} {'fastp@1 stored->uniform':>26} {'geomean stored->uniform':>26}")
        for n in sorted(al["arms"]):
            s, u = al["arms"][n]["stored"], al["arms"][n]["uniform"]
            d1 = u["fast_p_best@1.0"] - s["fast_p_best@1.0"]
            d2 = u["best_geomean"] - s["best_geomean"]
            mark = "  <-- MOVED" if abs(d1) > 1e-9 or abs(d2) > 0.005 else ""
            tag = n.replace("base_agent_gpt_oss_120b", "oss").replace("base_agent_gpt_5_6_terra", "terra")
            tag = re.sub(r"_itr30_GH200_.*$", "", tag) or "(truncation)"
            print(f"{tag:46s} {s['fast_p_best@1.0']:11.3f} -> {u['fast_p_best@1.0']:<11.3f}"
                  f" {s['best_geomean']:11.3f} -> {u['best_geomean']:<11.3f}{mark}")
        print()
    # ---- markdown ------------------------------------------------------
    mp = os.path.join(args.output_dir, "rescore_hack_threshold.md")
    L = []
    L.append(f"# Uniform-threshold re-score ({args.threshold:g}x)\n")
    L.append(f"Generated {doc['generated_utc']}. Baseline: `NVIDIA_GH200x2_median`.\n")
    if doc["incomplete_arms"]:
        L.append("> **STATUS: PARTIAL.** Incomplete: " + ", ".join(doc["incomplete_arms"]) +
                 ". `ANALYSIS_RULES.md:158` forbids partial prefixes as a final comparative\n"
                 "> result -- read the *deltas* as magnitude, not the levels as a leaderboard.\n")
    else:
        L.append(f"> **STATUS: COMPLETE.** All {len(arms)} arms finished 50/50 with\n"
                 "> `run_summary.json`. Levels are quotable. **But n=1 replicate per cell:** per\n"
                 "> `ANALYSIS_RULES.md` and open item 9 (log-SD 0.147 across identical-config\n"
                 "> replicates), a single replicate cannot support an arm-vs-arm winner claim.\n")
    L.append("## The seam\n")
    L.append(f"`src/kernelbench/eval.py` changed `excessive_speedup_threshold` {OLD_T:g} -> {NEW_T:g} at\n"
             f"**{SEAM_UTC}** (commit `588a6a5`). eval.py is re-imported by every eval spawn, so it\n"
             "reached all live arms with no restart and no log line. `is_hack` gates which iterations\n"
             "may form a best, so identical kernels scored differently either side of that instant.\n")
    L.append(f"- evals whose label changes under a uniform {args.threshold:g}x: "
             f"**{sum(a['n_flipped_evals'] for a in arms.values())}** of "
             f"{sum(a['n_evals'] for a in arms.values())}\n")
    L.append("- all flips are pre-seam evals in the (10, 30] band; re-scoring can only *raise* them,\n"
             "  so this is a one-directional correction toward the post-seam arms.\n")
    L.append("\n## Aligned within model\n")
    L.append("Cross-model contrast is invalid (different endpoints, GPUs, latency), so each model\n"
             "group is aligned on the problems common to all of its arms.\n")
    for model in sorted(aligned):
        al = aligned[model]
        L.append(f"\n### {model} -- {al['n_aligned']} aligned problems, {len(al['arms'])} arms\n")
        L.append("| arm | fast_p@1.0 stored | uniform | delta | geomean stored | uniform | delta |")
        L.append("|---|---|---|---|---|---|---|")
        for n in sorted(al["arms"]):
            s_, u_ = al["arms"][n]["stored"], al["arms"][n]["uniform"]
            tag = re.sub(r"_itr30_GH200_.*$", "",
                         n.replace("base_agent_gpt_oss_120b", "oss").replace("base_agent_gpt_5_6_terra", "terra")) or "(truncation)"
            d1 = u_["fast_p_best@1.0"] - s_["fast_p_best@1.0"]
            d2 = u_["best_geomean"] - s_["best_geomean"]
            b = "**" if (abs(d1) > 1e-9 or abs(d2) > 0.005) else ""
            L.append(f"| {b}{tag}{b} | {s_['fast_p_best@1.0']:.3f} | {u_['fast_p_best@1.0']:.3f} | "
                     f"{d1:+.3f} | {s_['best_geomean']:.3f} | {u_['best_geomean']:.3f} | {d2:+.3f} |")
    L.append("\n## What this does not fix\n")
    L.append("Only the reported metric. At run time `is_hack=True` also vetoed `is_new_best`, so a\n"
             "suppressed kernel was never banked and the agent kept debugging from a worse state.\n"
             "Re-scoring recovers the number, not the search trajectory.\n")
    with open(mp, "w") as fh:
        fh.write("\n".join(L) + "\n")

    tot = sum(a["n_flipped_evals"] for a in arms.values())
    print(f"evals whose is_hack label changed: {tot} of {sum(a['n_evals'] for a in arms.values())}")
    print(f"wrote {jp}\n      {cp}\n      {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
