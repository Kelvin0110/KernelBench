"""Paired per-problem log-ratio comparison for the L2 wave.

Why paired and not a difference of arm geomeans: problem difficulty dominates the
variance. Measured on this project's own data, pairing is ~1.8x tighter than
comparing arm-level numbers, which is the difference between a usable screen and
nothing at all at n<=50.

Two rules from CLAUDE.md 4 are load-bearing here and both are easy to get wrong:

  * Filter hacks PER SAMPLE, not per problem. Each problem record carries a
    top-level best_speedup/best_is_hack describing its single best attempt;
    filtering on that field discards the whole problem when its best attempt was a
    hack, even though a clean slower attempt usually exists. On one completed arm
    that mistake moved best_geomean 1.389 -> 1.579.

  * fast_p_best@1.0 is the headline; best_geomean is secondary
    (ANALYSIS_RULES.md:81-85 forbids best-geomean-only leaderboards).

Shape note: evolving_runs.json is {"runs": [problem, ...]}, each problem has
records[] of attempts and each attempt has a single `evaluation` DICT (not a list).

Usage:  paired_report.py <runs-root> --baseline <substring>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

HACK_THRESHOLD = 30.0


def short(run_dir: Path) -> str:
    s = run_dir.name.replace("base_agent_gpt_oss_120b_", "")
    i = s.find("_itr30_GH200")
    return s[:i] if i >= 0 else s


def is_hack(ev: dict) -> bool:
    """Stored flag, OR'd with a re-check of the excessive-speedup rule.

    The threshold is a default argument in eval.py, which every eval child
    re-imports, so it can change mid-wave (10x -> 30x on 2026-08-24) and a run's
    is_hack column can mix two rules. Verified for THIS wave: all 11 arms launched
    2026-08-27, and the stored flags are exactly the 12 correct samples at
    speedup >= 30 -- already uniform, with zero STRICT static failures.

    Do NOT treat a non-empty static_check_warnings as a hack. resolve_is_hack
    (static_check.py:44) fires only on STRICT *errors* or metadata
    .excessive_speedup; workload_shrink and "uses torch.matmul"-class warnings are
    recorded for audit and explicitly do not set the flag. Keying on that field
    discarded 383 of 756 clean samples and cut the control from 47 problems to 27.
    """
    if ev.get("is_hack"):
        return True
    sp = ev.get("speedup")
    return sp is not None and sp >= HACK_THRESHOLD


def best_clean(problem: dict) -> float | None:
    best = None
    for rec in problem.get("records") or []:
        ev = rec.get("evaluation") or {}
        if not ev.get("correct") or is_hack(ev):
            continue
        sp = ev.get("speedup")
        if sp is None or sp <= 0:
            continue
        if best is None or sp > best:
            best = float(sp)
    return best


def load_run(run_dir: Path) -> dict[str, float | None]:
    data = json.loads((run_dir / "evolving_runs.json").read_text())
    out: dict[str, float | None] = {}
    for prob in data.get("runs") or []:
        out[f"L{prob.get('level')}P{prob.get('problem_id')}"] = best_clean(prob)
    return out


def compare(base: dict, arm: dict) -> dict:
    """Paired log-ratio over problems where BOTH arms produced a clean kernel."""
    common = sorted(set(base) & set(arm))
    logs = []
    a_win = b_win = 0
    for k in common:
        b, a = base.get(k), arm.get(k)
        if b is None or a is None:
            continue
        logs.append(math.log(a / b))
        if a > b:
            a_win += 1
        elif b > a:
            b_win += 1

    # fast_p@1.0 on the full aligned denominator: a problem with no clean correct
    # sample counts as a fail, so the denominator is all common problems.
    fp_b = sum(1 for k in common if (base.get(k) or 0) > 1.0)
    fp_a = sum(1 for k in common if (arm.get(k) or 0) > 1.0)
    b01 = sum(1 for k in common if (base.get(k) or 0) > 1.0 and not (arm.get(k) or 0) > 1.0)
    b10 = sum(1 for k in common if not (base.get(k) or 0) > 1.0 and (arm.get(k) or 0) > 1.0)

    res = {
        "n_common": len(common), "n_paired": len(logs),
        "fp_base": fp_b / len(common) if common else 0.0,
        "fp_arm": fp_a / len(common) if common else 0.0,
        "b01": b01, "b10": b10, "arm_wins": a_win, "base_wins": b_win,
    }
    if len(logs) >= 2:
        m, sd = st.mean(logs), st.stdev(logs)
        se = sd / math.sqrt(len(logs))
        res.update(ratio=math.exp(m), lo=math.exp(m - 1.96 * se),
                   hi=math.exp(m + 1.96 * se), sd=sd)
    return res


def geomean_clean(probs: dict) -> tuple[float, int]:
    vals = [v for v in probs.values() if v]
    if not vals:
        return float("nan"), 0
    return math.exp(sum(math.log(v) for v in vals) / len(vals)), len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", type=Path)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()

    dirs = sorted(d for d in args.runs_root.iterdir()
                  if d.is_dir() and (d / "evolving_runs.json").exists())
    runs = {short(d): load_run(d) for d in dirs}

    base_key = next((k for k in runs if args.baseline in k), None)
    if base_key is None:
        raise SystemExit(f"no run matching {args.baseline!r}; have {sorted(runs)}")

    gm, n = geomean_clean(runs[base_key])
    print(f"baseline: {base_key}   geomean={gm:.3f} (n={n})   "
          f"fast_p@1.0={sum(1 for v in runs[base_key].values() if (v or 0) > 1) / len(runs[base_key]):.3f}\n")
    print(f"{'arm':<14} {'geo':>6} {'n':>4} {'ratio':>7} {'95% CI':>16} "
          f"{'fp':>6} {'dfp':>7} {'McN':>7} {'W-L':>7}")
    print("-" * 82)
    for key in sorted(runs):
        if key == base_key:
            continue
        r = compare(runs[base_key], runs[key])
        g, _ = geomean_clean(runs[key])
        ci = f"[{r['lo']:.3f},{r['hi']:.3f}]" if "lo" in r else "-"
        print(f"{key:<14} {g:>6.3f} {r['n_paired']:>4} {r.get('ratio', float('nan')):>7.3f} "
              f"{ci:>16} {r['fp_arm']:>6.3f} {r['fp_arm'] - r['fp_base']:>+7.3f} "
              f"{str(r['b10']) + '/' + str(r['b01']):>7} "
              f"{str(r['arm_wins']) + '-' + str(r['base_wins']):>7}")

    print("\nratio>1 favours the arm. dfp = fast_p@1.0 minus the control's. "
          "McN = gained/lost on fast_p.\nCI is the paired per-problem log-ratio and "
          "does NOT include between-run noise,\nwhich at n=1 per cell adds log-SD "
          "~0.15-0.18 -- see the identical-control pair.")


if __name__ == "__main__":
    main()
