"""Explicit A-vs-B paired contrasts, plus the pooled pre-seed effect.

The point of this file is the NULL contrast: q15_ctl_r2 vs q15_ctl_r1 are two arms
with byte-identical configuration, so whatever separation they show is pure
run-to-run noise and is the ruler every treatment contrast must be read against.
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paired_report import load_run, short  # noqa: E402

ROOTS = [
    Path("runs_evolving/gpt-oss-120b/l2redesign"),
    Path("runs_evolving/gpt-oss-120b/l2quick"),
]


def all_runs() -> dict[str, dict]:
    out = {}
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "evolving_runs.json").exists():
                out[short(d)] = load_run(d)
    return out


def logs_for(a: dict, b: dict) -> tuple[list[float], list[str]]:
    """log(a/b) over problems where both produced a clean correct kernel."""
    keys, vals = [], []
    for k in sorted(set(a) & set(b)):
        x, y = a.get(k), b.get(k)
        if x and y:
            vals.append(math.log(x / y))
            keys.append(k)
    return vals, keys


def report(label: str, logs: list[float]) -> None:
    if len(logs) < 2:
        print(f"{label:<34} n={len(logs):<3} (too few)")
        return
    m, sd = st.mean(logs), st.stdev(logs)
    se = sd / math.sqrt(len(logs))
    lo, hi = math.exp(m - 1.96 * se), math.exp(m + 1.96 * se)
    sig = "" if lo <= 1.0 <= hi else "  *"
    print(f"{label:<34} n={len(logs):<3} ratio={math.exp(m):.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  sd={sd:.3f}{sig}")


def main() -> None:
    R = all_runs()

    print("=== NULL CONTRAST: identical configuration, different run ===")
    print("Anything a treatment shows that is no larger than this is not an effect.\n")
    if "q15_ctl_r2" in R and "q15_ctl_r1" in R:
        report("ctl_r2 / ctl_r1  (both control)", logs_for(R["q15_ctl_r2"], R["q15_ctl_r1"])[0])

    print("\n=== PRE-SEED, paired within replicate (15 problems, clean window) ===")
    pooled: list[float] = []
    for r in ("r1", "r2"):
        pre, ctl = f"q15_pre_{r}", f"q15_ctl_{r}"
        if pre in R and ctl in R:
            lg, _ = logs_for(R[pre], R[ctl])
            report(f"pre_{r} / ctl_{r}", lg)
            pooled += lg
    report("POOLED pre / ctl", pooled)

    print("\n=== 50-problem arms vs their own control, restricted to problems 11-25 ===")
    print("The pre-seeded rules were distilled from L1P22/L1P50 (both in problems")
    print("1-10), so only the 11-25 window is uncontaminated for l2_preseed.\n")
    ctl50 = R.get("truncation")
    if ctl50:
        # subset order matters: recover the 11-25 window from the quick arms' key set
        window = set(R.get("q15_ctl_r1", {}))
        for arm in ("l2", "l2_hit", "l2_redesign", "l2_judge", "l2_preseed", "l2_extract"):
            if arm not in R:
                continue
            a = {k: v for k, v in R[arm].items() if k in window}
            b = {k: v for k, v in ctl50.items() if k in window}
            report(f"{arm} / truncation  [11-25]", logs_for(a, b)[0])

    print("\nA '*' marks a CI excluding 1.0 -- but note the null contrast above:")
    print("between-run noise is NOT in these intervals, so a starred result at")
    print("n=1 per cell is a screen, not a finding.")


if __name__ == "__main__":
    main()
