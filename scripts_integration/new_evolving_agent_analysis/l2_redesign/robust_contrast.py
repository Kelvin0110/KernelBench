"""Robust (outlier-resistant) version of the paired contrasts.

Motivation: the per-problem log-ratio SD on this wave is ~1.0, i.e. a factor of e.
At that spread the geometric mean is a poor summary -- a single problem where one
arm found a big win and the other did not moves it more than the other 44 combined.
So report, alongside the geomean:

  * median log-ratio         (robust location)
  * sign test on wins/losses (distribution-free, ignores magnitude entirely)
  * the 3 problems contributing most to the log-ratio, so the reader can see
    whether a result is broad or is one kernel.
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paired_report import load_run, short  # noqa: E402
from pair_contrast import ROOTS  # noqa: E402


def all_runs() -> dict[str, dict]:
    out = {}
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "evolving_runs.json").exists():
                out[short(d)] = load_run(d)
    return out


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial at p=0.5, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def analyse(name: str, a: dict, b: dict) -> None:
    pairs = []
    for k in sorted(set(a) & set(b)):
        x, y = a.get(k), b.get(k)
        if x and y:
            pairs.append((k, math.log(x / y), x, y))
    if len(pairs) < 2:
        print(f"{name}: too few pairs")
        return
    logs = [p[1] for p in pairs]
    wins = sum(1 for p in pairs if p[1] > 0)
    losses = sum(1 for p in pairs if p[1] < 0)
    med = st.median(logs)
    p = sign_test_p(wins, losses)

    print(f"{name:<32} n={len(pairs):<3} geo={math.exp(st.mean(logs)):.3f}  "
          f"median={math.exp(med):.3f}  W-L={wins}-{losses}  sign p={p:.3f}")

    top = sorted(pairs, key=lambda t: -abs(t[1]))[:3]
    detail = ", ".join(f"{k} {math.exp(l):.2f}x" for k, l, _, _ in top)
    share = sum(abs(t[1]) for t in top) / sum(abs(x) for x in logs) * 100
    print(f"{'':32}   top-3 movers: {detail}   ({share:.0f}% of total |log|)")


def main() -> None:
    R = all_runs()
    print("=== NULL: identical config ===")
    analyse("ctl_r2 / ctl_r1", R["q15_ctl_r2"], R["q15_ctl_r1"])

    print("\n=== 50-problem arms vs truncation control (all problems) ===")
    ctl = R["truncation"]
    for arm in ("l2", "l2_hit", "l2_redesign", "l2_judge", "l2_preseed", "l2_extract"):
        if arm in R:
            analyse(f"{arm} / truncation", R[arm], ctl)

    print("\n=== pre-seed replicates ===")
    for r in ("r1", "r2"):
        analyse(f"pre_{r} / ctl_{r}", R[f"q15_pre_{r}"], R[f"q15_ctl_{r}"])


if __name__ == "__main__":
    main()
