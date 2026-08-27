"""Is new_best_attributions an independent signal, or selections in disguise?

The gate has one outcome-based floor (min_new_bests) and it ships disabled. If
selection carries no outcome information (see outcome_lift.py), then a skill's
new-best count is just its selection count times a roughly constant base rate --
in which case min_new_bests is a noisier copy of min_selections and there is no
outcome evidence in the ledger at all.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def main(run_dir: str) -> None:
    rd = Path(run_dir)
    s = json.loads((rd / "l1_skill_usage.json").read_text())
    sel, nbs, ratios = [], [], []
    for v in (s.get("skills") or {}).values():
        a = int(v.get("total_selections") or 0)
        b = int(v.get("new_best_attributions") or 0)
        if a >= 10:
            sel.append(float(a))
            nbs.append(float(b))
            ratios.append(b / a)
    r = pearson(sel, nbs)
    print(f"{rd.name[:52]}")
    print(f"  skills with >=10 selections : {len(sel)}")
    print(f"  corr(selections, new_bests) : {r:.3f}")
    if ratios:
        print(f"  new_bests / selections      : median {st.median(ratios):.3f}  "
              f"IQR {st.quantiles(ratios, n=4)[0]:.3f}-{st.quantiles(ratios, n=4)[2]:.3f}")
        print(f"  spread of that ratio (CV)   : {st.pstdev(ratios)/st.mean(ratios):.3f}")
    print("  => a correlation near 1 with a tight ratio means min_new_bests")
    print("     is min_selections rescaled, not an independent outcome signal.")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(a)
        print()
