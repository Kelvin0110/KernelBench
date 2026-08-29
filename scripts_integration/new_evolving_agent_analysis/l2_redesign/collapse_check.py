"""Is the arm ranking an artifact of a few algebraically-collapsible problems?

Project memory (collapse-problems-inflate-terra-geomean) records L2P13/42/51/56 as
problems where the reference module is algebraically collapsible -- an agent that
spots the shortcut gets a 10-30x speedup, one that does not gets ~1x. Finding it is
close to a coin flip, so those problems inject enormous between-run variance into
any geometric mean.

robust_contrast.py showed the same handful of problem ids dominating every arm
contrast on this wave. This quantifies it: print the per-arm best clean speedup on
the suspect problems, then recompute each arm's geomean with them removed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paired_report import load_run, short  # noqa: E402
from pair_contrast import ROOTS  # noqa: E402

# Ids that dominated the |log| budget in robust_contrast.py, plus the ones project
# memory already names. Kept explicit rather than auto-selected: picking outliers
# by their effect on the very statistic under test would be circular.
SUSPECT = ["L2P42", "L2P51", "L2P97", "L2P13", "L2P56"]


def main() -> None:
    R: dict[str, dict] = {}
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "evolving_runs.json").exists():
                R[short(d)] = load_run(d)

    arms50 = [a for a in ("truncation", "l2", "l2_hit", "l2_redesign",
                          "l2_judge", "l2_preseed", "l2_extract") if a in R]

    print("Best clean speedup on the suspect problems (50-problem arms):\n")
    hdr = "".join(f"{s:>9}" for s in SUSPECT)
    print(f"{'arm':<14}{hdr}")
    for a in arms50:
        row = "".join(
            f"{R[a][p]:>9.2f}" if R[a].get(p) else f"{'--':>9}"
            for p in SUSPECT
        )
        print(f"{a:<14}{row}")

    print("\nSame problems across the FOUR 15-problem replicates")
    print("(q15_ctl_r1/r2 are byte-identical configurations):\n")
    arms15 = [a for a in ("q15_ctl_r1", "q15_ctl_r2", "q15_pre_r1", "q15_pre_r2") if a in R]
    print(f"{'arm':<14}{hdr}")
    for a in arms15:
        row = "".join(
            f"{R[a][p]:>9.2f}" if R[a].get(p) else f"{'--':>9}"
            for p in SUSPECT
        )
        print(f"{a:<14}{row}")

    print("\nArm geomean with and without the suspect problems:\n")
    print(f"{'arm':<14}{'all':>8}{'n':>4}{'excl':>9}{'n':>4}{'shift':>9}")
    for a in arms50:
        vals = [(k, v) for k, v in R[a].items() if v]
        g_all = math.exp(sum(math.log(v) for _, v in vals) / len(vals))
        keep = [(k, v) for k, v in vals if k not in SUSPECT]
        g_ex = math.exp(sum(math.log(v) for _, v in keep) / len(keep))
        print(f"{a:<14}{g_all:>8.3f}{len(vals):>4}{g_ex:>9.3f}{len(keep):>4}"
              f"{g_ex / g_all:>9.3f}")


if __name__ == "__main__":
    main()
