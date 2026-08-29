"""Re-run the contrasts with 'lottery' problems identified TREATMENT-AGNOSTICALLY.

collapse_check.py showed a handful of problems are bimodal: an arm either finds an
algebraic collapse (7-23x) or does not (~1x), and two byte-identical control arms
land on opposite sides. Those problems inject variance that has nothing to do with
any treatment, and because they are the largest speedups in the set they dominate
every geometric mean.

The trap is circularity: picking the problems that most affect the statistic under
test, then removing them, can manufacture any result. So the selection rule here
NEVER looks at which arm is which. A problem is a lottery iff, across all 11 arms,

    max(clean best speedup) / min(clean best speedup) >= SPREAD

i.e. the same problem under 11 independent runs produced wildly different outcomes.
That is a property of the problem. The arm labels are not consulted, so the rule
cannot be tuned toward a desired winner.

Both directions are reported: with and without, so the reader sees the whole effect
of the adjustment rather than only the flattering half.
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paired_report import load_run, short  # noqa: E402
from pair_contrast import ROOTS  # noqa: E402

SPREAD = 4.0
ARMS50 = ("truncation", "l2", "l2_hit", "l2_redesign", "l2_judge", "l2_preseed", "l2_extract")


def load_all() -> dict[str, dict]:
    out = {}
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "evolving_runs.json").exists():
                out[short(d)] = load_run(d)
    return out


def find_lottery(R: dict[str, dict]) -> list[str]:
    keys: set[str] = set()
    for probs in R.values():
        keys |= set(probs)
    out = []
    for k in sorted(keys):
        vals = [p[k] for p in R.values() if p.get(k)]
        if len(vals) >= 3 and max(vals) / min(vals) >= SPREAD:
            out.append(k)
    return out


def contrast(a: dict, b: dict, drop: set[str]) -> tuple[float, float, float, int, int, int]:
    logs, w, l = [], 0, 0
    for k in sorted(set(a) & set(b)):
        if k in drop:
            continue
        x, y = a.get(k), b.get(k)
        if x and y:
            logs.append(math.log(x / y))
            w += x > y
            l += y > x
    if len(logs) < 2:
        return (float("nan"),) * 3 + (0, 0, 0)
    m, sd = st.mean(logs), st.stdev(logs)
    se = sd / math.sqrt(len(logs))
    return math.exp(m), math.exp(m - 1.96 * se), math.exp(m + 1.96 * se), len(logs), w, l


def main() -> None:
    R = load_all()
    lottery = find_lottery(R)

    print(f"Lottery problems (max/min clean speedup across 11 arms >= {SPREAD}x),")
    print("selected without reference to arm identity:\n")
    for k in lottery:
        vals = sorted((p[k] for p in R.values() if p.get(k)), reverse=True)
        print(f"  {k:<8} n={len(vals):<3} "
              f"range {min(vals):.2f} - {max(vals):.2f}x   "
              f"values {[round(v, 1) for v in vals]}")
    print(f"\n{len(lottery)} of ~50 problems.\n")

    drop = set(lottery)
    ctl = R["truncation"]

    print("=== 50-problem arms vs truncation ===")
    print(f"{'arm':<14}{'ratio(all)':>12}{'ratio(adj)':>12}{'95% CI (adj)':>18}"
          f"{'n':>5}{'W-L':>8}")
    for a in ARMS50[1:]:
        if a not in R:
            continue
        r0 = contrast(R[a], ctl, set())
        r1 = contrast(R[a], ctl, drop)
        ci = f"[{r1[1]:.3f}, {r1[2]:.3f}]"
        print(f"{a:<14}{r0[0]:>12.3f}{r1[0]:>12.3f}{ci:>18}{r1[3]:>5}"
              f"{str(r1[4]) + '-' + str(r1[5]):>8}")

    print("\n=== NULL contrast, identical configuration ===")
    for tag, x, y in (("ctl_r2 / ctl_r1", "q15_ctl_r2", "q15_ctl_r1"),):
        r0 = contrast(R[x], R[y], set())
        r1 = contrast(R[x], R[y], drop)
        print(f"{tag:<14}{r0[0]:>12.3f}{r1[0]:>12.3f}"
              f"{f'[{r1[1]:.3f}, {r1[2]:.3f}]':>18}{r1[3]:>5}"
              f"{str(r1[4]) + '-' + str(r1[5]):>8}")

    print("\n=== pre-seed replicates ===")
    for r in ("r1", "r2"):
        r0 = contrast(R[f"q15_pre_{r}"], R[f"q15_ctl_{r}"], set())
        r1 = contrast(R[f"q15_pre_{r}"], R[f"q15_ctl_{r}"], drop)
        print(f"{'pre_' + r + ' / ctl_' + r:<14}{r0[0]:>12.3f}{r1[0]:>12.3f}"
              f"{f'[{r1[1]:.3f}, {r1[2]:.3f}]':>18}{r1[3]:>5}"
              f"{str(r1[4]) + '-' + str(r1[5]):>8}")

    print("\n=== arm geomeans, adjusted ===")
    print(f"{'arm':<14}{'all':>8}{'n':>4}{'adjusted':>10}{'n':>4}")
    for a in ARMS50:
        if a not in R:
            continue
        vals = [(k, v) for k, v in R[a].items() if v]
        keep = [v for k, v in vals if k not in drop]
        g_all = math.exp(sum(math.log(v) for _, v in vals) / len(vals))
        g_adj = math.exp(sum(math.log(v) for v in keep) / len(keep))
        print(f"{a:<14}{g_all:>8.3f}{len(vals):>4}{g_adj:>10.3f}{len(keep):>4}")


if __name__ == "__main__":
    main()
