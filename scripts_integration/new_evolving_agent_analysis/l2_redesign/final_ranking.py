"""One consolidated ranking of the six L2 designs.

Quality on this wave is a null (lottery_adjusted.py), so a ranking on quality alone
would be ranking noise. This puts quality, the sign test (distribution-free, so a
single lucky problem cannot carry an arm), the mechanism outcome, and the standing
prompt cost in one table, and prints the identical-configuration null contrast on
the same scale so the reader can see how much of the spread is real.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lottery_adjusted import ARMS50, find_lottery, load_all  # noqa: E402
from robust_contrast import sign_test_p  # noqa: E402

ROOT = Path("runs_evolving/gpt-oss-120b/l2redesign")

# pairs at cosine >= 0.80 among each arm's final standing rules, from
# standing_diversity.py (re-run it to refresh)
DUPES = {"l2_redesign": 0, "l2_judge": 2, "l2": 3, "l2_preseed": 3,
         "l2_hit": 4, "l2_extract": 1}
NOTE = {
    "l2_redesign": "hit0.6 + cap6 + dedup0.80",
    "l2_hit":      "hit0.7 only",
    "l2":          "SHIPPED defaults",
    "l2_preseed":  "frozen 5 rules, no promotion",
    "l2_extract":  "hit0.7, extract render",
    "l2_judge":    "loose floors + LLM judge",
}


def standing_chars(arm: str) -> int:
    for d in ROOT.iterdir():
        if f"_{arm}_itr30_" in d.name:
            f = d / "l2_standing.jsonl"
            if f.exists():
                return sum(len(str(json.loads(l).get("text") or ""))
                           for l in f.read_text().splitlines() if l.strip())
    return 0


def paired(a: dict, b: dict, drop: set[str]):
    logs, w, l = [], 0, 0
    for k in sorted(set(a) & set(b)):
        if k in drop:
            continue
        x, y = a.get(k), b.get(k)
        if x and y:
            logs.append(math.log(x / y))
            w += x > y
            l += y > x
    m, sd = st.mean(logs), st.stdev(logs)
    se = sd / math.sqrt(len(logs))
    return math.exp(m), math.exp(m - 1.96 * se), math.exp(m + 1.96 * se), w, l


def main() -> None:
    R = load_all()
    drop = set(find_lottery(R))
    ctl = R["truncation"]

    rows = []
    for a in ARMS50[1:]:
        if a not in R:
            continue
        ratio, lo, hi, w, l = paired(R[a], ctl, drop)
        keys = [k for k in R[a] if k not in drop]
        fp = sum(1 for k in keys if (R[a].get(k) or 0) > 1.0) / len(keys)
        rows.append((ratio, a, lo, hi, w, l, fp, sign_test_p(w, l)))
    rows.sort(reverse=True)

    ck = [k for k in ctl if k not in drop]
    fp_ctl = sum(1 for k in ck if (ctl.get(k) or 0) > 1.0) / len(ck)

    print("Lottery-adjusted. Ranked by paired ratio vs the truncation control.\n")
    print(f"{'design':<13}{'ratio':>7}{'95% CI':>16}{'W-L':>8}{'sign p':>8}"
          f"{'fast_p':>8}{'dup':>5}{'chars':>7}  note")
    print("-" * 95)
    for ratio, a, lo, hi, w, l, fp, p in rows:
        print(f"{a:<13}{ratio:>7.3f}{f'[{lo:.3f},{hi:.3f}]':>16}"
              f"{f'{w}-{l}':>8}{p:>8.2f}{fp:>8.3f}"
              f"{DUPES.get(a, -1):>5}{standing_chars(a):>7}  {NOTE.get(a, '')}")
    print(f"{'(control)':<13}{1.0:>7.3f}{'-':>16}{'-':>8}{'-':>8}{fp_ctl:>8.3f}"
          f"{0:>5}{0:>7}  no L2")

    nr, nlo, nhi, nw, nl = paired(R["q15_ctl_r2"], R["q15_ctl_r1"], drop)
    print(f"\nNULL contrast, two byte-identical control arms:")
    print(f"{'ctl_r2/ctl_r1':<13}{nr:>7.3f}{f'[{nlo:.3f},{nhi:.3f}]':>16}"
          f"{f'{nw}-{nl}':>8}")
    print("\nThe null sits inside the same band as every treatment: the whole")
    print("column is consistent with no effect. 'dup' = standing-rule pairs at")
    print("cosine >= 0.80; 'chars' = standing text added to EVERY coder prompt.")


if __name__ == "__main__":
    main()
