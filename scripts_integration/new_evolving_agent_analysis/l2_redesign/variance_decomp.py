"""How should GPU hours be spent: more arms, or a better-paired analysis?

Open item 10 puts arm-level replicate noise at log-SD 0.147, so a single
arm-vs-arm contrast needs ~x1.50 to clear 95%. That figure is computed on the
ARM-LEVEL geomean, which throws away the fact that both arms ran the SAME 50
problems. If most of the variance is per-problem and shared, a paired
per-problem analysis cancels it and buys power for free.

This decomposes the variance using every identical-config replicate set
available, and reports the standard error a paired design would achieve against
the unpaired one.

Hack filtering follows CLAUDE.md: walk records[].evaluation per SAMPLE and take
the best with (correct and not is_hack) -- filtering on the record-level
best_is_hack drops whole problems and moves the headline a long way.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def best_clean_speedup(run: dict) -> float | None:
    """Best PER-SAMPLE speedup with correct and not is_hack (CLAUDE.md §4).

    Filtering on the run-level ``best_is_hack`` instead drops the whole problem
    when its single best iteration was a hack, even though a clean slower sample
    usually exists -- worth 1.579 vs 1.389 on one completed control.
    """
    best = None
    for rec in run.get("records") or []:
        ev = rec.get("evaluation")
        if not isinstance(ev, dict):
            continue
        if not ev.get("correct") or ev.get("is_hack"):
            continue
        sp = ev.get("speedup")
        try:
            sp = float(sp)
        except (TypeError, ValueError):
            continue
        if sp > 0 and (best is None or sp > best):
            best = sp
    return best


def load_arm(run_dir: Path) -> dict[str, float]:
    p = run_dir / "evolving_runs.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    runs = data.get("runs") if isinstance(data, dict) else data
    out: dict[str, float] = {}
    for r in runs or []:
        key = str(r.get("problem_id") or r.get("subset_index") or "")
        if not key:
            continue
        v = best_clean_speedup(r)
        if v:
            out[key] = v
    return out


def main(groups: dict[str, list[Path]]) -> None:
    all_pair_logs: list[float] = []
    arm_geo_logs: dict[str, list[float]] = {}

    print(f"{'config':16s} {'reps':>4s} {'common':>7s} {'geomeans':>28s}")
    for name, dirs in groups.items():
        arms = [load_arm(d) for d in dirs]
        arms = [a for a in arms if a]
        if len(arms) < 2:
            continue
        common = set(arms[0])
        for a in arms[1:]:
            common &= set(a)
        if len(common) < 10:
            continue
        geos = []
        for a in arms:
            lg = [math.log(a[p]) for p in common]
            geos.append(math.exp(sum(lg) / len(lg)))
        arm_geo_logs[name] = [math.log(g) for g in geos]
        print(f"{name:16s} {len(arms):>4d} {len(common):>7d}   "
              + " ".join(f"{g:.3f}" for g in geos))

        # paired per-problem log ratios, every rep pair
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                for p in common:
                    all_pair_logs.append(math.log(arms[i][p]) - math.log(arms[j][p]))

    print()
    if not all_pair_logs:
        print("no replicate sets usable")
        return

    sd_pair = st.pstdev(all_pair_logs)
    # per-problem noise of ONE arm = sd of a difference / sqrt(2)
    sd_problem_single = sd_pair / math.sqrt(2)

    # arm-level: sd of log geomean across reps, pooled within config
    within = []
    for name, lg in arm_geo_logs.items():
        if len(lg) >= 2:
            m = sum(lg) / len(lg)
            within += [x - m for x in lg]
    sd_arm = st.pstdev(within) * math.sqrt(len(within) / max(1, len(within) - len(arm_geo_logs)))

    n = 50
    se_unpaired = sd_arm * math.sqrt(2)
    se_paired = sd_pair / math.sqrt(n)

    print(f"per-problem log-ratio SD (rep vs rep) : {sd_pair:.3f}   (n={len(all_pair_logs)} pairs)")
    print(f"  => single-arm per-problem log SD    : {sd_problem_single:.3f}")
    print(f"arm-level log-geomean SD (pooled)     : {sd_arm:.3f}   "
          f"(open item 10 quotes 0.147 from merge_sim08 alone)")
    print()
    print(f"SE of an arm-vs-arm contrast, n=1 per cell:")
    print(f"  unpaired (arm geomeans)  : {se_unpaired:.3f}  -> need x{math.exp(1.96*se_unpaired):.2f} for 95%")
    print(f"  paired per problem (n={n}) : {se_paired:.3f}  -> need x{math.exp(1.96*se_paired):.2f} for 95%")
    if se_paired > 0:
        print(f"  power ratio              : {se_unpaired/se_paired:.1f}x tighter paired")
        eq = (se_unpaired / se_paired) ** 2
        print(f"  => one PAIRED arm-pair is worth ~{eq:.0f} unpaired replicates")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs_ro/gpt-oss-120b/mean")
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        n = d.name
        # config = run name minus the trailing timestamp
        parts = n.split("_")
        cfg = "_".join(parts[:-5]) if len(parts) > 6 else n
        groups[cfg].append(d)
    main({k: v for k, v in groups.items() if len(v) >= 2})
