"""Did the standing cap actually bind, and what would unlimited have given?

The wave contains a natural experiment: `l2` and `l2_hit` ran with NO standing cap
(l2_standing_cap absent from run_summary.json), while `l2_redesign` and `l2_judge`
ran with cap 6. So the uncapped arms measure directly what the floors produce on a
50-problem run without any budget.

Caveat this cannot answer: a cap REFUSAL is recorded nowhere. select_l2_promotions
truncates with `eligible = eligible[:room]` and the census's `eligible_count` is
bound from the RETURN of that function (l2_promotion.py:832), i.e. post-cap. So the
census says how many were promoted, not how many cleared the floors and lost to the
cap -- which is the exact quantity CLAUDE.md 8.8 said a capped arm needs. Print the
trajectory instead and reason from the uncapped arms.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("runs_evolving/gpt-oss-120b/l2redesign")


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    print(f"{'arm':<13}{'cap':>5}{'dedup':>7}{'final':>7}  standing_after trajectory "
          f"(passes where it changed)")
    print("-" * 100)
    for d in sorted(ROOT.iterdir()):
        f = d / "l2_promotions.jsonl"
        s = d / "run_summary.json"
        if not f.exists() or not s.exists():
            continue
        name = d.name.replace("base_agent_gpt_oss_120b_", "")
        name = name[:name.find("_itr30_GH200")]
        cfg = json.loads(s.read_text())
        if not cfg.get("enable_l2"):
            continue
        cap = cfg.get("l2_standing_cap") or 0
        ded = cfg.get("l2_dedup_similarity") or 0.0

        census = [r for r in rows(f) if r.get("event") == "pass"]
        traj, last = [], None
        for r in census:
            v = r.get("standing_after")
            if v != last:
                traj.append(f"gi{r.get('global_iteration')}:{v}")
                last = v
        final = last if last is not None else 0
        print(f"{name:<13}{cap:>5}{ded:>7.2f}{final:>7}  {' '.join(traj)}")

    print("\nThe uncapped arms are the evidence: whatever they ended at is what the")
    print("floors alone produce at this run length.")


if __name__ == "__main__":
    main()
