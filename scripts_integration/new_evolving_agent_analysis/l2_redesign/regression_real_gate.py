"""Regression: drive the boundary replay through the REAL patched gate.

validate_replay.py re-implements the floors. This one imports the actual
``passes_floors`` / ``select_l2_promotions`` from the patched module, so an
exact match against l2_promotions.jsonl proves the redesign left the default
path byte-identical on real data -- not just on synthetic unit-test candidates.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "Self-Evolving-Agent"))

from evolving_common.governor.l2_promotion import (  # noqa: E402
    L2Candidate,
    L2PromotionConfig,
    score_candidate,
    select_l2_promotions,
)
from replay_l2_gate import load_truth  # noqa: E402
from sweep_gates import load_arm  # noqa: E402


def real_gate(cfg: L2PromotionConfig, nb: dict[str, int]):
    def gate(cands, standing, gi):
        objs = []
        for c in cands:
            o = L2Candidate(
                entry_id=c["entry_id"],
                entry=c["entry"],
                total_selections=c["selections"],
                tasks=c["tasks"],
                opportunity=c["opportunity"],
                rate=c["rate"],
                new_bests=nb.get(c["entry_id"], 0),
                total_offers=c["offers"],
                hit_rate=c["hit_rate"],
            )
            o.score = score_candidate(
                o.hit_rate if cfg.use_hit_rate else o.rate, o.tasks, o.new_bests
            )
            objs.append(o)
        chosen = select_l2_promotions(
            objs,
            cfg,
            standing=[{"entry_id": s["entry_id"], "entry": s["entry"]} for s in standing],
        )
        by_id = {c["entry_id"]: c for c in cands}
        return [by_id[o.entry_id] for o in chosen]

    return gate


def main(arm: str) -> int:
    from sweep_gates import run_gate

    rd = Path(arm)
    rows, ca, off, nb, ent = load_arm(rd)
    truth = load_truth(rd)
    cfg = L2PromotionConfig(enabled=True)  # all defaults
    got = run_gate(rows, ca, off, real_gate(cfg, nb), ent)

    tid = {str(t["entry_id"]): t for t in truth}
    gid = {str(g["entry_id"]): g for g in got}
    ok = set(tid) == set(gid)
    print(f"{rd.name}")
    print(f"  recorded : {len(truth)}  {sorted(tid)}")
    print(f"  replayed : {len(got)}  {sorted(gid)}")
    for eid in sorted(set(tid) & set(gid)):
        t, g = tid[eid], gid[eid]
        same = (
            t["promoted_at_global_iter"] == g["promoted_at"]
            and t["total_selections"] == g["selections"]
            and t["distinct_tasks"] == g["tasks"]
            and t["opportunity"] == g["opportunity"]
        )
        ok = ok and same
        print(f"   {'OK ' if same else 'DIFF'} id={eid} gi={g['promoted_at']} "
              f"sel={g['selections']} tasks={g['tasks']} opp={g['opportunity']}")
    print("  VERDICT:", "EXACT MATCH (defaults unchanged)" if ok else "REGRESSION")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = 0
    for a in sys.argv[1:]:
        rc |= main(a)
        print("-" * 72)
    sys.exit(rc)
