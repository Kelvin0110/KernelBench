"""Validate the offline replay against l2_promotions.jsonl ground truth.

The shipped gate is re-implemented here as a `gate` callable and replayed. If
the replay is faithful it must reproduce, for each real promotion: the entry id,
the boundary (promoted_at_global_iter), total_selections, distinct_tasks,
opportunity and selection_rate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_l2_gate import load_truth, load_new_best_totals, replay  # noqa: E402

# Shipped defaults, l2_promotion.py:69-76
MIN_TASKS = 3
MIN_SELECTIONS = 50
MIN_RATE = 0.70
MIN_NEW_BESTS = 0
MAX_ENTRIES = 0


def score_candidate(rate: float, tasks: int, new_bests: int) -> float:
    return float(rate) * math.log1p(max(0, tasks)) * math.log1p(max(0, new_bests))


def shipped_gate(candidates, standing, gi, ctx):
    eligible = [
        c
        for c in candidates
        if c["tasks"] >= MIN_TASKS
        and c["total_selections"] >= MIN_SELECTIONS
        and c["rate"] >= MIN_RATE
        and c["new_bests"] >= MIN_NEW_BESTS
    ]
    for c in eligible:
        c["score"] = score_candidate(c["rate"], c["tasks"], c["new_bests"])
    eligible.sort(key=lambda c: (-c["score"], c["entry_id"]))
    if MAX_ENTRIES > 0:
        eligible = eligible[:MAX_ENTRIES]
    return eligible


def main(run_dir: str) -> int:
    rd = Path(run_dir)
    truth = load_truth(rd)
    # new_bests is only stored as a final total; the gate's new-bests floor is
    # disabled by default so it cannot affect WHICH entries pass. Feed the final
    # total so the ranking score is comparable.
    nb_final = load_new_best_totals(rd)
    out = replay(rd, shipped_gate, new_best_hook=lambda eid, gi: nb_final.get(eid, 0))
    got = out["promotions"]

    print(f"run: {rd.name}")
    print(f"  iterations reconstructed : {out['n_events']}  (final global_iter {out['final_global_iter']})")
    print(f"  ground-truth promotions  : {len(truth)}")
    print(f"  replayed promotions      : {len(got)}")
    print()

    by_id_truth = {str(t["entry_id"]): t for t in truth}
    by_id_got = {str(g["entry_id"]): g for g in got}

    fields = [
        ("promoted_at_global_iter", "promoted_at_global_iter"),
        ("total_selections", "total_selections"),
        ("distinct_tasks", "tasks"),
        ("opportunity", "opportunity"),
    ]
    ok = True
    all_ids = sorted(set(by_id_truth) | set(by_id_got), key=lambda x: int(x) if x.isdigit() else 0)
    for eid in all_ids:
        t, g = by_id_truth.get(eid), by_id_got.get(eid)
        if t is None:
            print(f"  [EXTRA]   id={eid} replayed but not in ground truth: {g}")
            ok = False
            continue
        if g is None:
            print(f"  [MISSING] id={eid} in ground truth but not replayed: gi={t.get('promoted_at_global_iter')}")
            ok = False
            continue
        diffs = []
        for tk, gk in fields:
            tv, gv = t.get(tk), g.get(gk)
            if tv != gv:
                diffs.append(f"{tk}: truth={tv} replay={gv}")
        tr, gr = float(t.get("selection_rate", 0)), float(g.get("rate", 0))
        if abs(tr - gr) > 5e-4:
            diffs.append(f"selection_rate: truth={tr} replay={round(gr,4)}")
        status = "OK  " if not diffs else "DIFF"
        if diffs:
            ok = False
        print(f"  [{status}] id={eid} gi={g.get('promoted_at_global_iter')} "
              f"sel={g.get('total_selections')} tasks={g.get('tasks')} "
              f"opp={g.get('opportunity')} rate={round(gr,4)}")
        for d in diffs:
            print(f"           - {d}")
    print()
    print("  VERDICT:", "EXACT MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = 0
    for arg in sys.argv[1:]:
        rc |= main(arg)
        print("-" * 72)
    sys.exit(rc)
