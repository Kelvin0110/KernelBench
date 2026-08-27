"""Find the global_iter offset between the reconstructed and recorded scales.

Both L2 arms were resumed. Iterations discarded by the resume still bumped the
real ``global_iteration`` counter but left no chat records, so a reconstruction
that numbers surviving iterations 1..N runs on a shifted scale. ``created_at``
comes from the ledger (real scale), so it must be compared against a real-scale
boundary or ``opportunity`` is wrong.

This searches for the constant offset that makes the shipped gate reproduce the
recorded promotions exactly.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_l2_gate import (  # noqa: E402
    load_created_at,
    load_iteration_events,
    load_new_best_totals,
    load_truth,
    read_jsonl,
)

MIN_TASKS, MIN_SELECTIONS, MIN_RATE, MIN_NEW_BESTS = 3, 50, 0.70, 0


def replay_with_offset(
    events: list,
    created_at: dict[str, int],
    offset: int,
    nb_final: dict[str, int],
) -> list[dict]:
    ev: dict[str, dict] = {}
    standing_ids: set[str] = set()
    promotions: list[dict] = []
    gi = offset
    for idx, event in enumerate(events):
        gi += 1
        for eid in event.selected:
            if eid in standing_ids:
                continue
            rec = ev.setdefault(eid, {"sel": 0, "tasks": set()})
            rec["sel"] += 1
            rec["tasks"].add(event.task_key)
        is_last = (idx + 1 == len(events)) or (events[idx + 1].problem != event.problem)
        if not is_last:
            continue
        eligible = []
        for eid, rec in ev.items():
            if eid in standing_ids:
                continue
            opp = max(1, gi - created_at.get(eid, 0))
            rate = rec["sel"] / opp
            nb = nb_final.get(eid, 0)
            if (len(rec["tasks"]) >= MIN_TASKS and rec["sel"] >= MIN_SELECTIONS
                    and rate >= MIN_RATE and nb >= MIN_NEW_BESTS):
                eligible.append({
                    "entry_id": eid, "total_selections": rec["sel"],
                    "tasks": len(rec["tasks"]), "opportunity": opp, "rate": rate,
                    "new_bests": nb,
                    "score": rate * math.log1p(len(rec["tasks"])) * math.log1p(nb),
                    "promoted_at_global_iter": gi,
                })
        eligible.sort(key=lambda c: (-c["score"], c["entry_id"]))
        for c in eligible:
            standing_ids.add(c["entry_id"])
            promotions.append(c)
    return promotions


def mismatch(truth: list[dict], got: list[dict]) -> int:
    """Total absolute error across matched ids, plus a big penalty per set diff."""
    tid = {str(t["entry_id"]): t for t in truth}
    gid = {str(g["entry_id"]): g for g in got}
    err = 1000 * len(set(tid) ^ set(gid))
    for eid in set(tid) & set(gid):
        t, g = tid[eid], gid[eid]
        err += abs(int(t["promoted_at_global_iter"]) - int(g["promoted_at_global_iter"]))
        err += abs(int(t["total_selections"]) - int(g["total_selections"]))
        err += abs(int(t["distinct_tasks"]) - int(g["tasks"]))
    return err


def main(run_dir: str) -> None:
    rd = Path(run_dir)
    truth = load_truth(rd)
    nb_final = load_new_best_totals(rd)
    events = load_iteration_events(rd)
    ledger = json.loads((rd / "l1_skill_usage.json").read_text())
    real_final = int(ledger.get("global_iteration") or 0)
    print(f"run: {rd.name}")
    print(f"  reconstructed iterations = {len(events)}")
    print(f"  ledger final global_iter = {real_final}")
    print(f"  naive offset (ledger - reconstructed) = {real_final - len(events)}")
    print(f"  ground-truth promotions = {len(truth)}")
    if not truth:
        print("  (no ground truth in this arm; offset cannot be calibrated here)")
        return
    created_at = load_created_at(rd)
    best = None
    for off in range(0, 301):
        got = replay_with_offset(events, created_at, off, nb_final)
        err = mismatch(truth, got)
        if best is None or err < best[1]:
            best = (off, err, got)
    off, err, got = best
    print(f"  BEST offset = {off}  (error {err})")
    for g in sorted(got, key=lambda x: x["promoted_at_global_iter"]):
        t = next((t for t in truth if str(t["entry_id"]) == g["entry_id"]), None)
        tag = "OK" if t and t["promoted_at_global_iter"] == g["promoted_at_global_iter"] \
            and t["total_selections"] == g["total_selections"] else "DIFF"
        ts = f" truth(gi={t['promoted_at_global_iter']},sel={t['total_selections']},tasks={t['distinct_tasks']},opp={t['opportunity']})" if t else " truth(ABSENT)"
        print(f"   [{tag}] id={g['entry_id']} gi={g['promoted_at_global_iter']} "
              f"sel={g['total_selections']} tasks={g['tasks']} opp={g['opportunity']} "
              f"rate={g['rate']:.4f}{ts}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(a)
        print("-" * 72)
