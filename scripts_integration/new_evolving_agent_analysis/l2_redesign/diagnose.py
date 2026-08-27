"""Why does the same gate promote 9 / 4 / 0 rules on different arms?

Hypothesis under test: ``rate = total_selections / (global_iter - created_at)``
has a denominator that keeps growing after the skill stops being *visible* to
the extractor.

``read_l1_extractor_catalog`` (memory_manager.py:801-820) shows the extractor
only the newest ``DEFAULT_L1_EXTRACTOR_CATALOG_MAX = 50`` active skills unless
governance is on. An L2 arm has no governance flags, so the cap is in force.
A skill therefore accrues selections only while it is among the newest 50; after
that its numerator freezes while its denominator keeps ticking, so its rate
decays monotonically to 0.

How fast a skill scrolls out of that window is set by how fast the arm mints new
L1 skills -- an arm-level property, not a property of the skill. If that is the
mechanism, the promotion count is largely a function of catalog growth rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_l2_gate import (  # noqa: E402
    global_iter_offset,
    load_created_at,
    load_iteration_events,
    read_jsonl,
)

CATALOG_MAX = 50  # DEFAULT_L1_EXTRACTOR_CATALOG_MAX, memory_manager.py:48
MIN_TASKS, MIN_SELECTIONS, MIN_RATE = 3, 50, 0.70


def extractor_candidate_count(run_dir: Path) -> tuple[int, int]:
    """(median, max) number of candidate skills shown to the extractor."""
    counts = []
    ws_root = run_dir / "workspaces"
    for ws in sorted(ws_root.iterdir()) if ws_root.exists() else []:
        for rec in read_jsonl(ws / "chat_history.jsonl"):
            if rec.get("phase") != "extractor":
                continue
            msgs = rec.get("messages") or []
            blob = "\n".join(
                m.get("content") or "" for m in msgs if isinstance(m, dict)
            )
            # Entries are rendered with an "entry_id" marker per candidate.
            n = blob.count('"entry_id"') or blob.count("entry_id:")
            if n:
                counts.append(n)
    if not counts:
        return (0, 0)
    counts.sort()
    return (counts[len(counts) // 2], counts[-1])


def main(run_dir: str) -> dict:
    rd = Path(run_dir)
    events = load_iteration_events(rd)
    created_at = load_created_at(rd)
    offset = global_iter_offset(rd, len(events))
    n_entries = sum(1 for _ in read_jsonl(rd / "shared_l1.jsonl"))

    # Track, per skill, its best-ever rate and the boundary it occurred at.
    sel: dict[str, int] = {}
    tasks: dict[str, set] = {}
    best_rate: dict[str, tuple[float, int, int]] = {}  # eid -> (rate, gi, sel)
    # Max rate for skills that ALSO clear the other two floors at that moment.
    best_rate_qualified: dict[str, tuple[float, int, int]] = {}

    gi = offset
    for idx, e in enumerate(events):
        gi += 1
        for eid in e.selected:
            sel[eid] = sel.get(eid, 0) + 1
            tasks.setdefault(eid, set()).add(e.task_key)
        is_last = (idx + 1 == len(events)) or (events[idx + 1].problem != e.problem)
        if not is_last:
            continue
        for eid, s in sel.items():
            opp = max(1, gi - created_at.get(eid, 0))
            r = s / opp
            if eid not in best_rate or r > best_rate[eid][0]:
                best_rate[eid] = (r, gi, s)
            if s >= MIN_SELECTIONS and len(tasks.get(eid, ())) >= MIN_TASKS:
                if eid not in best_rate_qualified or r > best_rate_qualified[eid][0]:
                    best_rate_qualified[eid] = (r, gi, s)

    n_iters = len(events)
    growth = n_entries / n_iters if n_iters else 0.0
    window = CATALOG_MAX / growth if growth else float("inf")

    qual = sorted(best_rate_qualified.items(), key=lambda kv: -kv[1][0])
    n_pass_all = sum(1 for _, v in qual if v[0] >= MIN_RATE)

    med_cand, max_cand = extractor_candidate_count(rd)

    out = {
        "run": rd.name,
        "iterations": n_iters,
        "l1_entries_created": n_entries,
        "catalog_growth_per_iter": round(growth, 4),
        "visibility_window_iters": round(window, 1),
        "extractor_candidates_median": med_cand,
        "extractor_candidates_max": max_cand,
        "skills_clearing_tasks_and_selections": len(qual),
        "of_those_clearing_rate_0.70": n_pass_all,
        "top10_best_qualified_rate": [
            {"entry_id": k, "best_rate": round(v[0], 4), "at_global_iter": v[1], "selections": v[2]}
            for k, v in qual[:10]
        ],
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(a)
        print("-" * 72)
