"""Compare the shipped L2 rate metric against a visibility-normalized one.

Shipped:   rate     = selections / (global_iter - created_at)
Proposed:  hit_rate = selections / (iterations the skill was an extractor candidate)

The numerator can only grow while the skill is IN the extractor's candidate set;
the shipped denominator counts every iteration since creation regardless. Once a
skill scrolls out of the newest-50 tail cap, its numerator freezes while its
denominator keeps ticking, so its rate decays monotonically toward zero. How fast
that happens is set by how quickly the arm mints new L1 skills -- an arm-level
property, not a property of the skill.

hit_rate has no such decay: both numerator and denominator are supported on the
same set of iterations.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_l2_gate import load_created_at, read_jsonl  # noqa: E402

MIN_TASKS, MIN_SELECTIONS, MIN_RATE = 3, 50, 0.70


def analyse(vis_path: Path, run_dir: Path) -> dict:
    rows = list(read_jsonl(vis_path))
    created_at = load_created_at(run_dir)
    ledger = json.loads((run_dir / "l1_skill_usage.json").read_text())
    offset = max(0, int(ledger.get("global_iteration") or 0) - len(rows))

    sel: dict[str, int] = {}
    vis: dict[str, int] = {}
    tasks: dict[str, set] = {}
    # peak of each metric over the run, evaluated at task boundaries
    peak_rate: dict[str, float] = {}
    peak_hit: dict[str, float] = {}
    qualified: set[str] = set()  # cleared tasks>=3 and selections>=50 at some boundary

    cat_sizes: list[int] = []
    pick_counts: list[int] = []

    gi = offset
    for idx, r in enumerate(rows):
        gi += 1
        cands = r["candidates"]
        if cands:
            cat_sizes.append(len(cands))
            pick_counts.append(len(r["selected"]))
        for eid in cands:
            vis[eid] = vis.get(eid, 0) + 1
        for eid in r["selected"]:
            sel[eid] = sel.get(eid, 0) + 1
            tasks.setdefault(eid, set()).add(r["task_key"])

        is_last = (idx + 1 == len(rows)) or (rows[idx + 1]["problem"] != r["problem"])
        if not is_last:
            continue
        for eid, s in sel.items():
            opp = max(1, gi - created_at.get(eid, 0))
            rate = s / opp
            hit = s / max(1, vis.get(eid, 0))
            peak_rate[eid] = max(peak_rate.get(eid, 0.0), rate)
            peak_hit[eid] = max(peak_hit.get(eid, 0.0), hit)
            if s >= MIN_SELECTIONS and len(tasks.get(eid, ())) >= MIN_TASKS:
                qualified.add(eid)

    mean_cat = st.mean(cat_sizes) if cat_sizes else 0
    mean_pick = st.mean(pick_counts) if pick_counts else 0
    chance = (mean_pick / mean_cat) if mean_cat else 0

    q = sorted(qualified, key=lambda e: -peak_hit.get(e, 0))
    return {
        "run": run_dir.name,
        "iterations": len(rows),
        "l1_entries": sum(1 for _ in read_jsonl(run_dir / "shared_l1.jsonl")),
        "mean_catalog_shown": round(mean_cat, 1),
        "mean_picks": round(mean_pick, 2),
        "chance_rate": round(chance, 4),
        "median_visible_iters": st.median(vis.values()) if vis else 0,
        "qualified_skills": len(qualified),
        "shipped_rate": {
            "floor": MIN_RATE,
            "max_peak_among_qualified": round(max((peak_rate[e] for e in qualified), default=0), 4),
            "n_clearing_floor": sum(1 for e in qualified if peak_rate[e] >= MIN_RATE),
        },
        "hit_rate": {
            "max_peak_among_qualified": round(max((peak_hit[e] for e in qualified), default=0), 4),
            "median_peak_among_qualified": round(st.median([peak_hit[e] for e in q]) if q else 0, 4),
            "deciles": [round(x, 3) for x in st.quantiles([peak_hit[e] for e in q], n=10)] if len(q) > 9 else [],
        },
        "top10": [
            {
                "id": e,
                "selections": sel[e],
                "visible_iters": vis.get(e, 0),
                "peak_hit_rate": round(peak_hit[e], 4),
                "peak_shipped_rate": round(peak_rate[e], 4),
                "lift_over_chance": round(peak_hit[e] / chance, 2) if chance else 0,
            }
            for e in q[:10]
        ],
    }


if __name__ == "__main__":
    for run_dir in sys.argv[1:]:
        rd = Path(run_dir)
        vp = Path("out_l2") / f"{rd.name}.visibility.jsonl"
        print(json.dumps(analyse(vp, rd), indent=2))
        print("-" * 72)
