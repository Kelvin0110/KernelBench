"""Does "the extractor picked this skill" predict "this attempt got better"?

Every floor in the L2 gate is selection-based: min_tasks, min_selections and
min_rate all count how often an LLM that saw only titles and descriptions chose
the skill. The one outcome-based floor, min_new_bests, ships DISABLED. So the
tier promotes on popularity and merely records outcome.

That is only sound if selection predicts outcome. This measures it.

Design. For each skill, compare two conditional rates over iterations where the
skill was in the extractor's candidate list -- so both arms of the comparison
condition on the skill being *available* at that moment:

    p_sel   = P(new best | offered AND selected)
    p_unsel = P(new best | offered AND NOT selected)
    lift    = p_sel - p_unsel

This is observational, not causal: the extractor picks skills conditioned on the
current failure, so selection correlates with the state of the search. A large
positive lift would not prove the skills caused the improvement. But a lift at or
below zero WOULD show the gate is ranking on a signal that carries no outcome
information, which is the decision-relevant question.

``is_new_best`` is reconstructed exactly as governor.py:1241 defines it:
correct AND runtime valid AND NOT is_hack AND NOT excessive_speedup AND runtime
strictly better than the running best for that problem.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay_l2_gate import read_jsonl, task_key_for  # noqa: E402


def new_best_flags(run_dir: Path) -> dict[tuple[str, int], bool]:
    """(task_key, attempt) -> is_new_best, per governor.py:1241."""
    out: dict[tuple[str, int], bool] = {}
    data = json.loads((run_dir / "evolving_runs.json").read_text())
    for run in data.get("runs") or []:
        lvl, pid = run.get("level"), run.get("problem_id")
        if lvl is None or pid is None:
            continue
        tk = f"L{lvl}P{pid}"
        best_rt = -1.0
        for rec in run.get("records") or []:
            att = rec.get("attempt")
            ev = rec.get("evaluation") or {}
            rt = ev.get("runtime")
            try:
                rt = float(rt)
            except (TypeError, ValueError):
                rt = None
            excessive = bool((ev.get("metadata") or {}).get("excessive_speedup"))
            is_nb = (
                bool(ev.get("correct"))
                and rt is not None
                and rt >= 0
                and not bool(ev.get("is_hack"))
                and not excessive
                and (best_rt < 0 or rt < best_rt)
            )
            if is_nb:
                best_rt = rt
            if att is not None:
                out[(tk, int(att))] = is_nb
    return out


def analyse(run_dir: Path, vis_path: Path) -> dict:
    nb = new_best_flags(run_dir)
    rows = list(read_jsonl(vis_path))

    # per skill: offered/selected counts and new-best counts in each condition
    sel_n: dict[str, int] = {}
    sel_nb: dict[str, int] = {}
    uns_n: dict[str, int] = {}
    uns_nb: dict[str, int] = {}

    matched = 0
    for r in rows:
        key = (r["task_key"], int(r["iteration"]))
        if key not in nb:
            continue
        matched += 1
        hit = nb[key]
        chosen = set(r["selected"])
        for eid in r["candidates"]:
            if eid in chosen:
                sel_n[eid] = sel_n.get(eid, 0) + 1
                sel_nb[eid] = sel_nb.get(eid, 0) + int(hit)
            else:
                uns_n[eid] = uns_n.get(eid, 0) + 1
                uns_nb[eid] = uns_nb.get(eid, 0) + int(hit)

    lifts = []
    for eid in sel_n:
        ns, nu = sel_n[eid], uns_n.get(eid, 0)
        if ns < 10 or nu < 10:
            continue
        ps = sel_nb[eid] / ns
        pu = uns_nb.get(eid, 0) / nu
        lifts.append({"entry_id": eid, "n_sel": ns, "n_unsel": nu,
                      "p_sel": ps, "p_unsel": pu, "lift": ps - pu})

    # overall pooled rates
    tot_sel = sum(sel_n.values()) or 1
    tot_sel_nb = sum(sel_nb.values())
    tot_uns = sum(uns_n.values()) or 1
    tot_uns_nb = sum(uns_nb.values())

    # Stratify by attempt position. New bests concentrate early (the first
    # correct attempt is automatically a new best), so if selection behaviour
    # also varies with position the pooled figure is confounded. If every
    # stratum shows the same near-zero lift, it is not.
    BUCKETS = [(1, 5), (6, 15), (16, 30)]
    strata = []
    for lo, hi in BUCKETS:
        s_n = s_nb = u_n = u_nb = 0
        for r in rows:
            it = int(r["iteration"])
            if not (lo <= it <= hi):
                continue
            key = (r["task_key"], it)
            if key not in nb:
                continue
            hit = int(nb[key])
            chosen = set(r["selected"])
            for eid in r["candidates"]:
                if eid in chosen:
                    s_n += 1
                    s_nb += hit
                else:
                    u_n += 1
                    u_nb += hit
        if s_n and u_n:
            strata.append({
                "attempts": f"{lo}-{hi}",
                "p_sel": round(s_nb / s_n, 4),
                "p_unsel": round(u_nb / u_n, 4),
                "lift": round(s_nb / s_n - u_nb / u_n, 4),
                "n_sel": s_n,
            })

    # did the gate's chosen rules have better lift than the rest?
    promoted = {str(p["entry_id"]) for p in read_jsonl(run_dir / "l2_promotions.jsonl")
                if p.get("event") == "promote"}
    lp = [x["lift"] for x in lifts if x["entry_id"] in promoted]
    ln = [x["lift"] for x in lifts if x["entry_id"] not in promoted]

    return {
        "run": run_dir.name,
        "iterations_matched": matched,
        "skills_with_enough_data": len(lifts),
        "pooled_p_newbest_selected": round(tot_sel_nb / tot_sel, 4),
        "pooled_p_newbest_offered_not_selected": round(tot_uns_nb / tot_uns, 4),
        "pooled_lift": round(tot_sel_nb / tot_sel - tot_uns_nb / tot_uns, 4),
        "lift_by_attempt_bucket": strata,
        "per_skill_lift_median": round(st.median([x["lift"] for x in lifts]), 4) if lifts else None,
        "per_skill_lift_mean": round(st.mean([x["lift"] for x in lifts]), 4) if lifts else None,
        "per_skill_lift_frac_positive": round(
            sum(1 for x in lifts if x["lift"] > 0) / len(lifts), 3) if lifts else None,
        "promoted_n": len(lp),
        "promoted_lift_median": round(st.median(lp), 4) if lp else None,
        "not_promoted_lift_median": round(st.median(ln), 4) if ln else None,
        "top5_lift": sorted(lifts, key=lambda x: -x["lift"])[:5],
        "bottom5_lift": sorted(lifts, key=lambda x: x["lift"])[:5],
    }


if __name__ == "__main__":
    for a in sys.argv[1:]:
        rd = Path(a)
        vp = Path("out_l2") / f"{rd.name}.visibility.jsonl"
        if not vp.exists():
            print(f"{rd.name}: no visibility cache, run build_visibility_cache.py first")
            continue
        print(json.dumps(analyse(rd, vp), indent=2))
        print("-" * 72)
