"""Boundary-by-boundary sweep of candidate L2 promotion gates.

Uses the exact-visibility cache, so every quantity is measured rather than
modelled. The shipped gate is reproduced exactly (validated in
validate_replay.py), which is what makes the alternatives comparable.

Metrics per candidate at a boundary:
  selections   - times the extractor picked it
  offers       - times it was IN the extractor's candidate set
  tasks        - distinct task_keys it was picked on
  rate         - selections / (global_iter - created_at)   [SHIPPED]
  hit_rate     - selections / offers                        [PROPOSED]

``rate``'s denominator counts iterations in which the skill could not possibly
have been selected, because it had scrolled out of the newest-50 tail cap. Its
value therefore decays after the skill leaves the catalog, at a speed set by how
fast the arm mints new skills. ``hit_rate`` shares support with its numerator.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_l2_gate import load_created_at, load_entries, read_jsonl  # noqa: E402


def run_gate(rows, created_at, offset, gate, entries=None):
    """Replay one gate. Returns the promotion list."""
    sel: dict[str, int] = {}
    off: dict[str, int] = {}
    tasks: dict[str, set] = {}
    standing: list[dict] = []
    standing_ids: set[str] = set()
    promotions: list[dict] = []

    gi = offset
    for idx, r in enumerate(rows):
        gi += 1
        for eid in r["candidates"]:
            if eid in standing_ids:
                continue
            off[eid] = off.get(eid, 0) + 1
        for eid in r["selected"]:
            if eid in standing_ids:
                continue
            sel[eid] = sel.get(eid, 0) + 1
            tasks.setdefault(eid, set()).add(r["task_key"])

        is_last = (idx + 1 == len(rows)) or (rows[idx + 1]["problem"] != r["problem"])
        if not is_last:
            continue

        cands = []
        for eid, s in sel.items():
            if eid in standing_ids:
                continue
            o = off.get(eid, 0)
            opp = max(1, gi - created_at.get(eid, 0))
            cands.append(
                {
                    "entry_id": eid,
                    "selections": s,
                    "offers": o,
                    "tasks": len(tasks.get(eid, ())),
                    "opportunity": opp,
                    "rate": s / opp,
                    "hit_rate": s / max(1, o),
                    "entry": (entries or {}).get(eid, {}),
                }
            )
        for c in gate(cands, standing, gi):
            eid = c["entry_id"]
            if eid in standing_ids:
                continue
            standing_ids.add(eid)
            standing.append(c)
            promotions.append({**{k: v for k, v in c.items() if k != "entry"},
                               "promoted_at": gi, "task_key": r["task_key"]})
    return promotions


def make_gate(*, min_tasks=3, min_selections=50, min_rate=None, min_hit=None,
              max_entries=0, standing_cap=0, min_new_bests=0, nb=None,
              dedup=None, dedup_tau=None):
    """``max_entries`` reproduces the shipped PER-PASS cap.

    ``standing_cap`` is the proposed replacement: a bound on the size of the
    accumulated standing set, which is what CLAUDE.md 8.6 assumed
    ``--l2-max-entries`` already did.
    """

    def gate(cands, standing, gi):
        out = []
        for c in cands:
            if c["tasks"] < min_tasks or c["selections"] < min_selections:
                continue
            if min_rate is not None and c["rate"] < min_rate:
                continue
            if min_hit is not None and c["hit_rate"] < min_hit:
                continue
            if min_new_bests and (nb or {}).get(c["entry_id"], 0) < min_new_bests:
                continue
            out.append(c)
        key = "hit_rate" if min_hit is not None else "rate"
        for c in out:
            c["score"] = c[key] * math.log1p(c["tasks"]) * math.log1p((nb or {}).get(c["entry_id"], 0))
        out.sort(key=lambda c: (-c["score"], c["entry_id"]))
        if dedup is not None and dedup_tau is not None:
            kept = []
            for c in out:
                texts = [s["entry"] for s in standing] + [k["entry"] for k in kept]
                if any(dedup(c["entry"], t) >= dedup_tau for t in texts):
                    continue
                kept.append(c)
            out = kept
        if max_entries and max_entries > 0:
            out = out[:max_entries]
        if standing_cap and standing_cap > 0:
            room = max(0, standing_cap - len(standing))
            out = out[:room]
        return out

    return gate


def load_arm(run_dir: Path):
    vis = Path("out_l2") / f"{run_dir.name}.visibility.jsonl"
    rows = list(read_jsonl(vis))
    created_at = load_created_at(run_dir)
    ledger = json.loads((run_dir / "l1_skill_usage.json").read_text())
    offset = max(0, int(ledger.get("global_iteration") or 0) - len(rows))
    nb = {
        str(k): int((v or {}).get("new_best_attributions") or 0)
        for k, v in (ledger.get("skills") or {}).items()
    }
    return rows, created_at, offset, nb, load_entries(run_dir)


if __name__ == "__main__":
    arms = [Path(a) for a in sys.argv[1:]]
    loaded = {a.name: load_arm(a) for a in arms}

    print("=" * 78)
    print("SHIPPED GATE (tasks>=3, selections>=50, rate>=0.70)")
    print("=" * 78)
    for name, (rows, ca, off, nb, ent) in loaded.items():
        p = run_gate(rows, ca, off, make_gate(min_rate=0.70, nb=nb), ent)
        print(f"  {name[:46]:48s} promoted={len(p):2d}  {[x['entry_id'] for x in p]}")

    print()
    print("=" * 78)
    print("PROPOSED: hit_rate = selections/offers  (floor swept)")
    print("=" * 78)
    hdr = "  {:48s}".format("arm") + "".join(f"{h:>7}" for h in
          ["0.60", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"])
    print(hdr)
    for name, (rows, ca, off, nb, ent) in loaded.items():
        cells = []
        for h in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            p = run_gate(rows, ca, off, make_gate(min_hit=h, nb=nb), ent)
            cells.append(f"{len(p):>7d}")
        print(f"  {name[:46]:48s}" + "".join(cells))

    print()
    print("=" * 78)
    print("SHIPPED rate floor swept, for comparison")
    print("=" * 78)
    print("  {:48s}".format("arm") + "".join(f"{h:>7}" for h in
          ["0.50", "0.60", "0.65", "0.70", "0.75", "0.80"]))
    for name, (rows, ca, off, nb, ent) in loaded.items():
        cells = []
        for h in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
            p = run_gate(rows, ca, off, make_gate(min_rate=h, nb=nb), ent)
            cells.append(f"{len(p):>7d}")
        print(f"  {name[:46]:48s}" + "".join(cells))
