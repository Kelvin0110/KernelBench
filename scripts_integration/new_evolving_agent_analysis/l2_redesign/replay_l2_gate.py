"""Offline, boundary-by-boundary replay of the L2 promotion gate.

CLAUDE.md 8.8 says eligibility "is only replayable boundary by boundary". This
module does exactly that. The inputs are artifacts every arm already writes:

  workspaces/<problem>/chat_history.jsonl   phase=="extractor" ->
        {"selected_entry_ids": [...]}       per iteration, with a wall clock
  l1_skill_usage.json                       created_at_global_iter per skill
  shared_l1.jsonl                           entry content (for rendering / dedup)

``global_iteration`` advances exactly once per governor iteration
(gen3_stages.py: ``bump_global_iteration``), so numbering every iteration in
wall-clock order reconstructs it. Selections then accumulate exactly as
``record_l2_selection_evidence`` accumulates them.

Validated against ground truth in ``l2_promotions.jsonl`` -- see validate_replay.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


@dataclass
class IterationEvent:
    """One governor iteration: bumps global_iter, maybe selects skills."""

    wall: str
    problem: str  # workspace dir name, e.g. level_1_problem_100
    task_key: str  # L2 task key, e.g. L1P100
    iteration: int
    selected: list[str] = field(default_factory=list)


def task_key_for(workspace_name: str) -> str:
    """level_1_problem_100 -> L1P100 (the key record_l2_selection_evidence uses)."""
    parts = workspace_name.split("_")
    try:
        lvl = parts[parts.index("level") + 1]
        num = parts[parts.index("problem") + 1]
        return f"L{lvl}P{num}"
    except (ValueError, IndexError):
        return workspace_name


def load_iteration_events(run_dir: Path) -> list[IterationEvent]:
    """Every governor iteration in the run, in wall-clock order."""
    events: list[IterationEvent] = []
    ws_root = run_dir / "workspaces"
    if not ws_root.exists():
        return events
    for ws in sorted(ws_root.iterdir()):
        chat = ws / "chat_history.jsonl"
        if not chat.exists():
            continue
        # Group chat records by iteration; an iteration exists if ANY phase ran.
        per_iter: dict[int, dict[str, Any]] = {}
        for rec in read_jsonl(chat):
            it = rec.get("iteration")
            if it is None:
                continue
            it = int(it)
            slot = per_iter.setdefault(it, {"wall": rec.get("wall_time_utc"), "sel": []})
            wall = rec.get("wall_time_utc")
            if wall and (slot["wall"] is None or wall < slot["wall"]):
                slot["wall"] = wall
            if rec.get("phase") == "extractor":
                txt = rec.get("assistant_text")
                if txt:
                    try:
                        ids = json.loads(txt).get("selected_entry_ids") or []
                        slot["sel"] = [str(x).strip() for x in ids if str(x).strip()]
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass
        tk = task_key_for(ws.name)
        for it, slot in per_iter.items():
            events.append(
                IterationEvent(
                    wall=slot["wall"] or "",
                    problem=ws.name,
                    task_key=tk,
                    iteration=it,
                    selected=list(slot["sel"]),
                )
            )
    events.sort(key=lambda e: (e.wall, e.problem, e.iteration))
    return events


def load_created_at(run_dir: Path) -> dict[str, int]:
    p = run_dir / "l1_skill_usage.json"
    if not p.exists():
        return {}
    state = json.loads(p.read_text())
    return {
        str(k): int((v or {}).get("created_at_global_iter") or 0)
        for k, v in (state.get("skills") or {}).items()
    }


def load_new_best_totals(run_dir: Path) -> dict[str, int]:
    """Final new_best_attributions per skill (no per-boundary record exists)."""
    p = run_dir / "l1_skill_usage.json"
    if not p.exists():
        return {}
    state = json.loads(p.read_text())
    return {
        str(k): int((v or {}).get("new_best_attributions") or 0)
        for k, v in (state.get("skills") or {}).items()
    }


def load_entries(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in read_jsonl(run_dir / "shared_l1.jsonl"):
        eid = str(rec.get("entry_id", "")).strip()
        if eid:
            out[eid] = rec
    return out


def global_iter_offset(run_dir: Path, n_events: int) -> int:
    """Iterations that bumped the real counter but left no chat records.

    Both completed L2 arms were resumed; the discarded pre-resume iterations
    still advanced ``global_iteration``. ``created_at`` is stored on the real
    scale, so a reconstruction numbering surviving iterations 1..N must be
    shifted by this constant or ``opportunity`` is wrong.

    Verified exactly on the terra arm: offset 26 reproduces all four recorded
    promotions (id, boundary, selections, tasks, opportunity, rate) with zero
    error. See calibrate_offset.py.
    """
    p = run_dir / "l1_skill_usage.json"
    if not p.exists():
        return 0
    real_final = int(json.loads(p.read_text()).get("global_iteration") or 0)
    return max(0, real_final - n_events)


def load_truth(run_dir: Path) -> list[dict]:
    return [
        r
        for r in read_jsonl(run_dir / "l2_promotions.jsonl")
        if r.get("event") == "promote"
    ]


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


@dataclass
class Evidence:
    total_selections: int = 0
    tasks: set[str] = field(default_factory=set)
    new_bests: int = 0


Gate = Callable[[list[dict], list[dict], int, dict], list[dict]]


def replay(
    run_dir: Path,
    gate: Gate,
    *,
    created_at: dict[str, int] | None = None,
    new_best_hook: Callable[[str, int], int] | None = None,
) -> dict[str, Any]:
    """Walk iterations in order; run ``gate`` at every task boundary.

    ``gate(candidates, standing, global_iter, ctx)`` returns the candidate dicts
    to promote. Promoted ids leave the candidate pool (promotion removes them
    from the extractor catalog) and their evidence is frozen, exactly as
    l2_promotion.py does.
    """
    events = load_iteration_events(run_dir)
    created_at = created_at if created_at is not None else load_created_at(run_dir)
    entries = load_entries(run_dir)

    ev: dict[str, Evidence] = {}
    standing: list[dict] = []
    standing_ids: set[str] = set()
    promotions: list[dict] = []
    boundaries: list[dict] = []

    gi = global_iter_offset(run_dir, len(events))
    for idx, event in enumerate(events):
        gi += 1
        for eid in event.selected:
            if eid in standing_ids:
                continue  # promoted skills leave the catalog
            rec = ev.setdefault(eid, Evidence())
            rec.total_selections += 1
            rec.tasks.add(event.task_key)

        # Task boundary = last iteration of this problem.
        is_last = (idx + 1 == len(events)) or (events[idx + 1].problem != event.problem)
        if not is_last:
            continue

        candidates = []
        for eid, rec in ev.items():
            if eid in standing_ids:
                continue
            born = created_at.get(eid, 0)
            opportunity = max(1, gi - born)
            nb = rec.new_bests
            if new_best_hook is not None:
                nb = new_best_hook(eid, gi)
            candidates.append(
                {
                    "entry_id": eid,
                    "total_selections": rec.total_selections,
                    "tasks": len(rec.tasks),
                    "opportunity": opportunity,
                    "rate": rec.total_selections / opportunity,
                    "new_bests": nb,
                    "entry": entries.get(eid, {}),
                }
            )

        chosen = gate(candidates, standing, gi, {"task_key": event.task_key})
        for cand in chosen:
            eid = cand["entry_id"]
            if eid in standing_ids:
                continue
            standing_ids.add(eid)
            standing.append(cand)
            promotions.append(
                {
                    **{k: v for k, v in cand.items() if k != "entry"},
                    "promoted_at_global_iter": gi,
                    "task_key": event.task_key,
                }
            )
        boundaries.append(
            {
                "global_iter": gi,
                "task_key": event.task_key,
                "problem": event.problem,
                "standing": len(standing),
                "n_candidates": len(candidates),
            }
        )

    return {
        "promotions": promotions,
        "standing": standing,
        "boundaries": boundaries,
        "final_global_iter": gi,
        "n_events": len(events),
    }
