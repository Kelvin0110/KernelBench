"""Extract the exact extractor candidate set per iteration.

The extractor prompt renders its catalog as one line per candidate:

    - id=428 | challenge=Level 3 problem 22 | title=... | description=... | trigger=...

so the candidate set is recoverable exactly, rather than modelled. This matters
because ``rate``'s numerator can only grow while a skill is IN that set
(``read_l1_extractor_catalog`` returns ``entries[-50:]`` when governance is off,
memory_manager.py:798), while its denominator counts every iteration since the
skill was created.

Writes one JSON line per iteration:
    {"wall":..., "problem":..., "task_key":..., "iteration":...,
     "candidates":[ids], "selected":[ids]}
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_l2_gate import read_jsonl, task_key_for  # noqa: E402

# The extractor prompt renders TWO "- id=" blocks:
#   "## Already-active L2 standing rules (do NOT select these)" -> "- id=N | title=..."
#   "## L1 skill catalog (most recent skills - metadata only)"  -> "- id=N | challenge=... | title=..."
# Only the second is the selectable candidate set. Anchoring on "| challenge="
# excludes the standing block, which is NOT selectable and would otherwise
# inflate a promoted skill's visibility for the rest of the run.
CAND_RE = re.compile(r"^- id=(\d+) \| challenge=", re.M)
STANDING_RE = re.compile(r"^- id=(\d+) \| title=", re.M)


def build(run_dir: Path, out_path: Path) -> dict:
    rows: list[dict] = []
    ws_root = run_dir / "workspaces"
    n_prompt = 0
    for ws in sorted(ws_root.iterdir()) if ws_root.exists() else []:
        chat = ws / "chat_history.jsonl"
        if not chat.exists():
            continue
        per_iter: dict[int, dict] = {}
        for rec in read_jsonl(chat):
            it = rec.get("iteration")
            if it is None:
                continue
            it = int(it)
            slot = per_iter.setdefault(
                it,
                {
                    "wall": rec.get("wall_time_utc"),
                    "candidates": [],
                    "selected": [],
                    "standing": [],
                },
            )
            wall = rec.get("wall_time_utc")
            if wall and (slot["wall"] is None or wall < slot["wall"]):
                slot["wall"] = wall
            if rec.get("phase") != "extractor":
                continue
            n_prompt += 1
            blob = "\n".join(
                m.get("content") or ""
                for m in (rec.get("messages") or [])
                if isinstance(m, dict)
            )
            slot["candidates"] = CAND_RE.findall(blob)
            slot["standing"] = STANDING_RE.findall(blob)
            txt = rec.get("assistant_text")
            if txt:
                try:
                    ids = json.loads(txt).get("selected_entry_ids") or []
                    slot["selected"] = [str(x).strip() for x in ids if str(x).strip()]
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
        tk = task_key_for(ws.name)
        for it, slot in per_iter.items():
            rows.append(
                {
                    "wall": slot["wall"] or "",
                    "problem": ws.name,
                    "task_key": tk,
                    "iteration": it,
                    "candidates": slot["candidates"],
                    "selected": slot["selected"],
                    "standing": slot.get("standing") or [],
                }
            )
    rows.sort(key=lambda r: (r["wall"], r["problem"], r["iteration"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n_with_cand = sum(1 for r in rows if r["candidates"])
    return {
        "run": run_dir.name,
        "iterations": len(rows),
        "extractor_prompts": n_prompt,
        "iterations_with_candidates": n_with_cand,
        "out": str(out_path),
    }


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    for a in sys.argv[2:]:
        rd = Path(a)
        info = build(rd, out_dir / f"{rd.name}.visibility.jsonl")
        print(json.dumps(info))
